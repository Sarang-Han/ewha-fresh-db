# Ewha Fresh RAG 리팩토링 지시서

> 목표:  
> **정형 데이터(CSV)는 통째로 LLM에 주고**,  
> **비정형 데이터(Markdown)는 기존 RAG로 처리하는 이원 구조**로 백엔드를 전면 개편한다.  
> 별도의 “엔진/DB”는 만들지 않고, **라우터 + 프롬프트 설계**에 집중한다.

---

## 1. 전체 아키텍처 개요

### 1.1. 기존 구조 (요약)

- `embed_data.py`  
  - Markdown + CSV를 모두 청킹 후 벡터 DB(hybrid: BM25 + bge)에 저장
- `/ask` 핸들러
  - 입력:  
    ```json
    {
      "session_id": "string",
      "message": "질문",
      "user_grade": 4,
      "user_major": "컴퓨터공학"
    }
    ```
  - 공통 RAG 파이프라인 실행:
    1. Hybrid 검색 (Markdown + CSV 섞여 있음)
    2. 검색 결과 일부를 LLM 컨텍스트로 전달
    3. LLM이 답변 생성
  - 출력:
    ```json
    {
      "session_id": "string",
      "answer": "...",
      "sources": [...]
    }
    ```

### 1.2. 개편 후 구조

- **핵심 변경점**
  1. **질문 라우터(Intent Router)를 추가**하여, 질문을 먼저 분류:
     - `SCHEDULE` : 수강신청/학사일정/날짜/기간 조회
     - `GUIDE` : 휴학, 복학, 졸업 요건, 제도 설명 등
     - `OTHER` : 잡담 등(필요 시)
  2. **CSV는 벡터 DB에서 제거**하고,  
     - 일정 관련 질문(SCHEDULE)에서는 **CSV 전체 텍스트를 LLM에 직접 컨텍스트로 제공**
  3. Markdown RAG는 **규정/설명(GUIDE) 질문에만 사용**한다.

- **새 파이프라인**

```text
  User Request
      ↓
  Intent Router (LLM 기반)
      ↓
   ┌───────────────┬────────────────┐
   │ SCHEDULE      │ GUIDE          │
   │ (일정/수강)   │ (규정/제도)    │
   └───────────────┴────────────────┘
      ↓                ↓
  Full CSV + LLM   Markdown RAG + LLM
      ↓                ↓
           최종 Answer
```


2. 요청/응답 스키마

2.1. 요청

기존 스키마 유지:

{
  "session_id": "string",
  "message": "컴퓨터공학과 4학년의 수강신청 일정 알려주세요",
  "user_grade": 4,
  "user_major": "컴퓨터공학"
}

2.2. 응답

역시 기존 형식 유지:

{
  "session_id": "string",
  "answer": "자연어 답변...",
  "sources": [
    "official/수강신청/25-2_course_registration.csv",
    "official/학사일정/academic_calendar.csv"
  ]
}

	•	SCHEDULE:
	•	sources는 고정적으로 관련 CSV 파일 경로를 넣는다.
	•	GUIDE:
	•	실제 RAG에서 참조한 Markdown 파일 경로를 넣는다.

⸻

3. Intent Router 설계

3.1. Intent 타입 정의

코드 상에서 사용할 enum/상수:

class Intent(str, Enum):
    SCHEDULE = "SCHEDULE"  # 일정/수강신청, 기간, 날짜
    GUIDE = "GUIDE"        # 규정/절차/제도 설명
    OTHER = "OTHER"        # 잡담/기타 (필요 시)

3.2. 라우터 함수 시그니처

def classify_intent(message: str, user_grade: int | None = None, user_major: str | None = None) -> Intent:
    ...

3.3. 라우터 구현 전략
	1.	간단 키워드 기반 Pre-check (빠른 분기)
	•	아래 단어가 포함되면 우선 SCHEDULE로 가정:
	•	"수강신청", "장바구니", "수강 정정", "수강정정", "수강철회", "철회",
"학사일정", "등록기간", "등록 기간", "개강", "종강", "폐강", "수강 기간", "신청 기간"
	2.	애매한 경우는 LLM에 위임

3.4. LLM 라우터 프롬프트 템플릿

[시스템 역할]
당신은 이화여자대학교 학사 챗봇의 질문 분류기입니다.
사용자의 질문을 보고 아래 세 가지 중 하나로만 분류하세요.

- SCHEDULE: 수강신청, 장바구니, 학사일정, 등록 기간, 정정/철회 기간, 개강/종강 날짜 등
           "언제", "기간", "날짜", "시간"을 묻는 일정/데드라인 관련 질문.
- GUIDE   : 휴학, 복학, 자퇴, 전과, 복수전공, 졸업요건, 성적, 재수강, 학점 관련 규정/절차/방법.
- OTHER   : 위 두 가지에 해당하지 않는 잡담, 인사, 일반 질문 등.

반드시 아래 형식으로만 대답하세요.

예시:
SCHEDULE
GUIDE
OTHER

[사용자 정보]
학년: {{user_grade}}
전공: {{user_major}}

[질문]
{{message}}

	•	이 프롬프트를 사용해 SCHEDULE / GUIDE / OTHER 문자열만 받는다.


4. SCHEDULE 파이프라인 (CSV 전체 + LLM)

원칙:
	•	수강신청/학사일정 CSV는 벡터 DB에 넣지 않는다.
	•	일정 관련 질문은 이 CSV들을 통째로 텍스트로 넣고 LLM에게 “필터링 + 정리”를 시킨다.

4.1. CSV 로딩 전략

서버 시작 시, CSV를 텍스트로 로드해 전역/싱글톤으로 보관:

# pseudo-code

```
def load_csv_texts() -> dict[str, str]:
    base_dir = "data/official"  # 실제 경로 맞게 수정
    with open(f"{base_dir}/수강신청/25-2_course_registration.csv", encoding="utf-8") as f:
        course_registration_text = f.read()
    with open(f"{base_dir}/학사일정/academic_calendar.csv", encoding="utf-8") as f:
        academic_calendar_text = f.read()

    return {
        "course_registration": course_registration_text,
        "academic_calendar": academic_calendar_text,
    }
```
	•	필요 시 여러 학기/연도 CSV가 있다면, key를 term 기준으로 확장할 수 있다.

4.2. SCHEDULE 답변 함수 시그니처

def answer_schedule_question(
    message: str,
    user_grade: int | None,
    user_major: str | None,
    course_registration_csv: str,
    academic_calendar_csv: str,
) -> tuple[str, list[str]]:
    """
    return: (answer_text, sources_list)
    """

4.3. SCHEDULE LLM 프롬프트 템플릿

중요:
	•	CSV 텍스트는 그대로 넣되, LLM이 “정확히 필터링”하도록 규칙을 명확히 적는다.
	•	코파일럿이 그대로 쓸 수 있도록, 템플릿 문자열 형식으로 작성한다.

SCHEDULE_PROMPT_TEMPLATE = """
당신은 이화여자대학교 학사/수강신청 안내를 담당하는 AI 비서입니다.
아래에 주어진 "수강신청 일정표 CSV"와 "학사일정 CSV" 전체를 보고,
[사용자 정보]와 [질문]에 해당하는 일정을 모두 찾아서 정리해 주세요.

[역할]
- 당신의 역할은 "데이터 분석가 + 학사담당자" 입니다.
- CSV 데이터를 먼저 정확히 읽고, 조건에 맞는 행(row)만 필터링한 뒤,
  학생이 이해하기 쉬운 한국어 문장으로 설명해야 합니다.

[사용자 정보]
- 학년(grade): {user_grade}
- 전공(major): {user_major}

[질문]
{message}

[수강신청 일정표 CSV 전체]
```csv
{course_registration_csv}

[학사일정 CSV 전체]

{academic_calendar_csv}

[필터링/답변 규칙]
	1.	먼저 수강신청 CSV에서 [사용자 정보]와 [질문]에 해당하는 모든 행을 찾으세요.
	•	학년 조건:
	•	“4학년” 질문이면 4학년 또는 4학년 이상/이하로 명시된 행을 포함합니다.
	•	“3~4학년”, “전체 재학생” 등 범위로 표시된 행도 포함합니다.
	•	전공 조건:
	•	사용자의 전공(예: 컴퓨터공학과)을 정확히 포함하는 전공 대상 행을 우선 포함합니다.
	•	“전체”, “전 재학생”, “전 학년” 등 공통 대상 행도 함께 포함합니다.
	•	이벤트 종류:
	•	우선수강신청, 본수강신청, 학년별 수강신청, 전체학년 수강신청,
장바구니, 수강 정정(확인 및 변경), 수강철회 등
수강신청과 직접 관련된 모든 이벤트를 포함합니다.
	2.	학사일정 CSV는 보조적으로 사용합니다.
	•	개강일, 종강일, 등록기간 등, 질문과 관련이 있고 학생에게 도움이 될 만한 일정이 있으면 함께 알려줍니다.
	3.	답변 형식:
	•	시간 순으로 정렬해서 bullet list로 정리합니다.
	•	각 항목에 아래 정보를 포함합니다.
	•	일정 이름
	•	대상(학년, 전공)
	•	기간(날짜와 시간)
	•	간단한 설명(필요 시)
	•	예시 형식:
컴퓨터공학과 4학년 학생의 2025학년도 2학기 수강신청 관련 주요 일정은 다음과 같습니다.
	•	우선수강신청 (컴퓨터공학과 전공생)
	•	기간: 2025년 8월 4일(월) 13:00 ~ 16:00
	•	대상: 컴퓨터공학과 주/복수전공 2~4학년
	•	4학년 본수강신청
	•	기간: 2025년 8월 5일(화) 09:00 ~ 12:00
	•	대상: 4학년 이상 재학생
	4.	사용자의 질문에 불필요한 다른 학년/전공의 이벤트는 설명하지 않습니다.
	•	다만, 모든 학생 공통 일정(장바구니, 수강정정, 수강철회 등)은 함께 보여줍니다.
	5.	모르는 정보가 있으면 추측하지 말고,
“제공된 CSV에서 해당 정보를 찾을 수 없다”고 명시하세요.

이제 위 규칙에 따라 답변을 작성해 주세요.
“””

### 4.4. SCHEDULE 파이프라인 통합 예시 (의사 코드)

```python
def answer_schedule_question(...):
    prompt = SCHEDULE_PROMPT_TEMPLATE.format(
        user_grade=user_grade,
        user_major=user_major,
        message=message,
        course_registration_csv=course_registration_csv,
        academic_calendar_csv=academic_calendar_csv,
    )

    llm_output = call_llm(prompt)

    answer = llm_output  # 후처리 필요 시 추가

    sources = [
        "official/수강신청/25-2_course_registration.csv",
        "official/학사일정/academic_calendar.csv",
    ]
    return answer, sources


⸻

5. GUIDE 파이프라인 (Markdown RAG)

규정/절차/제도 설명 질문은 기존 RAG 파이프라인을 사용하되,
인덱스에서 CSV를 제거하고, Markdown 위주로 검색한다.

5.1. 인덱스 구성 변경
	•	embed_data.py에서 다음을 수행:
	•	official/학사안내/*.md 등 Markdown 파일만 임베딩
	•	25-2_course_registration.csv, academic_calendar.csv 등 CSV 파일은 벡터 DB에서 제외

5.2. GUIDE 답변 함수 시그니처

def answer_guide_question(
    message: str,
    user_grade: int | None,
    user_major: str | None,
) -> tuple[str, list[str]]:
    ...

5.3. GUIDE LLM 프롬프트 템플릿 (예시)

GUIDE_PROMPT_TEMPLATE = """
당신은 이화여자대학교 학사제도 안내를 담당하는 AI 비서입니다.
아래는 학사안내 공식 문서에서 검색된 관련 내용입니다.
이 내용을 기반으로, 학생의 질문에 정확하고 친절하게 답변해 주세요.

[사용자 정보]
- 학년: {user_grade}
- 전공: {user_major}

[질문]
{message}

[검색된 참고 문서]
{context}

[답변 지침]
1. 반드시 참고 문서의 내용을 우선으로 답변합니다.
2. 규정/조건/예외 사항이 있다면 함께 설명합니다.
3. 확실하지 않은 내용은 임의로 만들어내지 말고,
   "제공된 학사안내 문서에 해당 내용이 명시되어 있지 않습니다"라고 말합니다.
4. 신입생도 이해하기 쉬운 자연스러운 한국어로 설명합니다.
"""

	•	context에는 RAG로 가져온 Markdown 청크들을 묶어서 넣는다.

⸻

6. /ask 엔드포인트 통합 로직

최종적으로, 백엔드의 /ask 핸들러는 다음 흐름을 따른다.

def answer(request: AskRequest) -> AskResponse:
    # 1. Intent 분류
    intent = classify_intent(
        message=request.message,
        user_grade=request.user_grade,
        user_major=request.user_major,
    )

    # 2. Intent별 파이프라인 분기
    if intent == Intent.SCHEDULE:
        answer_text, sources = answer_schedule_question(
            message=request.message,
            user_grade=request.user_grade,
            user_major=request.user_major,
            course_registration_csv=CSV_TEXTS["course_registration"],
            academic_calendar_csv=CSV_TEXTS["academic_calendar"],
        )

    elif intent == Intent.GUIDE:
        answer_text, sources = answer_guide_question(
            message=request.message,
            user_grade=request.user_grade,
            user_major=request.user_major,
        )

    else:
        # 필요 시 OTHER 처리 (간단한 인사/에코 등)
        answer_text = "학사/수강신청과 관련된 질문을 해주시면 도와드릴 수 있어요 :)"
        sources = []

    # 3. 응답 포맷팅
    return AskResponse(
        session_id=request.session_id,
        answer=answer_text,
        sources=sources,
    )


⸻

7. 코파일럿을 위한 TODO 체크리스트

아래 항목들을 순서대로 수행하도록 코파일럿에게 맡기면 된다.

	1.	CSV를 벡터 DB 인덱싱에서 제거
	•	embed_data.py 수정:
	•	Markdown(.md)만 임베딩 대상에 포함
	•	25-2_course_registration.csv, academic_calendar.csv 등 CSV는 제외
	2.	Intent enum 및 라우터 구현
	•	Intent enum 정의 (SCHEDULE, GUIDE, OTHER)
	•	classify_intent(message, user_grade, user_major) 함수 구현:
	•	키워드 기반 Pre-check + LLM 라우터 프롬프트 사용
	3.	CSV 로딩 로직 추가
	•	서버 시작 시 CSV 파일들을 텍스트로 읽어 전역/싱글톤에 보관
	•	예: CSV_TEXTS = load_csv_texts()
	4.	SCHEDULE 파이프라인 구현
	•	answer_schedule_question(...) 함수 생성
	•	SCHEDULE_PROMPT_TEMPLATE 문자열 그대로 사용
	•	LLM 호출 후 answer, sources 반환
	5.	GUIDE 파이프라인 정리
	•	answer_guide_question(...) 함수 생성
	•	기존 RAG 로직을 이 함수 안으로 이동/정리
	•	GUIDE_PROMPT_TEMPLATE 적용
	•	검색 대상은 Markdown 인덱스만 사용
	6.	/ask 엔드포인트 리팩토링
	•	/ask 핸들러에서:
	1.	classify_intent 호출
	2.	Intent에 따라 answer_schedule_question 또는 answer_guide_question 호출
	3.	응답(JSON)에 answer, sources 채워서 반환

⸻

이 지시서의 목표는:
	•	“수강신청/일정 질문”에서 현재의 RAG+청킹 구조를 완전히 제거하고,
	•	“CSV 전체를 LLM에게 통으로 넘겨서 필터링/정리하게 만드는 구조”로 바꾸는 것,
	•	그 동시에 Markdown RAG는 규정/제도 설명용으로만 깔끔하게 분리하는 것이다.

위 내용을 기준으로 코파일럿이 구조를 전면 개편하도록 리팩토링을 진행하면 된다.

