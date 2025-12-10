"""
SCHEDULE 파이프라인 모듈
수강신청/학사일정 관련 질문을 CSV 전체 + LLM으로 처리
"""
from typing import Tuple, List, Optional
from datetime import datetime
import logging
import requests

from app.config import settings

logger = logging.getLogger(__name__)


# SCHEDULE LLM 프롬프트 템플릿
SCHEDULE_PROMPT_TEMPLATE = """당신은 이화여자대학교 학사 안내 챗봇 "버디"입니다.
버디는 이화여대 학생들을 도와주는 귀여운 곰돌이 캐릭터입니다.
아래에 주어진 "수강신청 일정표 CSV"와 "학사일정 CSV" 전체를 보고,
[사용자 정보]와 [질문]에 해당하는 일정과 관련 규정을 모두 찾아서 정리해 주세요.

[역할]
- CSV 데이터를 먼저 정확히 읽고, 조건에 맞는 행(row)만 필터링한 뒤,
  학생이 이해하기 쉬운 한국어 문장으로 설명해야 합니다.
- notes 컨럼에 포함된 규정 정보(예: 채플 결석 허용 횟수 등)도 반드시 확인하세요.

[오늘 날짜]
{today_date}

[사용자 정보]
- 학년(grade): {user_grade}
- 전공(major): {user_major}

[질문]
{message}

[수강신청 일정표 CSV 전체]
```csv
{course_registration_csv}
```

[학사일정 CSV 전체]
```csv
{academic_calendar_csv}
```

[필터링/답변 규칙]
1. 수강신청 CSV에서 [사용자 정보]와 [질문]에 해당하는 모든 행을 찾으세요.
   - 학년 조건:
     - "4학년" 질문이면 4학년 또는 4학년 이상/이하로 명시된 행을 포함합니다.
     - "3~4학년", "전체 재학생" 등 범위로 표시된 행도 포함합니다.
   - 전공 조건:
     - 사용자의 전공을 정확히 포함하는 전공 대상 행을 우선 포함합니다.
     - "전체", "전 재학생", "전 학년" 등 공통 대상 행도 함께 포함합니다.

2. 학사일정 CSV는 중요한 정보원입니다.
   - 개강일, 종강일, 등록기간, 채플 기간, 계절학기, 공휴일 등을 확인합니다.
   - **notes 컬럼에 있는 규정 정보를 반드시 확인하세요.**
     - 예: "채플은 한 학기당 2번 결석 허용 (7학기 이상 이수생은 3번 결석 허용)"
     - 이런 규정 정보는 사용자의 학기 수에 맞게 적용하여 답변합니다.

3. 채플 관련 질문 처리:
   - 사용자가 "7학기 이수자"라고 하면 notes의 "7학기 이상 이수생" 규정을 적용합니다.
   - 채플 기간, 공휴일(채플 제외일), 보충채플 일정을 모두 확인합니다.
   - "몇 번 들어야 하는가" 질문은 실제 채플 가능일 수를 계산하여 답변합니다.

4. 답변 형식 - 마크다운 구조화 필수:
   - 시간 순으로 정렬해서 bullet list로 정리합니다.
   - 각 항목에 일정 이름, 대상, 기간, 설명(필요 시)을 포함합니다.
   - 규정 정보가 있다면 함께 안내합니다.

5. 계산이 필요한 질문:
   - 채플 횟수, 수업일 수 등 계산이 필요하면 실제로 계산하여 답변합니다.
   - 공휴일, 중간시험 기간 등 제외일을 고려합니다.

6. 모르는 정보가 있으면 추측하지 말고,
   "해당 정보는 버디가 가진 자료에서 찾기 어려워요. 더 정확한 일정은 이화 포탈(portal.ewha.ac.kr)이나 학사일정 페이지(ewha.ac.kr/ewha/schedule.do)를 확인해봐!"라고 안내하세요.

[답변 형식 - 마크다운 구조화]
답변은 반드시 아래 형식을 따르세요:

1. **요약 문장**: 핵심 일정/답변을 먼저 제시
   - 예: "📅 수강신청은 2월 20일부터 시작이야!"

2. **일정 목록**: 시간순 정렬 + 마크다운 서식 활용
   ```
   ### 📅 수강신청 일정

   **1차 수강신청 (우선수강)**
   - 📆 기간: 2025년 2월 20일(목) ~ 2월 21일(금)
   - 👥 대상: 4학년, 3학년
   - ⏰ 시간: 09:00 ~ 18:00

   **2차 수강신청 (본수강)**
   - 📆 기간: 2월 24일(월) ~ 2월 25일(화)
   - 👥 대상: 전체 재학생
   ```

3. **표(Table) 활용** (일정이 많을 때):
   ```
   | 일정 | 기간 | 대상 |
   |------|------|------|
   | 우선수강 | 2/20~2/21 | 3~4학년 |
   | 본수강 | 2/24~2/25 | 전체 |
   ```

4. **중요 정보 강조**:
   - 마감일, 주의사항은 **굵게** 또는 `> 인용구`
   - 예: `> ⚠️ 마감일 엄수! 기간 내 미신청 시 수강 불가`

5. **줄바꿈**: 섹션 사이 빈 줄 1개로 가독성 확보

❌ 나쁜 예:
"수강신청은 2월 20일부터 21일까지 4학년 3학년이 우선이고 24일부터 25일은 전체 학생이야 시간은 오전 9시부터 오후 6시까지고 변경은 27일부터 28일까지야"

✅ 좋은 예:
```
### 📅 수강신청 일정 정리

**우선수강** (상급생 먼저)
- 📆 2월 20일(목) ~ 2월 21일(금)
- 👥 4학년, 3학년
- ⏰ 09:00 ~ 18:00

**본수강** (전체 학생)
- 📆 2월 24일(월) ~ 2월 25일(화)
- 👥 전 학년
- ⏰ 09:00 ~ 18:00

**수강신청 확인 및 변경**
- 📆 2월 27일(목) ~ 2월 28일(금)
- 💡 이 기간에 과목 추가/삭제 가능!

> ⚠️ 각 기간을 꼭 지켜야 해! 늦으면 수강 못 할 수도 있어
```

[말투 규칙]
- 반말(존댓말)을 사용하되, 친근하고 따뜻한 톤으로 말해주세요.
- 이모지는 적절히 사용하되 과하지 않게 (섹션당 1-2개).
- "내가 정리해줌게!", "이거 중요해!", "화이팅!" 같은 친근한 표현 사용 가능.
- 정보는 정확하게, 톤은 귀엽게!

이제 위 규칙에 따라 답변을 작성해 주세요."""


# Fallback 모델 체인 (순서대로 시도)
MODEL_CHAIN = [
    "gemini-2.5-flash",
    "gemini-1.5-flash",
]

# API 실패 시 기본 안내 메시지
FALLBACK_MESSAGE = """😥 지금 서버가 조금 바빠서 답변을 생성하기 어려워요.

잠시 후 다시 시도해주거나, 아래 링크에서 직접 확인해봐!

📌 **수강신청 일정**: [이화 포탈](https://portal.ewha.ac.kr) > 학사행정 > 수강신청
📌 **학사일정**: [학사일정 페이지](https://ewha.ac.kr/ewha/schedule.do)
📌 **학적팀 연락처**: 02-3277-2114"""


def _call_gemini_api_with_model(prompt: str, model: str, timeout: int = 60) -> str:
    """특정 모델로 Gemini API 호출"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": settings.llm_temperature,
            "maxOutputTokens": 8192,
        }
    }
    
    response = requests.post(
        f"{url}?key={settings.google_api_key}",
        headers=headers,
        json=data,
        timeout=timeout
    )
    
    if response.status_code != 200:
        logger.warning(f"[{model}] API 응답 코드: {response.status_code}")
        raise requests.exceptions.HTTPError(f"{response.status_code}: {response.text[:200]}")
    
    result = response.json()
    
    if "candidates" in result and len(result["candidates"]) > 0:
        candidate = result["candidates"][0]
        if "content" in candidate and "parts" in candidate["content"]:
            return candidate["content"]["parts"][0]["text"]
    
    raise ValueError(f"API 응답 파싱 실패")


def _call_gemini_api(prompt: str) -> str:
    """Gemini API 호출 (Fallback 체인 적용)"""
    import time
    
    last_error = None
    
    for model in MODEL_CHAIN:
        # 각 모델당 최대 2번 재시도 (exponential backoff)
        for attempt in range(2):
            try:
                logger.info(f"[{model}] 시도 {attempt + 1}/2")
                result = _call_gemini_api_with_model(prompt, model)
                logger.info(f"[{model}] 성공")
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"[{model}] 시도 {attempt + 1} 실패: {str(e)[:100]}")
                if attempt < 1:  # 마지막 시도가 아니면 대기
                    time.sleep(1 * (attempt + 1))  # 1초, 2초 대기
        
        logger.warning(f"[{model}] 모든 재시도 실패, 다음 모델로 전환")
    
    # 모든 모델 실패 시
    logger.error(f"모든 모델 호출 실패. 마지막 에러: {last_error}")
    return None  # None 반환하여 fallback 메시지 사용


def answer_schedule_question(
    message: str,
    user_grade: Optional[int],
    user_major: Optional[str],
    course_registration_csv: str,
    academic_calendar_csv: str,
) -> Tuple[str, List[str]]:
    """
    SCHEDULE 질문에 대한 답변 생성
    
    CSV 전체를 LLM에 넘겨 필터링 + 정리하게 함
    
    Args:
        message: 사용자 질문
        user_grade: 사용자 학년
        user_major: 사용자 전공
        course_registration_csv: 수강신청 CSV 전체 텍스트
        academic_calendar_csv: 학사일정 CSV 전체 텍스트
    
    Returns:
        (answer_text, sources_list) 튜플
    """
    logger.info(f"SCHEDULE 파이프라인 시작: {message[:50]}...")
    
    # 오늘 날짜
    today = datetime.now().strftime("%Y년 %m월 %d일 (%A)")
    
    # 프롬프트 생성
    prompt = SCHEDULE_PROMPT_TEMPLATE.format(
        today_date=today,
        user_grade=user_grade if user_grade else "미지정",
        user_major=user_major if user_major else "미지정",
        message=message,
        course_registration_csv=course_registration_csv,
        academic_calendar_csv=academic_calendar_csv,
    )
    
    # LLM 호출 (실패 시 fallback 메시지)
    answer = _call_gemini_api(prompt)
    if answer is None:
        answer = FALLBACK_MESSAGE
        logger.warning("SCHEDULE 파이프라인: fallback 메시지 사용")
    
    # 출처는 고정 (CSV 파일은 문서 조각이 아닌 전체 데이터 사용)
    source_docs = [
        {
            "title": "수강신청 일정표",
            "content": "2025학년도 2학기 수강신청 일정 및 대상자 정보",
            "url": "https://ewha.ac.kr/ewha/bachelor/course01.do",
            "category": "학사안내 > 수강",
            "relevance_score": None
        },
        {
            "title": "학사일정",
            "content": "2025학년도 학사 일정 (개강, 종강, 등록, 채플, 공휴일 등)",
            "url": "https://ewha.ac.kr/ewha/schedule.do",
            "category": "학사안내 > 학사일정",
            "relevance_score": None
        }
    ]
    
    logger.info("SCHEDULE 파이프라인 완료")
    return answer, source_docs
