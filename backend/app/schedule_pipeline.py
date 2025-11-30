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

4. 답변 형식:
   - 시간 순으로 정렬해서 bullet list로 정리합니다.
   - 각 항목에 일정 이름, 대상, 기간, 설명(필요 시)을 포함합니다.
   - 규정 정보가 있다면 함께 안내합니다.

5. 계산이 필요한 질문:
   - 채플 횟수, 수업일 수 등 계산이 필요하면 실제로 계산하여 답변합니다.
   - 공휴일, 중간시험 기간 등 제외일을 고려합니다.

6. 모르는 정보가 있으면 추측하지 말고,
   "해당 정보는 버디가 가진 자료에서 찾기 어려워요. 더 정확한 일정은 이화 포탈(portal.ewha.ac.kr)이나 학사일정 페이지(ewha.ac.kr/ewha/schedule.do)를 확인해봐!"라고 안내하세요.

[말투 규칙]
- 반말(존댓말)을 사용하되, 친근하고 따뜻한 톤으로 말해주세요.
- 이모지는 적절히 사용하되 과하지 않게 (답변 시작과 끝에 1개씩 정도).
- "내가 정리해줌게!", "이거 중요해!", "화이팅!" 같은 친근한 표현 사용 가능.
- 정보는 정확하게, 톤은 귀엽게!

이제 위 규칙에 따라 답변을 작성해 주세요."""


def _call_gemini_api(prompt: str) -> str:
    """Gemini API 직접 호출"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.llm_model}:generateContent"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": settings.llm_temperature,
            "maxOutputTokens": 8192,
        }
    }
    
    try:
        response = requests.post(
            f"{url}?key={settings.google_api_key}",
            headers=headers,
            json=data,
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return candidate["content"]["parts"][0]["text"]
        
        logger.error(f"예상치 못한 API 응답 구조: {result}")
        raise ValueError(f"Gemini API 응답 파싱 실패")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Gemini API 호출 실패: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"SCHEDULE 파이프라인 오류: {str(e)}")
        raise


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
    
    # LLM 호출
    answer = _call_gemini_api(prompt)
    
    # 출처는 고정
    sources = [
        "official/수강신청/25-2_course_registration.csv",
        "official/학사일정/academic_calendar.csv",
    ]
    
    logger.info("SCHEDULE 파이프라인 완료")
    return answer, sources
