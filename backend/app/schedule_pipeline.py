"""
SCHEDULE 파이프라인 모듈
수강신청/학사일정 관련 질문을 CSV 전체 + LLM으로 처리
"""
from typing import Tuple, List, Optional
import logging
import requests

from app.config import settings

logger = logging.getLogger(__name__)


# SCHEDULE LLM 프롬프트 템플릿
SCHEDULE_PROMPT_TEMPLATE = """당신은 이화여자대학교 학사/수강신청 안내를 담당하는 AI 비서입니다.
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
```

[학사일정 CSV 전체]
```csv
{academic_calendar_csv}
```

[필터링/답변 규칙]
1. 먼저 수강신청 CSV에서 [사용자 정보]와 [질문]에 해당하는 모든 행을 찾으세요.
   - 학년 조건:
     - "4학년" 질문이면 4학년 또는 4학년 이상/이하로 명시된 행을 포함합니다.
     - "3~4학년", "전체 재학생" 등 범위로 표시된 행도 포함합니다.
   - 전공 조건:
     - 사용자의 전공(예: 컴퓨터공학과)을 정확히 포함하는 전공 대상 행을 우선 포함합니다.
     - "전체", "전 재학생", "전 학년" 등 공통 대상 행도 함께 포함합니다.
   - 이벤트 종류:
     - 우선수강신청, 본수강신청, 학년별 수강신청, 전체학년 수강신청,
       장바구니, 수강 정정(확인 및 변경), 수강철회 등
       수강신청과 직접 관련된 모든 이벤트를 포함합니다.

2. 학사일정 CSV는 보조적으로 사용합니다.
   - 개강일, 종강일, 등록기간 등, 질문과 관련이 있고 학생에게 도움이 될 만한 일정이 있으면 함께 알려줍니다.

3. 답변 형식:
   - 시간 순으로 정렬해서 bullet list로 정리합니다.
   - 각 항목에 아래 정보를 포함합니다.
     - 일정 이름
     - 대상(학년, 전공)
     - 기간(날짜와 시간)
     - 간단한 설명(필요 시)
   - 예시 형식:
     컴퓨터공학과 4학년 학생의 2025학년도 2학기 수강신청 관련 주요 일정은 다음과 같습니다.
     • 우선수강신청 (컴퓨터공학과 전공생)
       - 기간: 2025년 8월 4일(월) 13:00 ~ 16:00
       - 대상: 컴퓨터공학과 주/복수전공 2~4학년
     • 4학년 본수강신청
       - 기간: 2025년 8월 5일(화) 09:00 ~ 12:00
       - 대상: 4학년 이상 재학생

4. 사용자의 질문에 불필요한 다른 학년/전공의 이벤트는 설명하지 않습니다.
   - 다만, 모든 학생 공통 일정(장바구니, 수강정정, 수강철회 등)은 함께 보여줍니다.

5. 모르는 정보가 있으면 추측하지 말고,
   "제공된 CSV에서 해당 정보를 찾을 수 없다"고 명시하세요.

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
    
    # 프롬프트 생성
    prompt = SCHEDULE_PROMPT_TEMPLATE.format(
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
