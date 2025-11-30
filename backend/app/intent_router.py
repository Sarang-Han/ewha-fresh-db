"""
Intent Router 모듈
질문을 SCHEDULE, GUIDE, OTHER 중 하나로 분류
"""
from enum import Enum
from typing import Optional
import logging
import requests

from app.config import settings

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """질문 의도 분류"""
    SCHEDULE = "SCHEDULE"  # 일정/수강신청, 기간, 날짜
    GUIDE = "GUIDE"        # 규정/절차/제도 설명
    OTHER = "OTHER"        # 잡담/기타


# 키워드 기반 Pre-check를 위한 SCHEDULE 키워드 목록
SCHEDULE_KEYWORDS = [
    # 수강신청 관련
    "수강신청", "장바구니", "수강 정정", "수강정정", "수강철회", "철회",
    "우선수강", "본수강", "확인 및 변경",
    # 학사일정 관련
    "학사일정", "등록기간", "등록 기간", "개강", "종강", "폐강",
    "수강 기간", "신청 기간",
    # 채플 관련 (학사일정 CSV에 규정 포함)
    "채플", "보충채플", "특별채플",
    # 기타 학사일정 이벤트
    "계절학기", "여름학기", "겨울학기", "공휴일", "휴무일", "휴일",
    "입학식", "졸업식", "학위수여", "복학", "휴학 마감", "휴학원서",
    # 일반 시간/날짜 표현
    "언제", "기간", "날짜", "시간", "일정", "마감", "몇 번", "몇번"
]

# LLM 라우터 프롬프트 템플릿
ROUTER_PROMPT_TEMPLATE = """[시스템 역할]
당신은 이화여자대학교 학사 챗봇의 질문 분류기입니다.
사용자의 질문을 보고 아래 세 가지 중 하나로만 분류하세요.

- SCHEDULE: 수강신청, 장바구니, 학사일정, 등록 기간, 정정/철회 기간, 개강/종강 날짜,
            채플(결석 횟수, 보충채플 일정 포함), 계절학기, 공휴일, 휴무일,
            복학/휴학 신청 기간, 입학식/졸업식 날짜 등
            학사 캘린더에서 찾을 수 있는 모든 일정 및 관련 규정 질문.
            "언제", "기간", "날짜", "시간", "몇 번" 등을 묻는 질문.
- GUIDE   : 휴학/복학/자퇴의 절차와 조건, 전과, 복수전공 신청 방법,
            졸업요건, 성적 산출 방식, 재수강 규정, 학점 인정 등
            학사 제도의 상세 규정/절차/방법을 묻는 질문.
- OTHER   : 위 두 가지에 해당하지 않는 잡담, 인사, 일반 질문 등.

[중요]
- 채플 관련 질문(결석 허용 횟수, 보충채플, 채플 기간 등)은 SCHEDULE입니다.
- "몇 번", "몇 회" 같은 횟수 질문도 학사일정과 관련되면 SCHEDULE입니다.

반드시 SCHEDULE, GUIDE, OTHER 중 하나의 단어만 출력하세요.

[사용자 정보]
학년: {user_grade}
전공: {user_major}

[질문]
{message}

분류 결과:"""


def _call_gemini_for_classification(prompt: str) -> str:
    """Gemini API 호출하여 분류 결과 반환"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.llm_model}:generateContent"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,  # 분류는 낮은 temperature
            "maxOutputTokens": 20,
        }
    }
    
    try:
        response = requests.post(
            f"{url}?key={settings.google_api_key}",
            headers=headers,
            json=data,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                text = candidate["content"]["parts"][0]["text"].strip().upper()
                return text
        
        return "OTHER"
        
    except Exception as e:
        logger.error(f"Intent 분류 API 호출 실패: {str(e)}")
        return "OTHER"


def classify_intent(
    message: str,
    user_grade: Optional[int] = None,
    user_major: Optional[str] = None
) -> Intent:
    """
    질문을 SCHEDULE, GUIDE, OTHER 중 하나로 분류
    
    1단계: 키워드 기반 Pre-check (빠른 분기)
    2단계: 애매한 경우 LLM에 위임
    
    Args:
        message: 사용자 질문
        user_grade: 사용자 학년
        user_major: 사용자 전공
    
    Returns:
        Intent enum 값
    """
    message_lower = message.lower()
    
    # 1단계: 키워드 기반 Pre-check
    for keyword in SCHEDULE_KEYWORDS:
        if keyword in message_lower:
            logger.info(f"Intent 분류 (키워드 '{keyword}' 매칭): SCHEDULE")
            return Intent.SCHEDULE
    
    # 2단계: LLM 라우터로 분류
    logger.info("Intent 분류: LLM 라우터 사용")
    
    prompt = ROUTER_PROMPT_TEMPLATE.format(
        user_grade=user_grade if user_grade else "미지정",
        user_major=user_major if user_major else "미지정",
        message=message
    )
    
    llm_result = _call_gemini_for_classification(prompt)
    
    # 결과 파싱 (SCHEDULE, GUIDE, OTHER 중 하나만 추출)
    if "SCHEDULE" in llm_result:
        intent = Intent.SCHEDULE
    elif "GUIDE" in llm_result:
        intent = Intent.GUIDE
    else:
        intent = Intent.OTHER
    
    logger.info(f"Intent 분류 결과 (LLM): {intent.value}")
    return intent
