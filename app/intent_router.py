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
    "수강신청", "수신", "장바구니", "수강 정정", "수강정정", "수정", "수강철회", "철회", "드랍",
    "우선수강", "본수강", "확인 및 변경", "수강변경",
    # 학사일정 관련
    "학사일정", "등록기간", "등록 기간", "개강", "종강", "폐강", "개강일", "종강일",
    "수강 기간", "신청 기간", "접수기간",
    # 채플 관련 (학사일정 CSV에 규정 포함)
    "채플", "보충채플", "특별채플", "예배",
    # 기타 학사일정 이벤트
    "계절학기", "여름학기", "겨울학기", "공휴일", "휴무일", "휴일", "방학",
    "입학식", "졸업식", "학위수여", "복학", "휴학 마감", "휴학원서", "복학신청",
    # 일반 시간/날짜 표현
    "언제", "기간", "날짜", "시간", "일정", "마감", "몇 번", "몇번", "며칠", "몇일"
]

# GUIDE 파이프라인으로 분류할 키워드 목록 (규정/절차/제도)
GUIDE_KEYWORDS = [
    # 등록 관련
    "정규등록", "학점등록", "등록금", "분할납부", "등록금 반환", "환불",
    # 학적변동
    "휴학", "복학", "자퇴", "제적", "재입학", "자퇴신청", "학교그만",
    "군복무휴학", "육아휴학", "임신출산휴학", "창업휴학",
    # 전공
    "복수전공", "부전공", "연계전공", "융합전공", "마이크로전공", "전과",
    "전공결정", "전공변경", "전공바꾸기", "전공선택", "자기설계전공", "스크랜튼",
    "이중전공", "투전공",  # 복수전공 동의어
    # 졸업 관련
    "졸업요건", "졸업학점", "졸업신청", "졸업유예", "졸업연기", "졸업보류", "졸업미루기", "학위취득유예",
    "과정수료", "학사학위과정수료", "수료", "조기졸업", "빨리졸업", "일찍졸업",
    "졸업논문", "졸업종합시험", "졸업시뮬레이션", "수강시뮬레이션",
    "학년수료", "훈련학점", "영어강의", "원어강의",
    # 성적 관련
    "성적", "학점", "평점", "우등", "학사경고", "유급", "경고",
    "재수강", "재이수", "학점인정", "학점포기", "수강포기",
    "출석인정", "생리공결", "병결", "질병결석", "공결",
    # 수강 규정 (일정이 아닌 제도)
    "석사", "대학원", "석사학위과정", "석사과목", "대학원 과목", "대학원과목",
    "학점교류", "교환학생", "타학교수강", "학교간학점교류",
    # 인증제
    "영어인증제", "정보인증제", "토익", "토플", "TOEIC", "TOEFL", "MOS",
    "텝스", "TEPS", "아이엘츠", "IELTS",  # 영어인증 추가
    # 편입
    "편입", "편입생", "전적대학", "학점이전", "편입학",
    # 자격증
    "교원자격증", "교직", "평생교육사", "교사자격증", "교생실습", "평생교육자격증", "평생교육",
    # 기타 제도
    "학점이월", "초과학점", "수업연한", "인문학 교양", "소프트웨어 교과목", "SW교과목",
]

# LLM 라우터 프롬프트 템플릿
ROUTER_PROMPT_TEMPLATE = """[시스템 역할]
당신은 이화여자대학교 학사 챗봇의 질문 분류기입니다.
사용자의 질문을 보고 아래 세 가지 중 하나로만 분류하세요.

- SCHEDULE: 수강신청, 장바구니, 학사일정, 등록 기간, 정정/철회 기간, 개강/종강 날짜,
            채플(결석 횟수, 보충채플 일정 포함), 계절학기 일정, 공휴일, 휴무일,
            복학/휴학 신청 기간, 입학식/졸업식 날짜 등
            "언제", "기간", "날짜", "시간", "몇 번" 등을 묻는 질문.
- GUIDE   : 휴학/복학/자퇴의 절차와 조건, 전과, 복수전공 신청 방법,
            **졸업요건, 졸업신청, 과정수료, 졸업유예, 조기졸업, 학년수료**,
            성적 산출 방식, 재수강 규정, 학점 인정,
            **학부생의 석사/대학원 과목 수강 자격 및 조건**,
            학점교류 자격요건, 교원자격증/평생교육사 취득 조건 등
            학사 제도의 상세 규정/절차/방법을 묻는 질문.
- OTHER   : 위 두 가지에 해당하지 않는 잡담, 인사, 일반 질문 등.

[중요]
- 채플 관련 질문(결석 허용 횟수, 보충채플, 채플 기간 등)은 SCHEDULE입니다.
- "석사 과목 수강 가능?", "대학원 과목 들을 수 있어?" 같은 자격/조건 질문은 GUIDE입니다.
- "석사 수강신청 언제야?" 같은 일정 질문은 SCHEDULE입니다.
- "과정수료 후 졸업 가능?", "졸업유예 신청 방법", "조기졸업 요건" 등은 GUIDE입니다.
- "졸업신청 기간 언제야?" 같은 일정 질문은 SCHEDULE입니다.

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
    
    1단계: 키워드 기반 Pre-check (빠른 분기) + 띄어쓰기 무시
    2단계: 애매한 경우 LLM에 위임
    
    Args:
        message: 사용자 질문
        user_grade: 사용자 학년
        user_major: 사용자 전공
    
    Returns:
        Intent enum 값
    """
    message_lower = message.lower()
    message_nospace = message_lower.replace(" ", "")  # 띄어쓰기 제거
    
    # 1단계: 키워드 기반 Pre-check (양쪽 모두 체크 후 우선순위 결정)
    schedule_matched = None
    guide_matched = None
    
    # SCHEDULE 키워드 체크 (원본 + 띄어쓰기 제거)
    for keyword in SCHEDULE_KEYWORDS:
        keyword_lower = keyword.lower()
        # 1) 원본 메시지에서 직접 매칭
        if keyword_lower in message_lower:
            schedule_matched = keyword
            break
        # 2) 띄어쓰기 제거 후 매칭 ("수강신청" == "수강 신청")
        keyword_nospace = keyword_lower.replace(" ", "")
        if keyword_nospace in message_nospace:
            schedule_matched = keyword
            break
    
    # GUIDE 키워드 체크 (원본 + 띄어쓰기 제거)
    for keyword in GUIDE_KEYWORDS:
        keyword_lower = keyword.lower()
        # 1) 원본 메시지에서 직접 매칭
        if keyword_lower in message_lower:
            guide_matched = keyword
            break
        # 2) 띄어쓰기 제거 후 매칭 ("졸업유예" == "졸업 유예")
        keyword_nospace = keyword_lower.replace(" ", "")
        if keyword_nospace in message_nospace:
            guide_matched = keyword
            break
    
    # 둘 다 매칭된 경우: GUIDE 우선 (규정/자격 질문이 일정보다 구체적)
    # 예: "석사 과목 수강신청 가능?" → "석사" + "수강신청" 둘 다 있으면 GUIDE
    if guide_matched and schedule_matched:
        logger.info(f"Intent 분류 (GUIDE '{guide_matched}' 우선, SCHEDULE '{schedule_matched}' 무시): GUIDE")
        return Intent.GUIDE
    elif guide_matched:
        logger.info(f"Intent 분류 (키워드 '{guide_matched}' 매칭): GUIDE")
        return Intent.GUIDE
    elif schedule_matched:
        logger.info(f"Intent 분류 (키워드 '{schedule_matched}' 매칭): SCHEDULE")
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
