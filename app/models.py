from pydantic import BaseModel, Field
from typing import Optional, List


class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    session_id: str = Field(..., description="브라우저 세션 ID")
    message: str = Field(..., description="사용자 질문", min_length=1)
    user_grade: Optional[int] = Field(None, description="학년 정보 (1-4)", ge=1, le=4)
    user_major: Optional[str] = Field(None, description="전공 정보")


class SourceDocument(BaseModel):
    """참조 문서 모델"""
    title: str = Field(..., description="문서 제목")
    content: str = Field(..., description="참조된 문서 조각 (chunk)")
    url: str = Field(..., description="원본 문서 URL")
    category: Optional[str] = Field(None, description="문서 카테고리 (학사안내 > 전공선택)")
    relevance_score: Optional[float] = Field(None, description="관련성 점수 (0~1)")


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    session_id: str
    answer: str
    sources: List[SourceDocument] = Field(default_factory=list, description="참고한 문서 조각들 (최대 5개)")


class HealthResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str
    message: str


class CalendarEventResponse(BaseModel):
    """학사일정 개별 이벤트 모델"""
    academic_year: int = Field(..., description="학년도")
    start_date: str = Field(..., description="시작일 (YYYY-MM-DD)")
    end_date: str = Field(..., description="종료일 (YYYY-MM-DD)")
    semester: str = Field(..., description="학기 (1, 2, summer, winter)")
    event_type: str = Field(..., description="이벤트 타입")
    title_raw: str = Field(..., description="일정명")
    target: str = Field(..., description="대상")
    is_holiday: bool = Field(..., description="공휴일 여부")
    notes: Optional[str] = Field(None, description="비고")

