from pydantic import BaseModel, Field
from typing import Optional, List


class ChatRequest(BaseModel):
    """채팅 요청 모델"""
    session_id: str = Field(..., description="브라우저 세션 ID")
    message: str = Field(..., description="사용자 질문", min_length=1)
    user_grade: Optional[int] = Field(None, description="학년 정보 (1-4)", ge=1, le=4)
    user_major: Optional[str] = Field(None, description="전공 정보")


class ChatResponse(BaseModel):
    """채팅 응답 모델"""
    session_id: str
    answer: str
    sources: List[str] = Field(default_factory=list, description="참고한 문서 출처")


class HealthResponse(BaseModel):
    """헬스체크 응답 모델"""
    status: str
    message: str
