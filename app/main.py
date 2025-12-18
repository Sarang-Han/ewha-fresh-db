from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.models import ChatRequest, ChatResponse, HealthResponse, SourceDocument
from app.intent_router import classify_intent, Intent
from app.csv_loader import initialize_csv_texts, get_csv_texts
from app.schedule_pipeline import answer_schedule_question
from app.guide_pipeline import initialize_guide_engine, answer_guide_question
from app.config import settings

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작 시
    logger.info("=== 애플리케이션 초기화 시작 ===")
    
    # 1. CSV 텍스트 로드 (SCHEDULE 파이프라인용)
    logger.info("CSV 텍스트 초기화 중...")
    initialize_csv_texts()
    
    # 2. GUIDE 엔진 초기화 (Markdown RAG)
    logger.info("GUIDE 엔진 초기화 중...")
    initialize_guide_engine()
    
    logger.info("=== 애플리케이션 초기화 완료 ===")
    
    yield
    
    # 종료 시
    logger.info("애플리케이션 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="이화여대 신입생 학사 챗봇 API",
    description="RAG 기반 학사 안내 챗봇 백엔드",
    version="0.1.0",
    lifespan=lifespan
)

# CORS 설정 (프론트엔드 연동을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def root():
    """루트 엔드포인트"""
    return HealthResponse(
        status="ok",
        message="이화여대 신입생 학사 챗봇 API가 실행 중입니다."
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크 엔드포인트"""
    return HealthResponse(
        status="healthy",
        message="서비스가 정상 작동 중입니다."
    )


@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """
    질문에 대한 답변을 반환하는 엔드포인트
    
    새 아키텍처:
    1. Intent Router로 질문 분류 (SCHEDULE, GUIDE, OTHER)
    2. SCHEDULE: CSV 전체 + LLM으로 처리
    3. GUIDE: Markdown RAG + LLM으로 처리
    4. OTHER: 기본 응답
    
    Args:
        request: 채팅 요청 (세션 ID, 메시지, 학년, 전공 등)
    
    Returns:
        ChatResponse: 답변 및 출처 정보
    """
    try:
        logger.info(f"세션 {request.session_id}로부터 질문: {request.message}")
        
        # 1. Intent 분류
        intent = classify_intent(
            message=request.message,
            user_grade=request.user_grade,
            user_major=request.user_major
        )
        logger.info(f"Intent 분류 결과: {intent.value}")
        
        # 2. Intent별 파이프라인 분기
        if intent == Intent.SCHEDULE:
            # SCHEDULE: CSV 전체를 LLM에 넘겨 처리
            csv_texts = get_csv_texts()
            answer, sources = answer_schedule_question(
                message=request.message,
                user_grade=request.user_grade,
                user_major=request.user_major,
                course_registration_csv=csv_texts.get("course_registration", ""),
                academic_calendar_csv=csv_texts.get("academic_calendar", ""),
            )
            
        elif intent == Intent.GUIDE:
            # GUIDE: Markdown RAG로 처리
            answer, sources = answer_guide_question(
                message=request.message,
                user_grade=request.user_grade,
                user_major=request.user_major
            )
            
        else:
            # OTHER: 기본 응답 (버디 캐릭터)
            answer = """
음... 이 부분은 버디가 가진 자료에서 찾기 어려운 내용이야 🥲

혹시 더 정확한 정보가 필요하다면 아래를 참고해봐!

- 📌 **학적팀 연락처**: 02-3277-2030, 2033
- 📌 **공지사항**: [공지사항 페이지](https://www.ewha.ac.kr/ewha/news/notice.do)

버디가 간혹 단어를 제대로 인식하지 못하는 경우가 있으니, **다른 말로 다시 물어봐**!

아는 범위에서 최대한 도와줄게 💚"""
            sources = []
        
        # sources를 SourceDocument 모델로 변환
        source_documents = [SourceDocument(**src) if isinstance(src, dict) else src for src in sources]
        
        logger.info(f"답변 생성 완료 (Intent: {intent.value}, 출처: {len(source_documents)}개)")
        
        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            sources=source_documents
        )
        
    except Exception as e:
        logger.error(f"질문 처리 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류가 발생했습니다: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        reload_dirs=["app", "scripts"],
        reload_excludes=[".venv", "*.pyc", "__pycache__", "chroma_db"]
    )
