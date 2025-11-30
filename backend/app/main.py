from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.models import ChatRequest, ChatResponse, HealthResponse
from app.rag_engine import RAGEngine
from app.config import settings

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# RAG 엔진 전역 인스턴스
rag_engine: RAGEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global rag_engine
    
    # 시작 시
    logger.info("RAG 엔진 초기화 중...")
    rag_engine = RAGEngine()
    logger.info("RAG 엔진 초기화 완료")
    
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
    
    Args:
        request: 채팅 요청 (세션 ID, 메시지, 학년, 전공 등)
    
    Returns:
        ChatResponse: 답변 및 출처 정보
    """
    try:
        logger.info(f"세션 {request.session_id}로부터 질문: {request.message}")
        
        # RAG 엔진을 통해 답변 생성
        answer, sources = rag_engine.get_answer(
            query=request.message,
            user_grade=request.user_grade,
            user_major=request.user_major
        )
        
        logger.info(f"답변 생성 완료 (출처: {len(sources)}개)")
        
        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            sources=sources
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
