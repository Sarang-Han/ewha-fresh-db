from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, List


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # Google API
    google_api_key: str
    
    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    
    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # LLM 설정 (폴백 체인: 앞에서부터 순서대로 시도)
    llm_temperature: float = 0.3
    model_chain: List[str] = Field(default=["gemini-2.5-flash", "gemini-2.5-flash-lite"])

    # 임베딩 모델 설정 (E5)
    embedding_model: str = Field(default="intfloat/multilingual-e5-large-instruct")
    embedding_device: str = Field(default="cpu")

    # RAG 설정
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=150)
    top_k_results: int = Field(default=8)  # 하이브리드 검색이 반환할 문서 수

    class Config:
        env_file = ".env"
        case_sensitive = False
        env_file_encoding = "utf-8"


# 전역 설정 객체
settings = Settings()
