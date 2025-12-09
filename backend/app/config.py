from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # Google API
    google_api_key: str
    
    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    
    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # LLM 설정
    llm_model: str = "gemini-2.5-flash"
    llm_temperature: float = 0.3
    
    # 임베딩 모델 설정 (E5)
    embedding_model: str = Field(default="intfloat/multilingual-e5-large-instruct")
    embedding_device: str = Field(default="cpu")  # "cuda" for GPU
    
    # RAG 설정
    chunk_size: int = Field(default=800)
    chunk_overlap: int = Field(default=150)
    top_k_results: int = Field(default=8)
    
    # CSV 처리 설정
    csv_chunk_rows: int = Field(default=50)  # CSV를 몇 행씩 묶어서 처리할지
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        env_file_encoding = "utf-8"


# 전역 설정 객체
settings = Settings()
