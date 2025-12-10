"""
데이터 임베딩 스크립트 (로컬 전용)
E5 임베딩 모델을 사용하여 Markdown 문서를 ChromaDB에 임베딩

사용법:
    uv run python -m scripts.embed_data

NOTE: 
    - 이 스크립트는 로컬에서만 실행
    - 생성된 chroma_db/는 Git에 커밋하여 배포
"""
import os
import sys
import shutil
from pathlib import Path
import logging
from typing import List

import yaml
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from app.config import settings
from app.guide_pipeline import E5Embeddings

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 문서 로더
# =============================================================================

def parse_yaml_frontmatter(content: str) -> tuple[dict, str]:
    """
    Markdown 문서에서 YAML frontmatter 파싱
    
    Returns:
        (메타데이터 dict, frontmatter 제외한 본문)
    """
    if not content.startswith("---"):
        return {}, content
    
    end_idx = content.find("---", 3)
    if end_idx == -1:
        return {}, content
    
    frontmatter_str = content[3:end_idx].strip()
    body = content[end_idx + 3:].strip()
    
    try:
        metadata = yaml.safe_load(frontmatter_str) or {}
        
        # ChromaDB는 리스트를 지원하지 않으므로 문자열로 변환
        if "topics" in metadata and isinstance(metadata["topics"], list):
            metadata["topics"] = " ".join(metadata["topics"])
        
        if "category_path" in metadata and isinstance(metadata["category_path"], list):
            metadata["category_path"] = " > ".join(metadata["category_path"])
        
        return metadata, body
        
    except yaml.YAMLError as e:
        logger.warning(f"YAML frontmatter 파싱 실패: {e}")
        return {}, content


def load_markdown_documents(data_dir: str) -> List[Document]:
    """Markdown 문서 로드 (YAML frontmatter 메타데이터 포함)"""
    logger.info(f"Markdown 문서 로드 중: {data_dir}")
    
    documents = []
    data_path = Path(data_dir)
    
    for md_file in data_path.glob("**/*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            metadata, body = parse_yaml_frontmatter(content)
            
            # source 메타데이터 추가
            relative_path = md_file.relative_to(data_path.parent)
            metadata["source"] = str(relative_path)
            
            if not body.strip():
                logger.warning(f"빈 문서 스킵: {md_file}")
                continue
            
            documents.append(Document(page_content=body, metadata=metadata))
            
        except Exception as e:
            logger.error(f"문서 로드 실패 ({md_file}): {e}")
    
    logger.info(f"로드된 Markdown 문서: {len(documents)}개")
    return documents


# =============================================================================
# 문서 처리
# =============================================================================

def split_documents(
    documents: List[Document], 
    chunk_size: int, 
    chunk_overlap: int
) -> List[Document]:
    """문서를 청크로 분할"""
    logger.info(f"문서 분할 중 (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"분할된 청크 수: {len(split_docs)}개")
    
    return split_docs


def create_vectorstore(documents: List[Document], persist_dir: str) -> Chroma:
    """벡터스토어 생성 및 문서 임베딩"""
    logger.info(f"E5 임베딩 모델 초기화 중: {settings.embedding_model}")
    embeddings = E5Embeddings(
        model_name=settings.embedding_model,
        device=settings.embedding_device
    )
    
    logger.info(f"벡터스토어 생성 중: {persist_dir}")
    
    # 기존 디렉터리 삭제
    if os.path.exists(persist_dir):
        logger.warning(f"기존 벡터스토어 삭제: {persist_dir}")
        shutil.rmtree(persist_dir)
    
    # 새 벡터스토어 생성
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    logger.info("벡터스토어 생성 완료")
    return vectorstore


# =============================================================================
# 메인
# =============================================================================

def main():
    """메인 실행 함수"""
    try:
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data" / "official"
        
        if not data_dir.exists():
            logger.error(f"데이터 디렉터리가 존재하지 않습니다: {data_dir}")
            sys.exit(1)
        
        logger.info("=" * 50)
        logger.info("데이터 임베딩 시작 (E5 모델)")
        logger.info("=" * 50)
        
        # 1. Markdown 문서 로드
        md_documents = load_markdown_documents(str(data_dir))
        
        if not md_documents:
            logger.warning("로드된 문서가 없습니다.")
            sys.exit(0)
        
        # 2. 문서 분할
        split_docs = split_documents(
            md_documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        # 3. 벡터스토어 생성
        persist_dir = str(project_root / settings.chroma_persist_dir)
        create_vectorstore(split_docs, persist_dir)
        
        logger.info("=" * 50)
        logger.info("임베딩 완료!")
        logger.info(f"  - 원본 문서: {len(md_documents)}개")
        logger.info(f"  - 청크 수: {len(split_docs)}개")
        logger.info(f"  - 저장 위치: {persist_dir}")
        logger.info("=" * 50)
        logger.info("이제 chroma_db/를 Git에 커밋하세요!")
        
    except Exception as e:
        logger.error(f"임베딩 중 오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
