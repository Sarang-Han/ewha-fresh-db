"""
데이터 임베딩 스크립트
data/official/ 디렉터리의 Markdown, CSV 파일들을 읽어서 ChromaDB에 임베딩
"""
import os
import sys
from pathlib import Path
import logging
import pandas as pd
from typing import List

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from app.config import settings
from app.rag_engine import E5Embeddings

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_markdown_documents(data_dir: str) -> List[Document]:
    """
    Markdown 문서 로드
    
    Args:
        data_dir: 데이터 디렉터리 경로
    
    Returns:
        문서 리스트
    """
    logger.info(f"Markdown 문서 로드 중: {data_dir}")
    
    documents = []
    
    # .md 파일 로드
    try:
        loader = DirectoryLoader(
            data_dir,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents.extend(loader.load())
        logger.info(f"로드된 Markdown 문서: {len(documents)}개")
    except Exception as e:
        logger.error(f"Markdown 문서 로드 실패: {str(e)}")
    
    return documents


def load_csv_documents(data_dir: str, chunk_rows: int = 5) -> List[Document]:
    """
    CSV 파일 로드 및 처리
    
    Args:
        data_dir: 데이터 디렉터리 경로
        chunk_rows: 몇 행씩 묶어서 문서로 만들지
    
    Returns:
        문서 리스트
    """
    logger.info(f"CSV 문서 로드 중: {data_dir}")
    
    documents = []
    data_path = Path(data_dir)
    csv_files = list(data_path.glob("**/*.csv"))
    
    logger.info(f"발견된 CSV 파일: {len(csv_files)}개")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, encoding="utf-8")
            logger.info(f"CSV 파일 로드: {csv_file.name} ({len(df)}행)")
            
            # 파일명에서 카테고리 추출
            relative_path = csv_file.relative_to(data_path)
            category = str(relative_path.parent)
            
            # CSV를 청크 단위로 분할하여 문서 생성
            for start_idx in range(0, len(df), chunk_rows):
                end_idx = min(start_idx + chunk_rows, len(df))
                chunk_df = df.iloc[start_idx:end_idx]
                
                # 테이블 형태의 텍스트로 변환
                content_parts = [f"# {csv_file.stem} (행 {start_idx+1}-{end_idx})\n"]
                content_parts.append(f"카테고리: {category}\n")
                content_parts.append("## 데이터:\n")
                
                for _, row in chunk_df.iterrows():
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    content_parts.append(f"- {row_text}")
                
                content = "\n".join(content_parts)
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": str(csv_file.relative_to(data_path.parent)),
                        "type": "csv",
                        "category": category,
                        "file_name": csv_file.name,
                        "row_range": f"{start_idx+1}-{end_idx}"
                    }
                )
                documents.append(doc)
            
        except Exception as e:
            logger.error(f"CSV 파일 로드 실패 ({csv_file.name}): {str(e)}")
    
    logger.info(f"생성된 CSV 문서: {len(documents)}개")
    return documents


def split_documents(documents: list, chunk_size: int, chunk_overlap: int) -> list:
    """
    문서를 청크로 분할
    
    Args:
        documents: 문서 리스트
        chunk_size: 청크 크기
        chunk_overlap: 청크 중첩 크기
    
    Returns:
        분할된 문서 리스트
    """
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
    """
    벡터스토어 생성 및 문서 임베딩
    
    Args:
        documents: 문서 리스트
        persist_dir: ChromaDB 저장 디렉터리
    
    Returns:
        Chroma 벡터스토어
    """
    logger.info(f"임베딩 모델 초기화 중: {settings.embedding_model}")
    embeddings = E5Embeddings(
        model_name=settings.embedding_model,
        device=settings.embedding_device
    )
    
    logger.info(f"벡터스토어 생성 및 문서 임베딩 중: {persist_dir}")
    
    # 기존 디렉터리가 있으면 삭제
    if os.path.exists(persist_dir):
        logger.warning(f"기존 벡터스토어 삭제 중: {persist_dir}")
        import shutil
        shutil.rmtree(persist_dir)
    
    # 새 벡터스토어 생성
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    logger.info("벡터스토어 생성 완료")
    return vectorstore


def main():
    """메인 실행 함수"""
    try:
        # 데이터 디렉터리 경로 (프로젝트 루트 기준)
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data" / "official"
        
        if not data_dir.exists():
            logger.error(f"데이터 디렉터리가 존재하지 않습니다: {data_dir}")
            sys.exit(1)
        
        logger.info("=== 데이터 임베딩 시작 ===")
        
        # 1. Markdown 문서 로드
        md_documents = load_markdown_documents(str(data_dir))
        logger.info(f"Markdown 문서: {len(md_documents)}개")
        
        # 2. CSV 문서 로드
        csv_documents = load_csv_documents(str(data_dir), chunk_rows=settings.csv_chunk_rows)
        logger.info(f"CSV 문서: {len(csv_documents)}개")
        
        # 3. 모든 문서 병합
        all_documents = md_documents + csv_documents
        
        if not all_documents:
            logger.warning("로드된 문서가 없습니다.")
            sys.exit(0)
        
        logger.info(f"총 문서 수: {len(all_documents)}개")
        
        # 4. Markdown 문서만 분할 (CSV는 이미 적절히 분할됨)
        split_md_docs = split_documents(
            md_documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        ) if md_documents else []
        
        # 5. 최종 문서 리스트 생성
        final_documents = split_md_docs + csv_documents
        logger.info(f"최종 청크 수: {len(final_documents)}개 (Markdown: {len(split_md_docs)}, CSV: {len(csv_documents)})")
        
        # 6. 벡터스토어 생성
        persist_dir = str(project_root / "backend" / settings.chroma_persist_dir)
        vectorstore = create_vectorstore(final_documents, persist_dir)
        
        logger.info("=== 데이터 임베딩 완료 ===")
        logger.info(f"총 {len(all_documents)}개 원본 문서, {len(final_documents)}개 최종 청크 임베딩 완료")
        logger.info(f"벡터스토어 저장 위치: {persist_dir}")
        
    except Exception as e:
        logger.error(f"임베딩 중 오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
