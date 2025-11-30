"""
데이터 임베딩 스크립트
data/official/ 디렉터리의 Markdown 파일들만 읽어서 ChromaDB에 임베딩

NOTE: CSV 파일(수강신청, 학사일정)은 벡터 DB에 포함하지 않음.
      SCHEDULE 질문은 CSV 전체를 LLM에 직접 전달하는 방식으로 처리.
"""
import os
import sys
from pathlib import Path
import logging
from typing import List

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
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


def parse_yaml_frontmatter(content: str) -> tuple[dict, str]:
    """
    Markdown 문서에서 YAML frontmatter 파싱
    
    Args:
        content: 문서 전체 내용
    
    Returns:
        (메타데이터 dict, frontmatter 제외한 본문)
    """
    import yaml
    
    if not content.startswith("---"):
        return {}, content
    
    # frontmatter 끝 찾기
    end_idx = content.find("---", 3)
    if end_idx == -1:
        return {}, content
    
    frontmatter_str = content[3:end_idx].strip()
    body = content[end_idx + 3:].strip()
    
    try:
        metadata = yaml.safe_load(frontmatter_str)
        if metadata is None:
            metadata = {}
        
        # ChromaDB는 리스트 타입을 지원하지 않으므로 문자열로 변환
        # topics 리스트를 문자열로 변환 (검색용)
        if "topics" in metadata and isinstance(metadata["topics"], list):
            metadata["topics"] = " ".join(metadata["topics"])
        
        # category_path 리스트를 문자열로 변환
        if "category_path" in metadata and isinstance(metadata["category_path"], list):
            metadata["category_path"] = " > ".join(metadata["category_path"])
        
        return metadata, body
    except yaml.YAMLError as e:
        logger.warning(f"YAML frontmatter 파싱 실패: {e}")
        return {}, content


def load_markdown_documents(data_dir: str) -> List[Document]:
    """
    Markdown 문서 로드 (YAML frontmatter 메타데이터 포함)
    
    Args:
        data_dir: 데이터 디렉터리 경로
    
    Returns:
        문서 리스트
    """
    logger.info(f"Markdown 문서 로드 중: {data_dir}")
    
    documents = []
    data_path = Path(data_dir)
    
    # .md 파일 직접 순회하여 frontmatter 파싱
    for md_file in data_path.glob("**/*.md"):
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # YAML frontmatter 파싱
            metadata, body = parse_yaml_frontmatter(content)
            
            # source 메타데이터 추가 (상대 경로)
            relative_path = md_file.relative_to(data_path.parent)
            metadata["source"] = str(relative_path)
            
            # 빈 문서는 스킵
            if not body.strip():
                logger.warning(f"빈 문서 스킵: {md_file}")
                continue
            
            doc = Document(page_content=body, metadata=metadata)
            documents.append(doc)
            
        except Exception as e:
            logger.error(f"문서 로드 실패 ({md_file}): {e}")
    
    logger.info(f"로드된 Markdown 문서: {len(documents)}개")
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
        
        # 1. Markdown 문서만 로드 (CSV는 벡터 DB에 포함하지 않음)
        md_documents = load_markdown_documents(str(data_dir))
        logger.info(f"Markdown 문서: {len(md_documents)}개")
        
        if not md_documents:
            logger.warning("로드된 Markdown 문서가 없습니다.")
            sys.exit(0)
        
        logger.info(f"총 문서 수: {len(md_documents)}개")
        
        # 2. Markdown 문서 분할
        split_md_docs = split_documents(
            md_documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        # 3. 최종 문서 리스트 생성
        final_documents = split_md_docs
        logger.info(f"최종 청크 수: {len(final_documents)}개 (Markdown만 포함)")
        
        # 4. 벡터스토어 생성
        persist_dir = str(project_root / "backend" / settings.chroma_persist_dir)
        vectorstore = create_vectorstore(final_documents, persist_dir)
        
        logger.info("=== 데이터 임베딩 완료 ===")
        logger.info(f"총 {len(md_documents)}개 원본 문서, {len(final_documents)}개 최종 청크 임베딩 완료")
        logger.info(f"벡터스토어 저장 위치: {persist_dir}")
        logger.info("NOTE: CSV 파일(수강신청, 학사일정)은 별도 파이프라인에서 처리됩니다.")
        
    except Exception as e:
        logger.error(f"임베딩 중 오류 발생: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
