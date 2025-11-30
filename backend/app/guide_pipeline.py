"""
GUIDE 파이프라인 모듈
학사규정/제도 관련 질문을 Markdown RAG로 처리
"""
from typing import Tuple, List, Optional
import logging
import os
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi
import numpy as np
import requests

from app.config import settings

logger = logging.getLogger(__name__)


# GUIDE LLM 프롬프트 템플릿
GUIDE_PROMPT_TEMPLATE = """당신은 이화여자대학교 학사 안내 챗봇 "버디"입니다.
버디는 이화여대 학생들을 도와주는 귀여운 곰돌이 캐릭터입니다.
아래는 학사안내 공식 문서에서 검색된 관련 내용입니다.
이 내용을 기반으로, 학생의 질문에 친절하고 친근하게 답변해 주세요.

[사용자 정보]
- 학년: {user_grade}
- 전공: {user_major}

[질문]
{message}

[검색된 참고 문서]
{context}

[답변 지침]
1. 반드시 참고 문서의 내용을 우선으로 답변합니다.
2. 규정/조건/예외 사항이 있다면 함께 설명합니다.
3. 확실하지 않은 내용은 임의로 만들어내지 말고,
   "이 부분은 버디가 가진 자료에서 찾을 수 없어요"라고 안내합니다.

[말투 규칙]
- 반말(존댓말)을 사용하되, 친근하고 따뜻한 톤으로 말해주세요.
- 이모지는 적절히 사용하되 과하지 않게 (답변 시작과 끝에 1개씩 정도).
- "내가 정리해줌게!", "이거 중요해!", "궁금한 거 또 물어봐!" 같은 친근한 표현 사용.
- 신입생도 이해하기 쉬운 자연스러운 한국어로 설명합니다.
- 정보는 정확하게, 톤은 귀엽게!"""


class E5Embeddings(Embeddings):
    """Multilingual E5 임베딩 모델 래퍼"""
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"E5 임베딩 모델 로드 완료 (device: {device})")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 임베딩 - E5는 'passage: ' 접두사 사용"""
        prefixed_texts = [f"passage: {text}" for text in texts]
        embeddings = self.model.encode(prefixed_texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """쿼리 임베딩 - E5는 'query: ' 접두사 사용"""
        prefixed_text = f"query: {text}"
        embedding = self.model.encode([prefixed_text], normalize_embeddings=True)
        return embedding[0].tolist()


class HybridRetriever:
    """하이브리드 검색기 (Semantic + BM25)"""
    
    def __init__(self, vectorstore: Chroma, documents: List[Document]):
        self.vectorstore = vectorstore
        self.documents = documents
        
        # BM25 인덱스 생성 (본문 + topics 메타데이터 포함)
        self.tokenized_corpus = [self._tokenize_with_metadata(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"하이브리드 검색기 초기화 완료 (문서 수: {len(documents)})")
    
    def _tokenize(self, text: str) -> List[str]:
        """한국어/영어 토큰화"""
        tokens = re.findall(r'[가-힣]+|[a-zA-Z]+|[0-9]+', text.lower())
        return tokens
    
    def _tokenize_with_metadata(self, doc: Document) -> List[str]:
        """문서 본문 + 메타데이터(topics) 토큰화"""
        # 본문 토큰화
        content_tokens = self._tokenize(doc.page_content)
        
        # topics 메타데이터가 있으면 추가 토큰화 (가중치 부여를 위해 3회 반복)
        metadata = doc.metadata if doc.metadata else {}
        
        # topics가 문자열이면 파싱 시도
        topics_str = metadata.get("topics", "")
        if topics_str:
            # YAML frontmatter에서 온 topics 리스트 파싱
            topic_tokens = self._tokenize(topics_str)
            # topics는 중요하므로 가중치 부여 (3배)
            content_tokens.extend(topic_tokens * 3)
        
        # title도 추가 (2배 가중치)
        title = metadata.get("title", "")
        if title:
            title_tokens = self._tokenize(title)
            content_tokens.extend(title_tokens * 2)
        
        return content_tokens
    
    def retrieve(self, query: str, k: int = 8, alpha: float = 0.5) -> List[Document]:
        """
        하이브리드 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            alpha: semantic(0.0) vs bm25(1.0) 가중치
        """
        # 1. Semantic 검색
        semantic_docs = self.vectorstore.similarity_search_with_score(query, k=k*2)
        semantic_dict = {id(doc): (doc, 1.0 - score) for doc, score in semantic_docs}
        
        # 2. BM25 검색
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        if max(bm25_scores) > 0:
            bm25_scores = bm25_scores / max(bm25_scores)
        
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k*2]
        bm25_dict = {id(self.documents[i]): (self.documents[i], bm25_scores[i]) 
                     for i in top_bm25_indices if bm25_scores[i] > 0}
        
        # 3. 하이브리드 스코어 계산
        all_doc_ids = set(semantic_dict.keys()) | set(bm25_dict.keys())
        hybrid_scores = {}
        
        for doc_id in all_doc_ids:
            semantic_score = semantic_dict.get(doc_id, (None, 0))[1]
            bm25_score = bm25_dict.get(doc_id, (None, 0))[1]
            
            hybrid_score = alpha * bm25_score + (1 - alpha) * semantic_score
            doc = semantic_dict.get(doc_id, bm25_dict.get(doc_id))[0]
            hybrid_scores[doc_id] = (doc, hybrid_score)
        
        sorted_docs = sorted(hybrid_scores.values(), key=lambda x: x[1], reverse=True)[:k]
        
        logger.info(f"하이브리드 검색 완료: {len(sorted_docs)}개 문서")
        return [doc for doc, score in sorted_docs]


class GuideEngine:
    """GUIDE 파이프라인용 RAG 엔진"""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.hybrid_retriever = None
        
        self._initialize()
    
    def _initialize(self):
        """구성 요소 초기화"""
        try:
            # 임베딩 모델 초기화
            logger.info(f"임베딩 모델 초기화 중: {settings.embedding_model}")
            self.embeddings = E5Embeddings(
                model_name=settings.embedding_model,
                device=settings.embedding_device
            )
            
            # ChromaDB 벡터스토어 로드
            logger.info(f"ChromaDB 로드 중: {settings.chroma_persist_dir}")
            if not os.path.exists(settings.chroma_persist_dir):
                logger.warning(f"ChromaDB 디렉터리가 존재하지 않습니다: {settings.chroma_persist_dir}")
                logger.warning("먼저 데이터 임베딩을 실행해주세요.")
                self.vectorstore = Chroma(
                    persist_directory=settings.chroma_persist_dir,
                    embedding_function=self.embeddings
                )
                self.hybrid_retriever = None
            else:
                self.vectorstore = Chroma(
                    persist_directory=settings.chroma_persist_dir,
                    embedding_function=self.embeddings
                )
                
                # 모든 문서 로드 (BM25용)
                logger.info("BM25 인덱스 구축을 위한 문서 로드 중...")
                all_docs = self.vectorstore.get()
                documents = []
                if all_docs and 'documents' in all_docs:
                    for i, doc_text in enumerate(all_docs['documents']):
                        metadata = all_docs['metadatas'][i] if 'metadatas' in all_docs else {}
                        documents.append(Document(page_content=doc_text, metadata=metadata))
                
                # 하이브리드 검색기 초기화
                if documents:
                    self.hybrid_retriever = HybridRetriever(self.vectorstore, documents)
                    logger.info(f"하이브리드 검색기 초기화 완료 (총 {len(documents)}개 문서)")
                else:
                    self.hybrid_retriever = None
                    logger.warning("문서가 없어 하이브리드 검색기를 초기화하지 못했습니다.")
            
            logger.info("GUIDE 엔진 초기화 완료")
            
        except Exception as e:
            logger.error(f"GUIDE 엔진 초기화 실패: {str(e)}")
            raise
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Gemini API 직접 호출"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.llm_model}:generateContent"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": settings.llm_temperature,
                "maxOutputTokens": 8192,
            }
        }
        
        try:
            response = requests.post(
                f"{url}?key={settings.google_api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    return candidate["content"]["parts"][0]["text"]
            
            logger.error(f"예상치 못한 API 응답 구조: {result}")
            raise ValueError("Gemini API 응답 파싱 실패")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API 호출 실패: {str(e)}")
            raise
    
    def get_answer(
        self,
        message: str,
        user_grade: Optional[int] = None,
        user_major: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        GUIDE 질문에 대한 답변 생성
        
        Args:
            message: 사용자 질문
            user_grade: 사용자 학년
            user_major: 사용자 전공
        
        Returns:
            (answer_text, sources_list) 튜플
        """
        logger.info(f"GUIDE 파이프라인 시작: {message[:50]}...")
        
        # 사용자 컨텍스트 추가
        context_query = message
        if user_grade or user_major:
            context_parts = []
            if user_grade:
                context_parts.append(f"{user_grade}학년")
            if user_major:
                context_parts.append(f"{user_major}과")
            context_query = f"{', '.join(context_parts)} 학생 질문: {message}"
        
        # 하이브리드 검색
        if self.hybrid_retriever:
            retrieved_docs = self.hybrid_retriever.retrieve(
                query=context_query,
                k=settings.top_k_results,
                alpha=0.4
            )
        else:
            logger.warning("하이브리드 검색기 미초기화, Semantic 검색만 사용")
            retrieved_docs = self.vectorstore.similarity_search(context_query, k=settings.top_k_results)
        
        logger.info(f"검색 완료: {len(retrieved_docs)}개 문서")
        
        # 컨텍스트 구성
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content
            if len(content) > 1500:
                content = content[:1500] + "..."
            context_parts.append(f"[문서{i}] (출처: {source})\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # 프롬프트 생성
        prompt = GUIDE_PROMPT_TEMPLATE.format(
            user_grade=user_grade if user_grade else "미지정",
            user_major=user_major if user_major else "미지정",
            message=message,
            context=context
        )
        
        # LLM 호출
        answer = self._call_gemini_api(prompt)
        
        # 출처 추출
        sources = []
        for doc in retrieved_docs:
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                source = doc.metadata["source"]
                if source not in sources:
                    sources.append(source)
        
        logger.info("GUIDE 파이프라인 완료")
        return answer, sources


# 전역 GUIDE 엔진 인스턴스
_guide_engine: Optional[GuideEngine] = None


def get_guide_engine() -> GuideEngine:
    """전역 GUIDE 엔진 반환 (싱글톤)"""
    global _guide_engine
    if _guide_engine is None:
        _guide_engine = GuideEngine()
    return _guide_engine


def initialize_guide_engine():
    """GUIDE 엔진 초기화"""
    global _guide_engine
    _guide_engine = GuideEngine()
    logger.info("GUIDE 엔진 전역 초기화 완료")


def answer_guide_question(
    message: str,
    user_grade: Optional[int] = None,
    user_major: Optional[str] = None
) -> Tuple[str, List[str]]:
    """
    GUIDE 질문에 대한 답변 생성 (편의 함수)
    
    Args:
        message: 사용자 질문
        user_grade: 사용자 학년
        user_major: 사용자 전공
    
    Returns:
        (answer_text, sources_list) 튜플
    """
    engine = get_guide_engine()
    return engine.get_answer(message, user_grade, user_major)
