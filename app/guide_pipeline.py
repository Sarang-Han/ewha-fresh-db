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


# Fallback 모델 체인 (순서대로 시도)
MODEL_CHAIN = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# API 실패 시 기본 안내 메시지
GUIDE_FALLBACK_MESSAGE = """😥 지금 서버가 조금 바빠서 답변을 생성하기 어려워요.

잠시 후 다시 시도해주거나, 아래 링크에서 직접 확인해봐!

📌 **학사안내**: [이화 포탈](https://portal.ewha.ac.kr)
📌 **학적팀 연락처**: 02-3277-2114"""


# =============================================================================
# 프롬프트 템플릿
# =============================================================================

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
   "이 부분은 버디가 가진 자료에서 찾기 어려워요. 더 정확한 정보는 이화 포탈(portal.ewha.ac.kr)이나 학적팀(02-3277-2114)에 문의해봐!"라고 안내합니다.

[답변 형식 - 마크다운 구조화 필수]
답변은 반드시 아래 마크다운 형식을 사용하여 구조화하세요:

1. **요약 문장** (1-2줄): 질문의 핵심 답변을 먼저 제시
   - 예: "✅ 복수전공은 2학년 1학기부터 신청할 수 있어!"

2. **세부 내용**: 마크다운 서식 적극 활용
   - `### 제목` 또는 `**굵은 글씨**`로 섹션 구분
   - 항목이 여러 개면 `- 불릿 포인트` 또는 `1. 번호 목록` 사용
   - 조건/자격/날짜는 **굵게** 강조
   - 중요한 주의사항은 `> 인용구` 또는 `⚠️` 이모지 활용

3. **표(Table) 활용** (필요시):
   - 비교 정보나 단계별 정보는 표로 정리

4. **줄바꿈과 공백**:
   - 섹션 사이에 빈 줄 1개 추가 (가독성)
   - 긴 문단은 2-3문장마다 줄바꿈

5. **숫자/날짜/학점**:
   - 중요 숫자는 **굵게** 또는 `코드 블록`으로 강조

[말투 규칙]
- 반말을 사용하되, 친근하고 따뜻한 톤으로 말해주세요.
- 이모지는 적절히 사용하되 과하지 않게 (섹션 제목에 1개, 중요 포인트에 1-2개).
- "내가 정리해줌게!", "이거 중요해!", "궁금한 거 또 물어봐!" 같은 친근한 표현 사용.
- 신입생도 이해하기 쉬운 자연스러운 한국어로 설명합니다.
- 정보는 정확하게, 톤은 귀엽게!"""


# =============================================================================
# E5 임베딩 모델
# =============================================================================

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


# =============================================================================
# 하이브리드 검색기
# =============================================================================

class HybridRetriever:
    """하이브리드 검색기 (Semantic + BM25)"""
    
    def __init__(self, vectorstore: Chroma, documents: List[Document]):
        self.vectorstore = vectorstore
        self.documents = documents
        
        # BM25 인덱스 생성
        self.tokenized_corpus = [self._tokenize_with_metadata(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"하이브리드 검색기 초기화 완료 (문서 수: {len(documents)})")
    
    def _tokenize(self, text: str) -> List[str]:
        """한국어/영어 토큰화"""
        return re.findall(r'[가-힣]+|[a-zA-Z]+|[0-9]+', text.lower())
    
    def _tokenize_with_metadata(self, doc: Document) -> List[str]:
        """문서 본문 + 메타데이터 토큰화 (가중치 적용)"""
        content_tokens = self._tokenize(doc.page_content)
        metadata = doc.metadata or {}
        
        # topics 가중치 3배
        topics_str = metadata.get("topics", "")
        if topics_str:
            content_tokens.extend(self._tokenize(topics_str) * 3)
        
        # title 가중치 2배
        title = metadata.get("title", "")
        if title:
            content_tokens.extend(self._tokenize(title) * 2)
        
        return content_tokens
    
    def retrieve(self, query: str, k: int = 8, alpha: float = 0.5) -> List[Document]:
        """
        하이브리드 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            alpha: BM25 가중치 (0.0=semantic only, 1.0=bm25 only)
        """
        # 1. Semantic 검색
        semantic_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        semantic_dict = {id(doc): (doc, 1.0 - score) for doc, score in semantic_results}
        
        # 2. BM25 검색
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # 정규화
        max_score = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores = bm25_scores / max_score
        
        top_indices = np.argsort(bm25_scores)[::-1][:k*2]
        bm25_dict = {
            id(self.documents[i]): (self.documents[i], bm25_scores[i])
            for i in top_indices if bm25_scores[i] > 0
        }
        
        # 3. 하이브리드 스코어 계산
        all_doc_ids = set(semantic_dict.keys()) | set(bm25_dict.keys())
        hybrid_scores = {}
        
        for doc_id in all_doc_ids:
            semantic_score = semantic_dict.get(doc_id, (None, 0))[1]
            bm25_score = bm25_dict.get(doc_id, (None, 0))[1]
            hybrid_score = alpha * bm25_score + (1 - alpha) * semantic_score
            doc = semantic_dict.get(doc_id, bm25_dict.get(doc_id))[0]
            hybrid_scores[doc_id] = (doc, hybrid_score)
        
        # 정렬 후 상위 k개 반환
        sorted_docs = sorted(hybrid_scores.values(), key=lambda x: x[1], reverse=True)[:k]
        
        logger.info(f"하이브리드 검색 완료: {len(sorted_docs)}개 문서")
        return [doc for doc, _ in sorted_docs]


# =============================================================================
# GUIDE 엔진
# =============================================================================

class GuideEngine:
    """GUIDE 파이프라인용 RAG 엔진"""
    
    def __init__(self):
        self.embeddings: Optional[E5Embeddings] = None
        self.vectorstore: Optional[Chroma] = None
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self._initialize()
    
    def _initialize(self):
        """엔진 초기화"""
        try:
            # E5 임베딩 모델 초기화
            logger.info(f"E5 임베딩 모델 초기화 중: {settings.embedding_model}")
            self.embeddings = E5Embeddings(
                model_name=settings.embedding_model,
                device=settings.embedding_device
            )
            
            # ChromaDB 로드
            logger.info(f"ChromaDB 로드 중: {settings.chroma_persist_dir}")
            
            if not os.path.exists(settings.chroma_persist_dir):
                logger.warning(f"ChromaDB 디렉터리 없음: {settings.chroma_persist_dir}")
                logger.warning("먼저 데이터 임베딩을 실행해주세요.")
                self.vectorstore = Chroma(
                    persist_directory=settings.chroma_persist_dir,
                    embedding_function=self.embeddings
                )
                return
            
            self.vectorstore = Chroma(
                persist_directory=settings.chroma_persist_dir,
                embedding_function=self.embeddings
            )
            
            # BM25용 문서 로드
            logger.info("BM25 인덱스 구축 중...")
            all_docs = self.vectorstore.get()
            documents = []
            
            if all_docs and 'documents' in all_docs:
                for i, doc_text in enumerate(all_docs['documents']):
                    metadata = all_docs['metadatas'][i] if 'metadatas' in all_docs else {}
                    documents.append(Document(page_content=doc_text, metadata=metadata))
            
            # 하이브리드 검색기 초기화
            if documents:
                self.hybrid_retriever = HybridRetriever(self.vectorstore, documents)
                logger.info(f"GUIDE 엔진 초기화 완료 (문서 {len(documents)}개)")
            else:
                logger.warning("문서가 없어 하이브리드 검색기 미초기화")
                
        except Exception as e:
            logger.error(f"GUIDE 엔진 초기화 실패: {str(e)}")
            raise
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Gemini API 호출 (Fallback 체인 적용)"""
        import time
        
        last_error = None
        
        for model in MODEL_CHAIN:
            for attempt in range(2):  # 각 모델당 2번 재시도
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                    
                    data = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": settings.llm_temperature,
                            "maxOutputTokens": 8192,
                        }
                    }
                    
                    response = requests.post(
                        f"{url}?key={settings.google_api_key}",
                        headers={"Content-Type": "application/json"},
                        json=data,
                        timeout=30
                    )
                    
                    if response.status_code != 200:
                        logger.warning(f"[{model}] GUIDE API 응답 코드: {response.status_code}")
                        raise requests.exceptions.HTTPError(f"{response.status_code}: {response.text[:200]}")
                    
                    result = response.json()
                    
                    if "candidates" in result and result["candidates"]:
                        candidate = result["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            logger.info(f"[{model}] GUIDE API 성공")
                            return candidate["content"]["parts"][0]["text"]
                    
                    raise ValueError("API 응답 파싱 실패")
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"[{model}] GUIDE 시도 {attempt + 1} 실패: {str(e)[:100]}")
                    if attempt < 1:
                        time.sleep(1 * (attempt + 1))
            
            logger.warning(f"[{model}] 모든 재시도 실패, 다음 모델로 전환")
        
        # 모든 모델 실패
        logger.error(f"GUIDE API 모든 모델 호출 실패. 마지막 에러: {last_error}")
        return None  # None 반환하여 fallback 메시지 사용
    
    def get_answer(
        self,
        message: str,
        user_grade: Optional[int] = None,
        user_major: Optional[str] = None
    ) -> Tuple[str, List[dict]]:
        """
        GUIDE 질문에 대한 답변 생성
        
        Returns:
            (answer_text, source_docs) 튜플
        """
        logger.info(f"GUIDE 파이프라인 시작: {message[:50]}...")
        
        # 컨텍스트 쿼리 구성
        context_parts = []
        if user_grade:
            context_parts.append(f"{user_grade}학년")
        if user_major:
            context_parts.append(f"{user_major}과")
        
        context_query = f"{', '.join(context_parts)} 학생 질문: {message}" if context_parts else message
        
        # 문서 검색
        if self.hybrid_retriever:
            retrieved_docs = self.hybrid_retriever.retrieve(
                query=context_query,
                k=settings.top_k_results,
                alpha=0.4  # Semantic 60%, BM25 40%
            )
        else:
            retrieved_docs = self.vectorstore.similarity_search(
                context_query, 
                k=settings.top_k_results
            )
        
        logger.info(f"검색 완료: {len(retrieved_docs)}개 문서")
        
        # 컨텍스트 구성
        context_texts = []
        for i, doc in enumerate(retrieved_docs, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content
            context_texts.append(f"[문서{i}] (출처: {source})\n{content}")
        
        context = "\n\n---\n\n".join(context_texts)
        
        # LLM 호출 (실패 시 fallback 메시지)
        prompt = GUIDE_PROMPT_TEMPLATE.format(
            user_grade=user_grade or "미지정",
            user_major=user_major or "미지정",
            message=message,
            context=context
        )
        answer = self._call_gemini_api(prompt)
        if answer is None:
            answer = GUIDE_FALLBACK_MESSAGE
            logger.warning("GUIDE 파이프라인: fallback 메시지 사용")
        
        # 출처 문서 구성 (중복 제거)
        source_docs = []
        seen_urls = set()
        
        for doc in retrieved_docs[:5]:
            metadata = doc.metadata or {}
            url = metadata.get("source", "")
            
            if url and url not in seen_urls:
                source_docs.append({
                    "title": metadata.get("title", "학사안내 문서"),
                    "content": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                    "url": url,
                    "category": metadata.get("category_path"),
                    "relevance_score": None
                })
                seen_urls.add(url)
        
        logger.info(f"GUIDE 파이프라인 완료 (참조 문서 {len(source_docs)}개)")
        return answer, source_docs


# =============================================================================
# 전역 엔진 관리
# =============================================================================

_guide_engine: Optional[GuideEngine] = None


def get_guide_engine() -> GuideEngine:
    """전역 GUIDE 엔진 반환 (싱글톤)"""
    global _guide_engine
    if _guide_engine is None:
        _guide_engine = GuideEngine()
    return _guide_engine


def initialize_guide_engine():
    """GUIDE 엔진 전역 초기화"""
    global _guide_engine
    _guide_engine = GuideEngine()
    logger.info("GUIDE 엔진 전역 초기화 완료")


def answer_guide_question(
    message: str,
    user_grade: Optional[int] = None,
    user_major: Optional[str] = None
) -> Tuple[str, List[dict]]:
    """GUIDE 질문에 대한 답변 생성 (편의 함수)"""
    engine = get_guide_engine()
    return engine.get_answer(message, user_grade, user_major)
