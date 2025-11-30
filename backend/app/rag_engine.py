import os
from typing import List, Tuple, Optional, Dict
import logging
import re
from collections import Counter

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
import requests
import json

from app.config import settings

logger = logging.getLogger(__name__)


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
    """하이브리드 검색기 (Semantic + BM25 + Reranking)"""
    
    def __init__(self, vectorstore: Chroma, documents: List[Document]):
        self.vectorstore = vectorstore
        self.documents = documents
        
        # BM25 인덱스 생성
        self.tokenized_corpus = [self._tokenize(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"하이브리드 검색기 초기화 완료 (문서 수: {len(documents)})")
    
    def _tokenize(self, text: str) -> List[str]:
        """한국어/영어 토큰화"""
        # 한국어 음절, 영어 단어, 숫자를 토큰으로 분리
        tokens = re.findall(r'[가-힣]+|[a-zA-Z]+|[0-9]+', text.lower())
        return tokens
    
    def _extract_keywords(self, query: str) -> List[str]:
        """쿼리에서 핵심 키워드 추출"""
        # 날짜 패턴 (YYYY-MM-DD, MM월 DD일 등)
        date_patterns = re.findall(r'\d{4}-\d{2}-\d{2}|\d{1,2}월\s*\d{1,2}일', query)
        
        # 학년 패턴
        grade_patterns = re.findall(r'[1-4]학년', query)
        
        # 전공/학과 패턴
        major_keywords = ['경영', '컴퓨터', '의학', '약학', '심리', '경제', '정치', '통계', '커뮤니케이션']
        found_majors = [m for m in major_keywords if m in query]
        
        # 학사 관련 키워드
        academic_keywords = ['수강신청', '휴학', '복학', '졸업', '등록', '장바구니', '채플', '학점교류']
        found_academic = [k for k in academic_keywords if k in query]
        
        keywords = date_patterns + grade_patterns + found_majors + found_academic
        return keywords
    
    def retrieve(self, query: str, k: int = 8, alpha: float = 0.5) -> List[Document]:
        """
        하이브리드 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            alpha: semantic(0.0) vs bm25(1.0) 가중치 (0.5 = 동등)
        """
        # 1. Semantic 검색 (Vector)
        semantic_docs = self.vectorstore.similarity_search_with_score(query, k=k*2)
        semantic_dict = {id(doc): (doc, 1.0 - score) for doc, score in semantic_docs}  # score normalize
        
        # 2. BM25 검색 (Keyword)
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # BM25 점수 정규화
        if max(bm25_scores) > 0:
            bm25_scores = bm25_scores / max(bm25_scores)
        
        # 상위 k*2개 선택
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k*2]
        bm25_dict = {id(self.documents[i]): (self.documents[i], bm25_scores[i]) 
                     for i in top_bm25_indices if bm25_scores[i] > 0}
        
        # 3. 키워드 부스팅
        keywords = self._extract_keywords(query)
        keyword_boost = {}
        if keywords:
            for doc_id, (doc, _) in {**semantic_dict, **bm25_dict}.items():
                boost = sum(1 for kw in keywords if kw in doc.page_content)
                keyword_boost[doc_id] = boost * 0.2  # 키워드당 20% 가중치
        
        # 4. 하이브리드 스코어 계산 (RRF - Reciprocal Rank Fusion)
        all_doc_ids = set(semantic_dict.keys()) | set(bm25_dict.keys())
        hybrid_scores = {}
        
        for doc_id in all_doc_ids:
            semantic_score = semantic_dict.get(doc_id, (None, 0))[1]
            bm25_score = bm25_dict.get(doc_id, (None, 0))[1]
            keyword_score = keyword_boost.get(doc_id, 0)
            
            # 가중 평균 + 키워드 부스트
            hybrid_score = (alpha * bm25_score + (1 - alpha) * semantic_score) + keyword_score
            
            # 문서 가져오기
            doc = semantic_dict.get(doc_id, bm25_dict.get(doc_id))[0]
            hybrid_scores[doc_id] = (doc, hybrid_score)
        
        # 5. 점수 기준 정렬 및 상위 k개 반환
        sorted_docs = sorted(hybrid_scores.values(), key=lambda x: x[1], reverse=True)[:k]
        
        logger.info(f"하이브리드 검색 완료: Semantic {len(semantic_dict)}개, BM25 {len(bm25_dict)}개, "
                   f"키워드 매칭 {len(keyword_boost)}개 → 최종 {len(sorted_docs)}개")
        
        return [doc for doc, score in sorted_docs]


class RAGEngine:
    """하이브리드 RAG 기반 질의응답 엔진"""
    
    def __init__(self):
        """RAG 엔진 초기화"""
        self.embeddings = None
        self.vectorstore = None
        self.hybrid_retriever = None
        self.llm_api_key = None
        self.llm_model_name = None
        
        self._initialize()
    
    def _initialize(self):
        """구성 요소 초기화"""
        try:
            # 임베딩 모델 초기화 (E5)
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
            
            # LLM 초기화 (Gemini API)
            logger.info(f"LLM 모델 초기화 중: {settings.llm_model}")
            self.llm_api_key = settings.google_api_key
            self.llm_model_name = settings.llm_model
            
            logger.info("RAG 엔진 초기화 완료")
            
        except Exception as e:
            logger.error(f"RAG 엔진 초기화 실패: {str(e)}")
            raise
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Gemini API 직접 호출"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.llm_model_name}:generateContent"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": settings.llm_temperature,
                "maxOutputTokens": 8192,  # 더 긴 응답 허용
            }
        }
        
        try:
            response = requests.post(
                f"{url}?key={self.llm_api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # 응답 구조 로깅
            logger.debug(f"Gemini API 응답: {result}")
            
            # 응답 파싱 (여러 구조 지원)
            if "candidates" in result and len(result["candidates"]) > 0:
                candidate = result["candidates"][0]
                
                # finishReason 확인
                finish_reason = candidate.get("finishReason", "UNKNOWN")
                if finish_reason == "MAX_TOKENS":
                    logger.warning("응답이 최대 토큰에 도달하여 잘렸습니다. 프롬프트를 줄이거나 maxOutputTokens를 늘려주세요.")
                
                # 구조 1: content.parts[0].text
                if "content" in candidate:
                    content = candidate["content"]
                    if "parts" in content and len(content["parts"]) > 0:
                        return content["parts"][0]["text"]
                    
                    # parts가 없지만 role만 있는 경우 (오류)
                    if "role" in content and "parts" not in content:
                        logger.error(f"응답에 텍스트 없음 (finishReason: {finish_reason})")
                        return "죄송합니다. 응답 생성 중 문제가 발생했습니다. 질문을 더 간단하게 해주시겠어요?"
                
                # 구조 2: text 직접 존재
                if "text" in candidate:
                    return candidate["text"]
                
                # 구조 3: output 필드
                if "output" in candidate:
                    return candidate["output"]
            
            # 파싱 실패 시 전체 응답 로깅
            logger.error(f"예상치 못한 API 응답 구조: {json.dumps(result, ensure_ascii=False)}")
            raise ValueError(f"Gemini API 응답 파싱 실패: {result}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API 호출 실패: {str(e)}")
            raise
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Gemini API 응답 파싱 오류: {str(e)}")
            raise ValueError(f"API 응답 형식 오류: {str(e)}")
    
    def get_answer(
        self,
        query: str,
        user_grade: Optional[int] = None,
        user_major: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        질문에 대한 답변 생성
        
        Args:
            query: 사용자 질문
            user_grade: 사용자 학년 (선택)
            user_major: 사용자 전공 (선택)
        
        Returns:
            (답변, 출처 목록) 튜플
        """
        try:
            # 사용자 컨텍스트 추가 (학년/전공 정보)
            context_query = query
            user_context = ""
            if user_grade or user_major:
                context_parts = []
                if user_grade:
                    context_parts.append(f"{user_grade}학년")
                if user_major:
                    context_parts.append(f"{user_major}과")
                user_context = f"[질문자: {', '.join(context_parts)} 학생]\n"
                context_query = f"{', '.join(context_parts)} 학생 질문: {query}"
            
            # 하이브리드 검색
            logger.info(f"하이브리드 검색 시작: {context_query}")
            
            if self.hybrid_retriever:
                # 하이브리드 검색 사용 (semantic + BM25 + keyword boosting)
                retrieved_docs = self.hybrid_retriever.retrieve(
                    query=context_query,
                    k=settings.top_k_results,
                    alpha=0.4  # BM25에 약간 더 가중치 (키워드 매칭 중요)
                )
            else:
                # Fallback: 순수 semantic 검색
                logger.warning("하이브리드 검색기 미초기화, Semantic 검색만 사용")
                retrieved_docs = self.vectorstore.similarity_search(context_query, k=settings.top_k_results)
            
            logger.info(f"검색 완료: {len(retrieved_docs)}개 문서")
            
            # 컨텍스트 구성 (간결화)
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                source = doc.metadata.get("source", "Unknown")
                content = doc.page_content
                # 너무 긴 문서는 잘라내기 (토큰 절약)
                if len(content) > 1000:
                    content = content[:1000] + "..."
                context_parts.append(f"[문서{i}] {content}")
            
            context = "\n\n".join(context_parts)
            
            # 프롬프트 생성 (간결화)
            prompt = f"""이화여대 학사 안내 챗봇입니다. 아래 문서를 바탕으로 정확하게 답변하세요.

{user_context}문서:
{context}

질문: {query}

답변 규칙:
- 문서의 날짜/시간 정보는 정확히 전달
- 간결하고 명확하게 답변
- 없는 정보는 "정보 없음" 명시

답변:"""
            
            # LLM 호출
            answer = self._call_gemini_api(prompt)
            
            # 출처 문서 추출
            sources = []
            for doc in retrieved_docs:
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    source = doc.metadata["source"]
                    if source not in sources:
                        sources.append(source)
            
            return answer, sources
            
        except Exception as e:
            logger.error(f"답변 생성 중 오류 발생: {str(e)}")
            raise
    
    def add_documents(self, documents: List[str], metadatas: List[dict]):
        """
        새로운 문서를 벡터스토어에 추가
        
        Args:
            documents: 문서 텍스트 리스트
            metadatas: 각 문서의 메타데이터 리스트
        """
        try:
            self.vectorstore.add_texts(
                texts=documents,
                metadatas=metadatas
            )
            logger.info(f"{len(documents)}개 문서 추가 완료")
        except Exception as e:
            logger.error(f"문서 추가 중 오류 발생: {str(e)}")
            raise
