# 이화여대 신입생 학사 챗봇 RAG 시스템 구조 문서

## 시스템 개요

### 목적
이화여대 신입생을 위한 학사 안내 챗봇으로, RAG(Retrieval-Augmented Generation) 기반의 질의응답 시스템

### 데이터 소스
```
data/official/
├── 학사안내/          # 9개 마크다운 파일 (등록, 수강신청, 전공, 학점, 졸업 등)
├── 수강신청/          # 25-2_course_registration.csv (수강신청 일정)
├── 학사일정/          # academic_calendar.csv (학사 일정)
└── etc/              # 기타 문서
```

### 주요 컴포넌트

#### 1. FastAPI 서버 (`app/main.py`)
- **엔드포인트**
  - `POST /ask`: 질문 처리 및 답변 반환
  - `GET /health`: 헬스체크
  - `GET /`: 루트 엔드포인트

- **요청/응답 모델** (`app/models.py`)
  ```python
  ChatRequest:
    - session_id: str
    - message: str
    - user_grade: Optional[int]  # 1-4
    - user_major: Optional[str]
  
  ChatResponse:
    - session_id: str
    - answer: str
    - sources: List[str]  # 참고 문서 경로
  ```

#### 2. RAG 엔진 (`app/rag_engine.py`)
- **E5Embeddings**: Multilingual-E5-Large-Instruct 임베딩
- **HybridRetriever**: 하이브리드 검색 (Semantic + BM25 + Keyword)
- **RAGEngine**: 전체 RAG 파이프라인 조정

#### 3. 설정 관리 (`app/config.py`)
- 환경변수 기반 설정 (pydantic-settings)
- Google API Key, 모델 설정, 벡터스토어 경로 등

---

## 데이터 파이프라인

### 임베딩 프로세스 (`scripts/embed_data.py`)

#### 1단계: 문서 로드
```python
# Markdown 문서
- DirectoryLoader로 .md 파일 로드
- UTF-8 인코딩
- 경로: data/official/**/*.md

# CSV 문서
- pandas로 읽기
- 50행씩 청크로 분할 (설정: csv_chunk_rows)
- 각 청크를 하나의 Document로 변환
```

**CSV 처리 방식의 문제점**:
- 50행씩 무작위로 묶어 하나의 문서로 만듦
- 행 간 의미적 연관성 무시
- 예: 수강신청 일정 CSV에서 "4학년 수강신청", "3학년 수강신청"이 서로 다른 청크에 분산될 수 있음

#### 2단계: 문서 분할 (Chunking)
```python
RecursiveCharacterTextSplitter:
  - chunk_size: 800 (설정값)
  - chunk_overlap: 150 (설정값)
  - separators: ["\n\n", "\n", " ", ""]
  
적용 대상: Markdown 문서만
CSV 문서: 이미 행 단위로 분할되어 추가 분할 안 함
```

**청킹 전략의 문제점**:
- Markdown 800자 vs CSV 50행 → 불균형
- CSV는 이미 청크화되어 추가 최적화 불가
- 중요 정보가 청크 경계에서 잘릴 수 있음

#### 3단계: 임베딩 및 저장
```python
E5Embeddings:
  - 모델: intfloat/multilingual-e5-large-instruct
  - 디바이스: CPU (설정 변경 가능)
  - 문서: "passage: {text}" 접두사
  - 쿼리: "query: {text}" 접두사
  
ChromaDB:
  - 저장 위치: ./backend/chroma_db
  - 컬렉션: 단일 컬렉션 (모든 문서 혼재)
```

**메타데이터 구조**:
```python
Markdown:
  - source: 상대 경로
  - type: "markdown"
  
CSV:
  - source: 상대 경로
  - type: "csv"
  - category: 상위 폴더 이름
  - file_name: 파일명
  - row_range: "1-50" 형태의 행 범위
```

---

## 검색 시스템

### HybridRetriever 구조

#### 1. Semantic Search (벡터 유사도)
```python
vectorstore.similarity_search_with_score(query, k=k*2)
- E5 임베딩 기반 코사인 유사도
- 상위 k*2개 문서 검색 (기본 16개)
- 점수 정규화: 1.0 - score
```

**Semantic Search의 장점**:
- 의미적으로 유사한 문서 검색 가능
- "휴학 신청 방법" → "휴학 절차" 문서 매칭

**Semantic Search의 문제점**:
- 정확한 날짜/숫자 정보에 약함
- "8월 5일 수강신청" 쿼리 → 날짜 정보 불일치 가능

#### 2. BM25 (키워드 검색)
```python
BM25Okapi(tokenized_corpus)
- 토큰화: re.findall(r'[가-힣]+|[a-zA-Z]+|[0-9]+', text.lower())
- 한국어 음절, 영어 단어, 숫자 단위 분리
- 점수 정규화: score / max(scores)
```

#### 3. Keyword Boosting
```python
키워드 패턴:
- 날짜: \d{4}-\d{2}-\d{2} | \d{1,2}월\s*\d{1,2}일
- 학년: [1-4]학년
- 전공: 경영, 컴퓨터, 의학, 약학 등 하드코딩된 9개
- 학사 키워드: 수강신청, 휴학, 복학, 졸업 등 8개
```

#### 4. 하이브리드 스코어 계산
```python
hybrid_score = (alpha * bm25_score + (1 - alpha) * semantic_score) + keyword_boost

alpha = 0.4 (BM25에 약간 더 가중치)
- BM25: 40%
- Semantic: 60%
- Keyword: 추가 부스트
```

#### 5. 검색 결과
```python
최종 반환: 상위 k개 문서 (기본 8개)
- 점수 기준 내림차순 정렬
- Document 객체 리스트
```

---

## 생성 시스템

### LLM: Gemini 2.5 Pro

```
Parameters:
  - temperature: 0.3 (낮음 → 일관된 답변)
  - maxOutputTokens: 8192
  - timeout: 30초
```

### 프롬프트 구조

#### 1. 컨텍스트 구성
```python
context_parts = []
for i, doc in enumerate(retrieved_docs, 1):
    source = doc.metadata.get("source", "Unknown")
    content = doc.page_content
    # 1000자 제한
    if len(content) > 1000:
        content = content[:1000] + "..."
    context_parts.append(f"[문서{i}] {content}")

context = "\n\n".join(context_parts)
```

#### 2. 프롬프트 템플릿
```python
prompt = f"""이화여대 학사 안내 챗봇입니다. 아래 문서를 바탕으로 정확하게 답변하세요.

{user_context}문서:
{context}

질문: {query}

답변 규칙:
- 문서의 날짜/시간 정보는 정확히 전달
- 간결하고 명확하게 답변
- 없는 정보는 "정보 없음" 명시

답변:"""
```

#### 3. 사용자 컨텍스트
```python
user_context = ""
if user_grade or user_major:
    context_parts = []
    if user_grade:
        context_parts.append(f"{user_grade}학년")
    if user_major:
        context_parts.append(f"{user_major}과")
    user_context = f"[질문자: {', '.join(context_parts)} 학생]\n"
```

### 출처 추출
```python
sources = []
for doc in retrieved_docs:
    if hasattr(doc, "metadata") and "source" in doc.metadata:
        source = doc.metadata["source"]
        if source not in sources:
            sources.append(source)
```

---

## 현재 구현의 문제점

### 🔴 Critical Issues

#### 1. 데이터 처리 문제
**문제**: CSV 50행 단위 임의 청크 분할
- **영향**: 연관된 정보가 분산됨
- **예시**: 수강신청 일정 CSV
  - "4학년 수강신청": 행 5-10
  - "3학년 수강신청": 행 11-16
  - → 서로 다른 청크로 분리되면 비교 불가
- **해결 방향**: 
  - CSV 구조 이해 후 의미 단위로 분할
  - 각 행을 독립 Document로 생성
  - 메타데이터에 카테고리 정보 강화

#### 2. 검색 품질 문제
**문제**: Semantic Search와 BM25의 불균형
- **Semantic Search**:
  - ✅ 개념 이해 우수
  - ❌ 정확한 날짜/숫자 약함
  - 예: "8월 5일" → "8월 초" 문서도 높은 점수
  
- **BM25**:
  - ✅ 키워드 매칭 정확
  - ❌ 토큰화 방식 단순 (음절 단위)
  - ❌ 형태소 분석 없음
  - 예: "수강신청하다" ≠ "수강신청"

- **하이브리드 가중치**:
  - alpha=0.4 고정 → 쿼리 유형 무시
  - 날짜 쿼리: BM25 우선해야 함
  - 개념 쿼리: Semantic 우선해야 함

**해결 방향**:
- 형태소 분석기 도입 (KoNLPy, Mecab)
- 쿼리 분류 → 동적 alpha 조정
- RRF(Reciprocal Rank Fusion) 적용

#### 3. 키워드 부스팅 한계
**문제**: 하드코딩된 전공/키워드 목록
```python
major_keywords = ['경영', '컴퓨터', '의학', '약학', '심리', 
                  '경제', '정치', '통계', '커뮤니케이션']  # 9개만
academic_keywords = ['수강신청', '휴학', '복학', '졸업', 
                     '등록', '장바구니', '채플', '학점교류']  # 8개만
```
- 이화여대 실제 전공: 약 70개
- 학사 관련 용어: 수백 개
- **영향**: 대부분의 전공/키워드가 부스팅 안 됨

**해결 방향**:
- 전체 전공 목록 데이터베이스화
- NER(Named Entity Recognition) 도입
- 도메인 특화 사전 구축

#### 4. 프롬프트 엔지니어링 부족
**문제**: 지시문이 너무 단순
```python
"""답변 규칙:
- 문서의 날짜/시간 정보는 정확히 전달
- 간결하고 명확하게 답변
- 없는 정보는 "정보 없음" 명시"""
```

**부족한 요소**:
- ❌ 답변 형식 지정 (단락, 리스트 등)
- ❌ 출처 인용 방법
- ❌ Few-shot 예시
- ❌ 학년/전공별 맞춤 정보 제공 지시
- ❌ 모호한 질문 처리 방법

**해결 방향**:
- Chain-of-Thought 프롬프팅
- 출처 번호 기반 인용 ([문서1], [문서2])
- 답변 템플릿 제공
- Few-shot 예시 추가

#### 5. 출처 추적 문제
**문제**: 답변과 출처의 연결 불명확
```python
sources = []  # 단순 파일 경로 리스트
for doc in retrieved_docs:
    sources.append(doc.metadata["source"])
```

- 답변의 어떤 문장이 어떤 출처에서 왔는지 알 수 없음
- 사용자 신뢰도 저하

**해결 방향**:
- 답변 내 인용 번호 삽입 ([1], [2])
- 문서별 인용 구간 추적
- 출처 요약 정보 제공

### 🟡 Performance Issues

#### 6. 컨텍스트 윈도우 활용
**문제**: 1000자 제한으로 정보 손실
```python
if len(content) > 1000:
    content = content[:1000] + "..."
```
- 8개 문서 × 1000자 = 8000자
- Gemini 2.5 Pro: 최대 1,048,576 토큰 (약 300만 자)
- **현재 활용률**: 0.3% 미만

**해결 방향**:
- 문서 길이 제한 완화 또는 제거
- 동적 컨텍스트 압축
- 긴 문서는 요약 후 전체 제공

#### 7. 에러 핸들링 부족
**문제**: API 오류 처리 단순
- 타임아웃: 30초 고정
- 재시도 로직 없음
- Rate limit 처리 없음

**해결 방향**:
- Exponential backoff 재시도
- Fallback 응답 메커니즘
- 에러별 맞춤 처리

#### 8. 청크 크기 불균형
**문제**: 
- Markdown: 800자 청크 (RecursiveCharacterTextSplitter)
- CSV: 50행 청크 (고정)

**영향**:
- 문서 타입별 검색 정확도 불균형
- CSV 청크가 너무 클 수 있음 (50행 × 200자/행 = 10,000자)

**해결 방향**:
- 문서 타입별 최적 청크 크기 실험
- 동적 청크 크기 조정
- 청크 품질 메트릭 도입

### 🟢 Minor Issues

#### 9. 메타데이터 활용 부족
**현재 메타데이터**:
```python
{
  "source": "경로",
  "type": "csv/markdown",
  "category": "폴더명",
  "file_name": "파일명",
  "row_range": "1-50"  # CSV만
}
```

---

## 기술 스택

### Backend
- **Framework**: FastAPI 0.109.0+
- **Language**: Python 3.11+
- **ASGI Server**: Uvicorn

### RAG Components
- **Vector Store**: ChromaDB 0.4.22+
- **Embeddings**: 
  - 모델: `intfloat/multilingual-e5-large-instruct`
  - 라이브러리: Sentence Transformers 2.2.0+
- **Keyword Search**: rank-bm25 0.2.2+
- **LLM**: Google Gemini 2.5 Pro (API)

### LangChain
- **Core**: langchain-core 0.3.0+
- **Community**: langchain-community 0.3.0+
- **Text Splitters**: langchain-text-splitters 0.3.0+
- **Chroma Integration**: langchain-chroma 0.1.0+

### Data Processing
- **Pandas**: 2.0.0+
- **NumPy**: 1.24.0+

### Configuration
- **Pydantic**: 2.5.0+ (데이터 검증)
- **Pydantic Settings**: 2.1.0+ (환경변수)
- **Python-dotenv**: 1.0.0+ (.env 파일)


---

## 설정 파일 (`app/config.py`)

```python
class Settings:
    # Google API
    google_api_key: str
    
    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    
    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # LLM 설정
    llm_model: str = "gemini-2.5-pro"
    llm_temperature: float = 0.3
    
    # 임베딩 모델
    embedding_model: str = "intfloat/multilingual-e5-large-instruct"
    embedding_device: str = "cpu"
    
    # RAG 설정
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_results: int = 8
    
    # CSV 처리
    csv_chunk_rows: int = 50
```

---

## 실행 흐름

### 1. 임베딩 단계 (오프라인)
```bash
cd backend
python -m scripts.embed_data
```

**과정**:
1. `data/official/` 디렉터리 스캔
2. Markdown 파일 로드 → 800자 청크 분할
3. CSV 파일 로드 → 50행 청크 생성
4. E5 모델로 임베딩 생성
5. ChromaDB에 저장 (`./backend/chroma_db`)

### 2. 서버 시작
```bash
python -m app.main
# 또는
./run_dev.sh
```

**초기화 과정**:
1. RAGEngine 인스턴스 생성
2. E5Embeddings 로드
3. ChromaDB 로드
4. HybridRetriever 초기화 (BM25 인덱스 구축)
5. Gemini API 설정

### 3. 질의응답 처리
```
POST /ask
{
  "session_id": "abc123",
  "message": "4학년 수강신청은 언제인가요?",
  "user_grade": 4,
  "user_major": "컴퓨터공학과"
}
```

**처리 과정**:
1. **쿼리 전처리**
   ```python
   context_query = "4학년, 컴퓨터공학과 학생 질문: 4학년 수강신청은 언제인가요?"
   ```

2. **하이브리드 검색**
   - Semantic Search: 16개 후보 검색
   - BM25 Search: 16개 후보 검색
   - Keyword Boosting: 날짜, 학년 패턴 감지
   - 하이브리드 스코어 계산 (alpha=0.4)
   - 상위 8개 선택

3. **컨텍스트 구성**
   ```python
   [문서1] 25-2_course_registration.csv 내용 (1000자 제한)
   [문서2] 학사안내/수강신청.md 내용 (1000자 제한)
   ...
   [문서8] ...
   ```

4. **프롬프트 생성**
   ```python
   """이화여대 학사 안내 챗봇입니다. 아래 문서를 바탕으로 정확하게 답변하세요.
   
   [질문자: 4학년, 컴퓨터공학과 학생]
   문서:
   [문서1] ...
   [문서8] ...
   
   질문: 4학년 수강신청은 언제인가요?
   
   답변 규칙:
   - 문서의 날짜/시간 정보는 정확히 전달
   - 간결하고 명확하게 답변
   - 없는 정보는 "정보 없음" 명시
   
   답변:"""
   ```

5. **LLM 호출** (Gemini 2.5 Pro)
   - 타임아웃: 30초
   - Temperature: 0.3
   - Max tokens: 8192

6. **응답 반환**
   ```json
   {
     "session_id": "abc123",
     "answer": "4학년 수강신청은 2025년 8월 5일 09:00~12:00입니다...",
     "sources": [
       "data/official/수강신청/25-2_course_registration.csv",
       "data/official/학사안내/4-academic-course.md"
     ]
   }
   ```

---

## 성능 개선을 위한 우선순위

### ⭐⭐⭐ High Priority (즉시 개선 필요)
1. **CSV 청킹 전략 재설계**
   - 50행 고정 → 의미 단위 또는 행별 독립 Document
   - 메타데이터 강화 (학년, 전공, 날짜 범위 등)

2. **형태소 분석 도입**
   - KoNLPy Mecab 또는 Kiwi
   - BM25 토큰화 개선

3. **프롬프트 엔지니어링**
   - Chain-of-Thought 프롬프팅
   - Few-shot 예시 추가
   - 출처 인용 형식 명시

4. **전공/키워드 데이터베이스**
   - 하드코딩 제거
   - 전체 전공 목록 로드
   - NER 기반 동적 키워드 추출

### ⭐⭐ Medium Priority (단기 개선)
5. **동적 하이브리드 가중치**
   - 쿼리 분류 (날짜/개념/절차 등)
   - 쿼리 유형별 alpha 조정

6. **컨텍스트 윈도우 최적화**
   - 1000자 제한 완화
   - 동적 컨텍스트 압축
   - 문서 요약 기능

7. **에러 핸들링 강화**
   - Exponential backoff 재시도
   - Rate limit 처리
   - Fallback 응답

### ⭐ Low Priority (중장기 개선)
8. **메타데이터 확장**
   - 문서 중요도, 작성일, 대상 학년/전공
   - 메타데이터 기반 필터링

9. **로깅 및 모니터링**
   - 검색 품질 메트릭
   - 응답 시간 측정
   - 사용자 피드백 수집

10. **Reranking 모델 추가**
    - Cross-encoder 기반 재정렬
    - 검색 정확도 향상

---

## 참고: 디렉터리 구조
```
ewha-fresh-db/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI 서버
│   │   ├── models.py            # Request/Response 모델
│   │   ├── config.py            # 설정
│   │   └── rag_engine.py        # RAG 핵심 로직
│   ├── scripts/
│   │   ├── __init__.py
│   │   └── embed_data.py        # 임베딩 스크립트
│   ├── chroma_db/               # 벡터스토어 (gitignore)
│   ├── models/                  # E5 모델 캐시
│   ├── pyproject.toml           # 의존성
│   └── .env                     # 환경변수
└── data/
    └── official/
        ├── 학사안내/            # 9개 .md 파일
        ├── 수강신청/            # .csv
        ├── 학사일정/            # .csv
        └── etc/
```

---

## 결론

현재 RAG 시스템은 **기본적인 질의응답은 가능하지만**, 다음과 같은 핵심 문제로 인해 **성능이 저하**되고 있습니다:

1. **데이터 처리**: CSV 청킹 방식의 비효율성
2. **검색 정확도**: 단순 토큰화 및 고정 가중치
3. **프롬프트**: 지시문 부족 및 출처 추적 미흡
4. **확장성**: 하드코딩된 키워드 및 전공 목록

**개선 우선순위**는 **CSV 재처리 → 형태소 분석 → 프롬프트 개선** 순서로 진행하는 것을 권장합니다.