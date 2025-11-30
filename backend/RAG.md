# 이화여대 학사 챗봇 RAG 시스템 구조 문서

> **버디(Buddy)** - 이화여대 학생들을 도와주는 귀여운 곰돌이 학사 안내 챗봇 🐻

## 시스템 개요

### 목적
이화여대 학생을 위한 학사 안내 챗봇으로, **Intent 기반 이원화 RAG** 시스템

### 핵심 설계 원칙
```
정형 데이터(CSV) → LLM에 전체 전달하여 필터링/분석
비정형 데이터(Markdown) → 기존 RAG(벡터 검색)로 처리
```

### 데이터 소스
```
data/official/
├── 학사안내/              # 10개 마크다운 파일 (등록, 수강신청, 전공, 학점, 졸업 등)
│   ├── 1-academic-enrollment.md
│   ├── 2-academic-register.md
│   └── ... (총 10개)
├── 수강신청/              # CSV (SCHEDULE 파이프라인에서 직접 사용)
│   └── 25-2_course_registration.csv
├── 학사일정/              # CSV (SCHEDULE 파이프라인에서 직접 사용)
│   └── academic_calendar.csv
└── etc/
```

---

## 모듈 구조

### 파일 구성
```
backend/app/
├── main.py              # FastAPI 서버, /ask 엔드포인트
├── models.py            # Request/Response Pydantic 모델
├── config.py            # 환경변수 설정
├── intent_router.py     # Intent 분류기 (SCHEDULE/GUIDE/OTHER)
├── csv_loader.py        # CSV 텍스트 로더 (전역 싱글톤)
├── schedule_pipeline.py # SCHEDULE 파이프라인 (CSV + LLM)
├── guide_pipeline.py    # GUIDE 파이프라인 (Markdown RAG + LLM)
└── rag_engine.py        # (레거시) 기존 통합 RAG 엔진

backend/scripts/
└── embed_data.py        # Markdown 임베딩 스크립트 (CSV 제외)
```

---

## 1. Intent Router (`intent_router.py`)

### 역할
사용자 질문을 **SCHEDULE**, **GUIDE**, **OTHER** 중 하나로 분류

### Intent 정의
```python
class Intent(str, Enum):
    SCHEDULE = "SCHEDULE"  # 일정/수강신청, 채플, 계절학기, 공휴일 등
    GUIDE = "GUIDE"        # 규정/절차/제도 설명 (휴학, 졸업요건 등)
    OTHER = "OTHER"        # 잡담/기타
```

### 분류 전략: 2단계 하이브리드

#### 1단계: 키워드 Pre-check (빠른 분기)
```python
SCHEDULE_KEYWORDS = [
    # 수강신청 관련
    "수강신청", "장바구니", "수강정정", "수강철회", "우선수강", "본수강",
    # 학사일정 관련
    "학사일정", "개강", "종강", "등록기간",
    # 채플 관련 (학사일정 CSV에 규정 포함)
    "채플", "보충채플", "특별채플",
    # 기타 이벤트
    "계절학기", "여름학기", "겨울학기", "공휴일", "휴무일",
    "입학식", "졸업식", "학위수여", "복학", "휴학 마감",
    # 시간/횟수 표현
    "언제", "기간", "날짜", "시간", "일정", "마감", "몇 번", "몇번"
]
```
- 키워드 매칭 시 즉시 `SCHEDULE` 반환 (LLM 호출 생략)

#### 2단계: LLM 라우터 (애매한 경우)
```python
ROUTER_PROMPT_TEMPLATE = """[시스템 역할]
당신은 이화여자대학교 학사 챗봇의 질문 분류기입니다.
...
[중요]
- 채플 관련 질문(결석 허용 횟수, 보충채플, 채플 기간 등)은 SCHEDULE입니다.
- "몇 번", "몇 회" 같은 횟수 질문도 학사일정과 관련되면 SCHEDULE입니다.

반드시 SCHEDULE, GUIDE, OTHER 중 하나의 단어만 출력하세요.
"""
```
- Temperature: 0.1 (일관된 분류)
- Timeout: 10초
- 실패 시 기본값: `OTHER`

---

## 2. CSV Loader (`csv_loader.py`)

### 역할
수강신청/학사일정 CSV를 **텍스트 전체로** 로드하여 전역 싱글톤으로 보관

### 로드 대상
```python
CSV_TEXTS = {
    "course_registration": "25-2_course_registration.csv 전체 텍스트",
    "academic_calendar": "academic_calendar.csv 전체 텍스트"
}
```

### 초기화 시점
- 서버 시작 시 `lifespan` 이벤트에서 `initialize_csv_texts()` 호출
- 한 번 로드 후 메모리에 유지

### 설계 이유
```
기존 문제: CSV를 50행씩 청킹 → 연관 정보 분산 → 검색 품질 저하
해결책: CSV 전체를 LLM에 전달 → LLM이 직접 필터링/분석
```

---

## 3. SCHEDULE 파이프라인 (`schedule_pipeline.py`)

### 역할
수강신청/학사일정/채플 등 **일정 관련 질문**을 CSV 전체 + LLM으로 처리

### 처리 흐름
```
1. CSV 전체 텍스트 (course_registration + academic_calendar)
2. 사용자 정보 (학년, 전공, 오늘 날짜)
3. 질문
   ↓
   SCHEDULE_PROMPT_TEMPLATE
   ↓
   Gemini 2.5 Pro
   ↓
4. 필터링된 답변 + 고정 출처
```

### 프롬프트 핵심 요소

#### 역할 설정
```
당신은 이화여자대학교 학사 안내 챗봇 "버디"입니다.
버디는 이화여대 학생들을 도와주는 귀여운 곰돌이 캐릭터입니다.
```

#### 컨텍스트 제공
```
[오늘 날짜] 2025년 11월 30일 (Saturday)
[사용자 정보] 학년: 4, 전공: 컴공
[질문] ...
[수강신청 일정표 CSV 전체] ...
[학사일정 CSV 전체] ...
```

#### 처리 규칙
1. **학년/전공 조건 필터링**: 사용자 조건에 맞는 행만 추출
2. **notes 컬럼 확인**: 채플 결석 규정 등 숨은 규정 정보 활용
3. **채플 특수 처리**: 7학기 이상 규정, 공휴일 제외, 보충채플 안내
4. **계산 수행**: 채플 횟수, 수업일 수 등 실제 계산

#### 말투 규칙
```
- 반말(존댓말) + 친근하고 따뜻한 톤
- 이모지 적절히 사용 (시작/끝에 1개씩)
- "내가 정리해줄게!", "이거 중요해!", "화이팅!" 표현 사용
- 정보는 정확하게, 톤은 귀엽게!
```

### 출처 반환
```python
sources = [
    "official/수강신청/25-2_course_registration.csv",
    "official/학사일정/academic_calendar.csv",
]
```
- SCHEDULE은 항상 고정 출처 반환

---

## 4. GUIDE 파이프라인 (`guide_pipeline.py`)

### 역할
휴학/복학/졸업요건 등 **학사 제도/규정 질문**을 Markdown RAG + LLM으로 처리

### 처리 흐름
```
1. 사용자 질문
   ↓
2. HybridRetriever (Semantic + BM25)
   ↓
3. 상위 k개 Markdown 청크 검색
   ↓
4. GUIDE_PROMPT_TEMPLATE + 검색 결과
   ↓
5. Gemini 2.5 Pro
   ↓
6. 답변 + 실제 참조 출처
```

### HybridRetriever 구조

#### Semantic Search (벡터 유사도)
```python
vectorstore.similarity_search_with_score(query, k=k*2)
# E5 임베딩 기반 코사인 유사도
```

#### BM25 (키워드 검색)
```python
BM25Okapi(tokenized_corpus)
# 토큰화: 한국어 음절, 영어 단어, 숫자 분리
```

#### 하이브리드 스코어
```python
hybrid_score = alpha * bm25_score + (1 - alpha) * semantic_score
# alpha = 0.4 (BM25 40%, Semantic 60%)
```

### 프롬프트
```python
GUIDE_PROMPT_TEMPLATE = """당신은 이화여자대학교 학사 안내 챗봇 "버디"입니다.
버디는 이화여대 학생들을 도와주는 귀여운 곰돌이 캐릭터입니다.
...
[말투 규칙]
- 반말(존댓말) + 친근하고 따뜻한 톤
- "내가 정리해줄게!", "이거 중요해!", "궁금한 거 또 물어봐!"
- 정보는 정확하게, 톤은 귀엽게!
"""
```

### 출처 반환
```python
sources = [doc.metadata["source"] for doc in retrieved_docs]
# 실제 검색된 Markdown 파일 경로
```

---

## 5. 데이터 임베딩 (`scripts/embed_data.py`)

### 변경사항 (기존 대비)
```diff
- CSV + Markdown 모두 벡터 DB에 저장
+ Markdown만 벡터 DB에 저장
+ CSV는 SCHEDULE 파이프라인에서 직접 사용
```

### 임베딩 대상
```python
# Markdown 문서만 로드
loader = DirectoryLoader(
    data_dir,
    glob="**/*.md",  # .md 파일만
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
```

### 청킹 전략
```python
RecursiveCharacterTextSplitter:
  - chunk_size: 800
  - chunk_overlap: 150
  - separators: ["\n\n", "\n", " ", ""]
```

### 실행 방법
```bash
cd backend
python -m scripts.embed_data
```

---

## 6. FastAPI 서버 (`main.py`)

### 엔드포인트

#### `POST /ask`
```python
Request:
{
    "session_id": "string",
    "message": "4학년 수강신청 언제야?",
    "user_grade": 4,
    "user_major": "컴퓨터공학"
}

Response:
{
    "session_id": "string",
    "answer": "🐻 안녕! 4학년 수강신청 일정 정리해줄게! ...",
    "sources": ["official/수강신청/25-2_course_registration.csv", ...]
}
```

#### `GET /health`
```json
{"status": "healthy", "message": "서비스가 정상 작동 중입니다."}
```

### 서버 초기화 (lifespan)
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. CSV 텍스트 로드 (SCHEDULE 파이프라인용)
    initialize_csv_texts()
    
    # 2. GUIDE 엔진 초기화 (Markdown RAG)
    initialize_guide_engine()
    
    yield
```

### /ask 처리 로직
```python
# 1. Intent 분류
intent = classify_intent(message, user_grade, user_major)

# 2. Intent별 파이프라인 분기
if intent == Intent.SCHEDULE:
    answer, sources = answer_schedule_question(...)
elif intent == Intent.GUIDE:
    answer, sources = answer_guide_question(...)
else:  # OTHER
    answer = "안녕! 나는 이화여대 학사 안내 곰돌이 버디야 🐻 ..."
    sources = []

# 3. 응답 반환
return ChatResponse(session_id, answer, sources)
```

---

## 기술 스택

### Backend
| 구성요소 | 기술 |
|---------|------|
| Framework | FastAPI |
| Language | Python 3.11+ |
| ASGI Server | Uvicorn |

### RAG Components
| 구성요소 | 기술 |
|---------|------|
| Vector Store | ChromaDB |
| Embeddings | `intfloat/multilingual-e5-large-instruct` |
| Keyword Search | rank-bm25 (BM25Okapi) |
| LLM | Google Gemini 2.5 Pro |

### LangChain
- langchain-core, langchain-community
- langchain-text-splitters
- langchain-chroma

---

## 설정 (`config.py`)

```python
class Settings:
    # Google API
    google_api_key: str
    
    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    
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
```

---

## 캐릭터 설정: 버디 🐻

### 컨셉
- **이름**: 버디 (Buddy)
- **종족**: 귀여운 곰돌이
- **역할**: 이화여대 학사 안내 챗봇

### 말투 특징
| 요소 | 설명 |
|------|------|
| 어조 | 반말(존댓말), 친근하고 따뜻함 |
| 이모지 | 적절히 사용 (🐻💚 등), 과하지 않게 |
| 표현 | "내가 정리해줄게!", "이거 중요해!", "화이팅!" |
| 원칙 | **정보는 정확하게, 톤은 귀엽게!** |

### 기본 응답 (OTHER)
```
안녕! 나는 이화여대 학사 안내 곰돌이 버디야 🐻 
수강신청 일정이나 학사 제도에 대해 궁금한 게 있으면 편하게 물어봐! 
내가 도와줄게 💚
```