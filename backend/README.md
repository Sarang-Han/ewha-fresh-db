# 이화여대 신입생 학사 챗봇 백엔드

RAG 기반 학사 안내 챗봇 백엔드 API

## 기술 스택

- **언어**: Python 3.11+
- **웹 프레임워크**: FastAPI
- **벡터 DB**: ChromaDB
- **RAG**: LangChain
- **LLM**: Google Gemini 2.5 Flash
- **임베딩**: Google text-embedding-004
- **패키지 관리**: uv

## 프로젝트 구조

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI 애플리케이션 진입점
│   ├── config.py         # 설정 관리
│   ├── models.py         # Pydantic 모델
│   └── rag_engine.py     # RAG 엔진 구현
├── scripts/
│   ├── __init__.py
│   └── embed_data.py     # 데이터 임베딩 스크립트
├── chroma_db/            # ChromaDB 저장소 (자동 생성)
├── pyproject.toml        # 의존성 정의
├── Dockerfile
├── .env.example
└── README.md
```

## Docker 실행

### 로컬 Docker

```bash
# 이미지 빌드
docker build -t ewha-chatbot-backend ./backend

# 컨테이너 실행
docker run -p 8000:8000 \
  -e GOOGLE_API_KEY=your_api_key \
  -v $(pwd)/backend/chroma_db:/app/chroma_db \
  ewha-chatbot-backend
```

### Docker Compose

```bash
# 프로젝트 루트에서 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f backend

# 중지
docker-compose down
```

## API 엔드포인트

### POST /ask

챗봇에 질문하고 답변을 받습니다.

**요청 예시:**
```json
{
  "session_id": "user-session-123",
  "message": "휴학은 어떻게 신청하나요?",
  "user_grade": 1,
  "user_major": "컴퓨터공학"
}
```

**응답 예시:**
```json
{
  "session_id": "user-session-123",
  "answer": "휴학 신청은 학기 시작 전에...",
  "sources": [
    "data/official/학사안내/1-academic-enrollment.md"
  ]
}
```

### GET /health

서버 상태를 확인합니다.

**응답 예시:**
```json
{
  "status": "healthy",
  "message": "서비스가 정상 작동 중입니다."
}
```