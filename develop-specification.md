# Backend / AI Develop Specification

이 문서는 `ewha-fresh-db` 저장소 안 `backend/` 폴더에서 구현할 신입생용 학사 챗봇 백엔드/AI 코드의 설계 스펙 문서임. 동일한 저장소의 루트 경로 내의 `data/` 안의 구축된 데이터 파일을 벡터 DB에 쌓은 뒤 RAG 모델에 사용할 예정임.

## 1. 목표

- 이화여대 신입생을 위한 학사 안내 챗봇 AI/백엔드 구현
- 프론트엔드(Next.js, Vercel 배포)에서 **HTTP POST**로 호출
    - 브라우저 세션 아이디 + /ask 엔드포인트를 통한 기본 챗봇 답변
- 주요 기능
  - 학사 정보(MD 문서 기반) RAG 질의응답
  - 사용자의 학년/전공 조건을 반영한 챗봇 답변 (추후)

## 2. 기술 스택

- 언어: **Python 3.11+**
- 웹 프레임워크: **FastAPI**
- WSGI/ASGI 서버: **Uvicorn**
- 벡터 스토어: **ChromaDB** (로컬 디렉터리)
- RAG 유틸: **LangChain**
  - `langchain-community` (Chroma 래퍼 등)
  - `langchain-google-genai` (Gemini / text-embedding-004)
- LLM: **Gemini 2.5 Flash** (기본)
- 데이터 파일
  - `data/official/` : 공식 학사 안내 문서 모음 (등록, 휴학, 자퇴, 수강, 졸업 등)
- 백엔드: **Docker**
- python 환경: uv