#!/bin/bash
# 개발 서버 실행 스크립트

cd "$(dirname "$0")"

echo "이화여대 신입생 챗봇 백엔드 개발 서버 시작..."
echo "서버 주소: http://localhost:8000"
echo "API 문서: http://localhost:8000/docs"
echo ""

uv run uvicorn app.main:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000 \
  --reload-dir app \
  --reload-dir scripts \
  --reload-exclude '.venv/*' \
  --reload-exclude '*.pyc' \
  --reload-exclude '__pycache__/*' \
  --reload-exclude 'chroma_db/*'
