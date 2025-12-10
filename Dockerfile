# Hugging Face Spaces Docker Deployment
# https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.11-slim

# HF Spaces 요구사항: 비-root 사용자 생성
RUN useradd -m -u 1000 user

# 시스템 의존성 설치 (root 권한 필요)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 생성 및 권한 설정
RUN mkdir -p /app && chown -R user:user /app

# 사용자 전환
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# uv 설치
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 파일 복사 (사용자 권한)
COPY --chown=user:user pyproject.toml .
COPY --chown=user:user uv.lock .
COPY --chown=user:user app/ ./app/
COPY --chown=user:user data/ ./data/
COPY --chown=user:user chroma_db/ ./chroma_db/

# 의존성 설치
RUN /home/user/.local/bin/uv sync

# HF Spaces 요구사항: 포트 7860
EXPOSE 7860

# 애플리케이션 실행
CMD ["/home/user/.local/bin/uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
