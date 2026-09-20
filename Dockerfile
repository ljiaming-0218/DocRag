FROM python:3.11-slim

ARG APP_VERSION=dev
ARG BUILD_COMMIT=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    APP_VERSION=${APP_VERSION} \
    BUILD_COMMIT=${BUILD_COMMIT}

LABEL org.opencontainers.image.title="DocRAG Agent" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.revision="${BUILD_COMMIT}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /home/user/app

RUN mkdir -p \
        /home/user/app/runtime/chroma_bge_m3_api \
        /home/user/app/runtime/uploads \
    && chown -R user:user /home/user/app

COPY medrag/backend/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

USER user

RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBEDDING_PROVIDER=siliconflow \
    EMBEDDING_MODEL=BAAI/bge-m3 \
    EMBEDDING_API_BASE=https://api.siliconflow.cn/v1 \
    EMBEDDING_DIMENSION=1024 \
    EMBEDDING_BATCH_SIZE=32 \
    EMBEDDING_TIMEOUT_SECONDS=60 \
    EMBEDDING_MAX_RETRIES=2 \
    CHROMA_DIR=/home/user/app/runtime/chroma_bge_m3_api \
    UPLOAD_DIR=/home/user/app/runtime/uploads \
    OCR_ENABLED=true \
    OCR_LANGUAGES=eng+chi_sim \
    OCR_DPI=300 \
    OCR_MIN_TEXT_CHARS=20 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

COPY --chown=user:user medrag ./medrag

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "medrag/backend", "--host", "0.0.0.0", "--port", "7860"]
