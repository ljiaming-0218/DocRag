FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-chi-sim \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /home/user/app

RUN mkdir -p \
        /home/user/app/runtime/chroma_db \
        /home/user/app/runtime/uploads \
    && chown -R user:user /home/user/app

COPY medrag/backend/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

USER user

RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('BAAI/bge-small-zh-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    CHROMA_DIR=/home/user/app/runtime/chroma_db \
    UPLOAD_DIR=/home/user/app/runtime/uploads \
    OCR_ENABLED=true \
    OCR_LANGUAGES=eng+chi_sim \
    OCR_DPI=300 \
    OCR_MIN_TEXT_CHARS=20 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata

COPY --chown=user:user medrag ./medrag

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "medrag/backend", "--host", "0.0.0.0", "--port", "7860"]
