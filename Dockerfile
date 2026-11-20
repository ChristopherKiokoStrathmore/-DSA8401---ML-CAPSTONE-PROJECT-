FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_BACKEND=tfidf \
    TFIDF_DIR=/app/artifacts/tfidf \
    WEIGHTS_DIR=/app/artifacts/transformer \
    TRANSFORMER_DIR=/app/artifacts/transformer \
    PORT=8000 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU wheel first. The default PyPI torch==2.5.1 build is the CUDA wheel.
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --upgrade pip \
    && grep -v '^torch==' requirements.txt > /tmp/requirements.notorch.txt \
    && pip install --no-cache-dir "torch==2.5.1+cpu" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.notorch.txt

COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY data/sample ./data/sample
COPY artifacts/transformer/weights.sha256 /app/artifacts/transformer/weights.sha256
COPY docker/entrypoint.sh /entrypoint.sh

# --no-deps keeps the CPU torch installed above. pyproject allows torch>=2.5.1.
RUN pip install --no-cache-dir --no-deps . \
    && chmod 755 /entrypoint.sh \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /app/artifacts/tfidf /app/artifacts/transformer /data \
    && python -m capstone.serving.fixture \
        --labels /app/data/sample/labels \
        --out /app/artifacts/tfidf \
    && chown -R app:app /app /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD python -c "import os,urllib.request; port=os.environ.get('PORT','8000'); urllib.request.urlopen('http://127.0.0.1:%s/health' % port, timeout=8)"

ENTRYPOINT ["/entrypoint.sh"]
