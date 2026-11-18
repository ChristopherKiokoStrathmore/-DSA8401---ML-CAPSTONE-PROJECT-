FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_BACKEND=tfidf \
    TFIDF_DIR=/app/artifacts/tfidf \
    TRANSFORMER_DIR=/app/artifacts/transformer

WORKDIR /app

# CPU wheel only. The +cpu version is not the default PyPI build.
RUN pip install --upgrade pip \
    && pip install "torch==2.15.0+cpu" \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple

COPY pyproject.toml README.md requirements.txt ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY data/sample ./data/sample
COPY artifacts/transformer/weights.sha256 /app/artifacts/transformer/weights.sha256
COPY docker/entrypoint.sh /entrypoint.sh

RUN pip install . \
    && chmod 755 /entrypoint.sh \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin app \
    && mkdir -p /app/artifacts/tfidf /app/artifacts/transformer \
    && python -m capstone.serving.fixture \
        --labels /app/data/sample/labels \
        --out /app/artifacts/tfidf \
    && chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

ENTRYPOINT ["/entrypoint.sh"]
