FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_BACKEND=transformer \
    PORT=7860 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU torch stays under the 16 GB Space. The default PyPI wheel is the CUDA build.
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --upgrade pip \
    && grep -v '^torch==' requirements.txt > /tmp/requirements.notorch.txt \
    && pip install --no-cache-dir "torch==2.5.1+cpu" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /tmp/requirements.notorch.txt \
    && pip install --no-cache-dir --force-reinstall "torch==2.5.1+cpu" \
        --index-url https://download.pytorch.org/whl/cpu

COPY src ./src
COPY config ./config
COPY scripts ./scripts
COPY data/sample ./data/sample
COPY artifacts/transformer/weights.sha256 ./artifacts/transformer/weights.sha256
COPY bootstrap.py entrypoint.sh ./

RUN pip install --no-cache-dir --no-deps . \
    && chmod 755 /app/entrypoint.sh \
    && useradd --create-home --uid 1000 --shell /bin/sh app \
    && chown -R app:app /app

USER app

EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
