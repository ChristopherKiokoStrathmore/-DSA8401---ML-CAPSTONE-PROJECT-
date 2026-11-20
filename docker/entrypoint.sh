#!/bin/sh
# Bind 0.0.0.0:$PORT (8000 in Compose, Railway sets PORT). Transformer mode
# downloads only afroxlmr/ into WEIGHTS_DIR when model.pt is missing or the
# SHA-256 does not match. A matching file skips the download.
set -eu

cd /app
port="${PORT:-8000}"
weights_dir="${WEIGHTS_DIR:-/app/artifacts/transformer}"
tfidf_dir="${TFIDF_DIR:-/app/artifacts/tfidf}"
export PORT="$port"
export WEIGHTS_DIR="$weights_dir"
export TFIDF_DIR="$tfidf_dir"

if [ "$(id -u)" = "0" ]; then
  mkdir -p "$weights_dir" "$tfidf_dir"
  chown -R app:app "$weights_dir" "$tfidf_dir"
  exec su -p -s /bin/sh app -c "exec /entrypoint.sh"
fi

if [ "${MODEL_BACKEND:-tfidf}" = "transformer" ]; then
  export TRANSFORMER_DIR="$weights_dir"
  python /app/scripts/download_weights.py --out "$weights_dir"
else
  export TRANSFORMER_DIR="${TRANSFORMER_DIR:-$weights_dir}"
  if [ ! -f "${tfidf_dir}/labels.json" ]; then
    mkdir -p "$tfidf_dir" 2>/dev/null || true
    if python -m capstone.serving.fixture --labels /app/data/sample/labels --out "$tfidf_dir"; then
      echo "HARNESS CHECK trained tfidf fixture into ${tfidf_dir}" >&2
    else
      echo "tfidf artefacts are missing and the fixture could not be trained" >&2
    fi
  fi
fi

exec python -m uvicorn capstone.serving.app:app --host 0.0.0.0 --port "$port"
