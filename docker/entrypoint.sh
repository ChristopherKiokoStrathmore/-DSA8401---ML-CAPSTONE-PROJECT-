#!/bin/sh
# Runs as the non-root app user. If the artefacts volume is empty, train the
# synthetic TF-IDF fixture so `docker compose up` can answer /predict.
set -eu

cd /app
tfidf_dir="${TFIDF_DIR:-/app/artifacts/tfidf}"

if [ "${MODEL_BACKEND:-tfidf}" = "tfidf" ] && [ ! -f "${tfidf_dir}/labels.json" ]; then
  mkdir -p "${tfidf_dir}" 2>/dev/null || true
  if python -m capstone.serving.fixture --labels /app/data/sample/labels --out "${tfidf_dir}"; then
    echo "trained tfidf fixture into ${tfidf_dir}" >&2
  else
    echo "tfidf artefacts are missing and the fixture could not be trained" >&2
  fi
fi

exec python -m uvicorn capstone.serving.app:app --host 0.0.0.0 --port 8000
