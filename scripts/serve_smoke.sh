#!/usr/bin/env bash
# Train the synthetic TF-IDF fixture, start the API, and curl /health and /predict.
# Predictions from this fixture are a harness check, not a model-quality result.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${BIN:-.venv/bin}"
PORT="${SERVE_PORT:-8765}"
OUT="${TFIDF_DIR:-$ROOT/artifacts/tfidf}"
LOG="$ROOT/artifacts/serve-smoke.log"

if [[ ! -x "$BIN/python" ]]; then
  echo "missing $BIN/python; run make install first" >&2
  exit 1
fi

mkdir -p "$ROOT/artifacts"
# Prefer the Phase 3 pipelines (mask + word/char TF-IDF + logistic regression).
# The fixture is only the fallback when those files have not been trained yet.
if [[ ! -f "$OUT/issue.joblib" || ! -f "$OUT/sentiment.joblib" || ! -f "$OUT/urgency.joblib" || ! -f "$OUT/labels.json" ]]; then
  echo "Phase 3 TF-IDF artefacts missing; fitting them on the sample" >&2
  MANIFESTS="$ROOT/reports/manifests"
  if [[ ! -f "$MANIFESTS/issue.json" ]]; then
    "$BIN/python" -m capstone.splits --labels "$ROOT/data/sample/labels" \
      --out "$MANIFESTS" --kind harness
  fi
  "$BIN/python" -m capstone.models.baselines \
    --labels "$ROOT/data/sample/labels" --manifests "$MANIFESTS" \
    --out "$OUT" --kind harness --seed 42
else
  echo "using Phase 3 TF-IDF artefacts in $OUT" >&2
fi

MODEL_BACKEND=tfidf TFIDF_DIR="$OUT" \
  "$BIN/python" -m uvicorn capstone.serving.app:app --host 127.0.0.1 --port "$PORT" \
  >"$LOG" 2>&1 &
PID=$!

cleanup() {
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 50); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "server exited before /health succeeded" >&2
    cat "$LOG" >&2
    exit 1
  fi
  sleep 0.2
done

if [[ "$ready" -ne 1 ]]; then
  echo "timed out waiting for /health" >&2
  cat "$LOG" >&2
  exit 1
fi

echo "=== GET /health ==="
curl -sf "http://127.0.0.1:${PORT}/health"
echo
echo "=== POST /predict ==="
curl -sf -X POST "http://127.0.0.1:${PORT}/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"My line has been down since morning, please help"}'
echo
