#!/bin/sh
# Hugging Face Spaces routes to port 7860. Weights are downloaded here, not in git.
set -eu
cd /app
python /app/bootstrap.py
set -a
# shellcheck disable=SC1091
. /tmp/serve.env
set +a
exec python -m uvicorn capstone.serving.app:app --host 0.0.0.0 --port "${PORT:-7860}"
