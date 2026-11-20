#!/usr/bin/env bash
# Lay out the Hugging Face Space root in the directory given as $1.
# The Docker build context is this directory, not the git root.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: scripts/stage_hf_space.sh <directory>" >&2
  exit 1
fi

root=$(cd "$(dirname "$0")/.." && pwd)
stage=$1
mkdir -p "$stage/artifacts/transformer" "$stage/data"

cp "$root/deploy/hf_space/Dockerfile" "$stage/Dockerfile"
cp "$root/deploy/hf_space/README.md" "$stage/README.md"
cp "$root/deploy/hf_space/bootstrap.py" "$stage/bootstrap.py"
cp "$root/deploy/hf_space/entrypoint.sh" "$stage/entrypoint.sh"
cp "$root/pyproject.toml" "$root/requirements.txt" "$stage/"
tar -C "$root" -cf - \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  src config scripts data/sample \
  | tar -C "$stage" -xf -
cp "$root/artifacts/transformer/weights.sha256" "$stage/artifacts/transformer/weights.sha256"

if [ -e "$stage/data/private" ] || [ -e "$stage/.env" ]; then
  echo "stage contains private data" >&2
  exit 1
fi
if find "$stage" \( -name '*.pt' -o -name '*.bin' -o -name '.env' \) | grep -q .; then
  echo "stage contains weights or a dotenv file" >&2
  exit 1
fi
echo "$stage"
