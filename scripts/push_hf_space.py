#!/usr/bin/env python3
"""Create the public Docker Space if needed, then upload the staged tree.

A fresh clone needs only ``pip install huggingface_hub``. ``HF_TOKEN`` is read
from the environment and is not printed. ``HF_SPACE`` defaults to
``Kioko1/airtel-multihead``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_SPACE = "Kioko1/airtel-multihead"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("HF_TOKEN is not set", file=sys.stderr)
        raise SystemExit(1)
    repo_id = os.environ.get("HF_SPACE", DEFAULT_SPACE).strip() or DEFAULT_SPACE
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        raise SystemExit(1) from None

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        space_hardware="cpu-basic",
        private=False,
        exist_ok=True,
    )
    root = Path(__file__).resolve().parents[1]
    stage_script = root / "scripts" / "stage_hf_space.sh"
    with tempfile.TemporaryDirectory(prefix="hf-space-") as tmp:
        stage = Path(tmp) / "space"
        stage.mkdir()
        subprocess.run([str(stage_script), str(stage)], check=True, cwd=root)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(stage),
            repo_type="space",
            commit_message="Publish the Docker Space API",
        )
    print(f"uploaded {repo_id}")


if __name__ == "__main__":
    main()
