"""Choose a weights directory, run the download script, or fall back to the fixture.

Writes ``/tmp/serve.env`` for the entrypoint. A failed download is a harness
check: the synthetic TF-IDF fixture serves ``/predict`` and the log says so.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def cache_dir() -> Path:
    data = Path("/data")
    if data.is_dir() and os.access(data, os.W_OK):
        return data / "afroxlmr"
    return Path("/tmp/afroxlmr")


def write_env(backend: str, tfidf: Path, transformer: Path) -> None:
    lines = [
        f"MODEL_BACKEND={backend}",
        f"TFIDF_DIR={tfidf}",
        f"TRANSFORMER_DIR={transformer}",
    ]
    Path("/tmp/serve.env").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sidecars(dest: Path) -> None:
    """config.json and the tokenizer are not inside model.pt. Add them if absent."""
    if (dest / "config.json").is_file() and (dest / "tokenizer_config.json").is_file():
        return
    from transformers import AutoConfig, AutoTokenizer

    from capstone.labels import ISSUE_CLASSES, SENTIMENT_CLASSES, URGENCY_CLASSES

    model_id = "Davlan/afro-xlmr-base"
    if not (dest / "config.json").is_file():
        encoder = AutoConfig.from_pretrained(model_id)
        payload = {
            "encoder": encoder.to_dict(),
            "freeze_bottom_layers": 8,
            "freeze_embeddings": True,
            "hidden_size": int(encoder.hidden_size),
            "issue_labels": list(ISSUE_CLASSES),
            "max_length": 128,
            "model_name": model_id,
            "num_hidden_layers": int(encoder.num_hidden_layers),
            "seed": 42,
            "sentiment_labels": list(SENTIMENT_CLASSES),
            "tiny": False,
            "tokenizer_type": "huggingface",
            "urgency_labels": list(URGENCY_CLASSES),
        }
        import json

        (dest / "config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if not (dest / "tokenizer_config.json").is_file():
        AutoTokenizer.from_pretrained(model_id).save_pretrained(dest)
    threshold = dest / "threshold.json"
    if not threshold.is_file():
        threshold.write_text('{"threshold": 0.5}\n', encoding="utf-8")


def harness(transformer: Path) -> None:
    tfidf = Path("/tmp/tfidf-harness")
    print(
        "HARNESS CHECK weight download failed; serving the synthetic TF-IDF fixture",
        file=sys.stderr,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "capstone.serving.fixture",
            "--labels",
            "/app/data/sample/labels",
            "--out",
            str(tfidf),
        ],
        check=True,
    )
    write_env("tfidf", tfidf, transformer)


def main() -> None:
    transformer = cache_dir()
    transformer.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [sys.executable, "/app/scripts/download_weights.py", "--out", str(transformer)],
        check=False,
    )
    if completed.returncode != 0 or not (transformer / "model.pt").is_file():
        harness(transformer)
        return
    sidecars(transformer)
    write_env("transformer", Path("/tmp/tfidf-harness"), transformer)
    print(f"weights ready at {transformer}", file=sys.stderr)


if __name__ == "__main__":
    main()
