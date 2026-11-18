"""The public repo must not track real tabular extracts."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABULAR = (".xlsx", ".xlsm", ".csv", ".parquet")
ARTEFACTS = (".joblib", ".pt", ".bin", ".safetensors", ".pkl")


def test_no_tabular_files_tracked_outside_sample() -> None:
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    bad = []
    for path in tracked:
        if path.startswith("data/sample/"):
            continue
        if path.lower().endswith(TABULAR) or path.lower().endswith(ARTEFACTS):
            bad.append(path)
    assert bad == []


def test_gitignore_blocks_private_data_and_build_artifacts() -> None:
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for token in (
        "data/private/",
        "artifacts/",
        ".env*",
        "__pycache__/",
        "node_modules/",
        ".next/",
        "artifacts/tfidf/",
        "*.joblib",
        "*.pt",
        "!artifacts/transformer/weights.sha256",
    ):
        assert token in text


def test_checked_in_sample_matches_generator(tmp_path: Path) -> None:
    """Smoke writes this file. It has to stay byte-identical to seed 42."""
    from capstone.generate_sample import generate_rows, write_sample

    sample = ROOT / "data" / "sample" / "messages.csv"
    assert sample.is_file(), "data/sample/messages.csv is missing; run make smoke"
    expected = tmp_path / "expected.csv"
    write_sample(expected, generate_rows(n=400, seed=42))
    assert sample.read_bytes() == expected.read_bytes()
