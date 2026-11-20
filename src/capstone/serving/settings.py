"""Process settings for the serving app. Read once, when the app is created."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "https://multi-head.vercel.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)
# Preview deployments on Vercel, plus the Next.js dev server on any local port.
DEFAULT_CORS_ORIGIN_REGEX = r"(?:https://.*\.vercel\.app|http://(?:localhost|127\.0\.0\.1):\d+)\Z"


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


@dataclass(frozen=True)
class Settings:
    """Runtime configuration. Environment variables override the defaults."""

    backend: str = "tfidf"
    tfidf_dir: Path = Path("artifacts/tfidf")
    transformer_dir: Path = Path("artifacts/afroxlmr")
    max_text_chars: int = 4000
    max_batch_size: int = 32
    cors_origins: tuple[str, ...] = DEFAULT_CORS_ORIGINS
    cors_origin_regex: str | None = DEFAULT_CORS_ORIGIN_REGEX

    @classmethod
    def from_env(cls) -> Settings:
        backend = os.environ.get("MODEL_BACKEND", "tfidf").strip().lower() or "tfidf"
        origins_raw = os.environ.get("CORS_ORIGINS")
        if origins_raw is None or origins_raw.strip() == "":
            origins = ("*",)
        else:
            origins = tuple(part.strip() for part in origins_raw.split(",") if part.strip())
        if "CORS_ORIGIN_REGEX" in os.environ:
            regex_raw = os.environ["CORS_ORIGIN_REGEX"].strip()
            regex: str | None = regex_raw or None
        elif origins == ("*",):
            regex = None
        else:
            regex = DEFAULT_CORS_ORIGIN_REGEX
        return cls(
            backend=backend,
            tfidf_dir=Path(os.environ.get("TFIDF_DIR", "artifacts/tfidf")),
            transformer_dir=Path(os.environ.get("TRANSFORMER_DIR", "artifacts/afroxlmr")),
            max_text_chars=_positive_int("MAX_TEXT_CHARS", 4000),
            max_batch_size=_positive_int("MAX_BATCH_SIZE", 32),
            cors_origins=origins,
            cors_origin_regex=regex,
        )

    def checkpoint_dir(self) -> Path:
        if self.backend == "transformer":
            return self.transformer_dir
        return self.tfidf_dir
