"""Download AfroXLMR inference artefacts and check them against weights.sha256.

The URL comes from ``AFROXLMR_WEIGHTS_URL`` or ``WEIGHTS_URL``, then from
``config/weights.json``. A Google Drive file link is fetched with gdown.
Anything else is fetched with urllib, including ``file://`` for tests.

``weights.sha256`` is the usual sha256sum layout: ``<digest>  <filename>``,
with ``#`` comments ignored. One entry means the downloaded blob must match
and is saved under that name. Several entries mean the download is a zip and
each member is hashed before it is written. A mismatch deletes the partial
download and exits 1.

The committed digest is the Colab T4 ``model.pt`` (1,112,288,438 bytes). The
committed URL is ``TODO_DRIVE_URL`` until the owner pastes a Drive link. The
script refuses that URL before any download. A file that does not match the
digest is deleted and the script exits 1.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

_PLACEHOLDER_MARKERS = ("placeholder", "replace_me", "todo_drive_url")


@dataclass(frozen=True)
class ShaEntry:
    digest: str
    name: str


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def find_root() -> Path:
    """Repo root in a checkout, or the process cwd when the package is installed."""
    cwd = Path.cwd()
    if (cwd / "config" / "weights.json").is_file():
        return cwd
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "config" / "weights.json").is_file():
            return parent
    return cwd


def is_google_drive_url(url: str) -> bool:
    host = urllib.parse.urlsplit(url).netloc.lower()
    if host in {"drive.google.com", "docs.google.com"}:
        return True
    return host.endswith(".drive.google.com")


def is_placeholder_url(url: str) -> bool:
    lowered = url.strip().lower()
    if lowered == "":
        return True
    return any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def display_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def parse_sha256_file(path: Path) -> list[ShaEntry]:
    if not path.is_file():
        fail(f"SHA-256 file not found: {path}")
    entries: list[ShaEntry] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            fail(f"{path}:{line_number}: expected '<sha256>  <filename>'")
        digest, name = parts
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            fail(f"{path}:{line_number}: digest is not 64 hex characters")
        if name.startswith("/") or ".." in Path(name).parts:
            fail(f"{path}:{line_number}: refusing unsafe filename {name!r}")
        entries.append(ShaEntry(digest=digest.lower(), name=name))
    if not entries:
        fail(f"{path} has no digests")
    return entries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_zip_member(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"could not parse {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"{path} must be a JSON object")
    return {str(key): str(value) for key, value in payload.items()}


def resolve_settings(
    *,
    url: str | None,
    sha256_path: Path | None,
    out_dir: Path | None,
    config_path: Path | None,
    environ: dict[str, str],
) -> tuple[str, Path, Path]:
    config = load_config(config_path or (find_root() / "config" / "weights.json"))
    chosen_url = (
        url
        or environ.get("AFROXLMR_WEIGHTS_URL")
        or environ.get("WEIGHTS_URL")
        or config.get("url")
        or ""
    ).strip()
    sha_raw = (
        sha256_path
        or _env_path(environ, "AFROXLMR_WEIGHTS_SHA256")
        or _maybe_path(config.get("sha256_file"))
        or (find_root() / "artifacts" / "transformer" / "weights.sha256")
    )
    out_raw = (
        out_dir
        or _env_path(environ, "AFROXLMR_WEIGHTS_DIR")
        or _maybe_path(config.get("out_dir"))
        or (find_root() / "artifacts" / "transformer")
    )
    if chosen_url == "":
        fail("No AfroXLMR weights URL. Set AFROXLMR_WEIGHTS_URL or edit config/weights.json.")
    return chosen_url, _resolve(sha_raw), _resolve(out_raw)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if is_google_drive_url(url):
        try:
            import gdown
        except ImportError:
            fail(
                "gdown is required to download a Google Drive link. Install the package and retry."
            )
        try:
            result = gdown.download(url, output=str(dest), quiet=False, fuzzy=True)
        except Exception as exc:
            dest.unlink(missing_ok=True)
            fail(f"gdown failed for {display_url(url)}: {type(exc).__name__}: {exc}")
        if not result or not dest.is_file() or dest.stat().st_size == 0:
            dest.unlink(missing_ok=True)
            fail(f"gdown did not write a file for {display_url(url)}")
        return
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        fail(f"download failed for {display_url(url)}: {type(exc).__name__}: {exc}")
    if not dest.is_file() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        fail(f"download failed for {display_url(url)}: empty file")


def install_download(downloaded: Path, entries: list[ShaEntry], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(entries) == 1:
        _check_digest(sha256_file(downloaded), entries[0])
        target = _safe_target(out_dir, entries[0].name)
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded.replace(target)
        print(f"verified {entries[0].name}")
        return
    if not zipfile.is_zipfile(downloaded):
        downloaded.unlink(missing_ok=True)
        fail(
            "weights.sha256 lists more than one file but the download is not a zip. "
            "Nothing was written."
        )
    with zipfile.ZipFile(downloaded) as archive:
        names = set(archive.namelist())
        verified: list[ShaEntry] = []
        for entry in entries:
            if entry.name not in names:
                downloaded.unlink(missing_ok=True)
                fail(f"archive is missing {entry.name}. Nothing was written.")
            _check_digest(sha256_zip_member(archive, entry.name), entry)
            verified.append(entry)
        for entry in verified:
            target = _safe_target(out_dir, entry.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry.name) as source, target.open("wb") as sink:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    sink.write(chunk)
            print(f"verified {entry.name}")
    downloaded.unlink(missing_ok=True)


def run(
    *,
    url: str | None = None,
    sha256_path: Path | None = None,
    out_dir: Path | None = None,
    config_path: Path | None = None,
    environ: dict[str, str] | None = None,
) -> None:
    chosen_url, sha_path, destination = resolve_settings(
        url=url,
        sha256_path=sha256_path,
        out_dir=out_dir,
        config_path=config_path,
        environ=dict(os.environ) if environ is None else environ,
    )
    if is_placeholder_url(chosen_url):
        fail(
            "AfroXLMR weights URL is still the placeholder. Set AFROXLMR_WEIGHTS_URL "
            "to the public file link (Google Drive is supported) or edit config/weights.json. "
            "Refusing to download."
        )
    entries = parse_sha256_file(sha_path)
    destination.mkdir(parents=True, exist_ok=True)
    partial = destination / ".partial-download"
    try:
        download(chosen_url, partial)
        install_download(partial, entries, destination)
    except SystemExit:
        partial.unlink(missing_ok=True)
        raise
    finally:
        partial.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download AfroXLMR artefacts and verify SHA-256.")
    parser.add_argument("--url", default=None, help="Overrides AFROXLMR_WEIGHTS_URL and config.")
    parser.add_argument("--sha256", type=Path, default=None, help="Digest file to check against.")
    parser.add_argument("--out", type=Path, default=None, help="Directory to write verified files.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON config. Default config/weights.json.",
    )
    args = parser.parse_args(argv)
    run(url=args.url, sha256_path=args.sha256, out_dir=args.out, config_path=args.config)


def _check_digest(actual: str, entry: ShaEntry) -> None:
    if hmac.compare_digest(actual, entry.digest):
        return
    hint = ""
    if set(entry.digest) == {"0"}:
        hint = (
            " The committed digest is still the placeholder. "
            "Replace it in weights.sha256 with the real SHA-256."
        )
    fail(
        f"SHA-256 mismatch for {entry.name}: expected {entry.digest} got {actual}.{hint} "
        "The download was deleted."
    )


def _safe_target(out_dir: Path, name: str) -> Path:
    root = out_dir.resolve()
    target = (out_dir / name).resolve()
    if not target.is_relative_to(root):
        fail(f"refusing to write outside {root}: {name}")
    return target


def _env_path(environ: dict[str, str], name: str) -> Path | None:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return Path(raw)


def _maybe_path(raw: str | None) -> Path | None:
    if raw is None or raw.strip() == "":
        return None
    return Path(raw)


def _resolve(path: Path) -> Path:
    if path.is_absolute():
        return path
    return find_root() / path


if __name__ == "__main__":
    main()
