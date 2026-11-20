"""Download AfroXLMR inference artefacts and check them against weights.sha256.

The URL comes from ``AFROXLMR_WEIGHTS_URL`` or ``WEIGHTS_URL``, then from
``config/weights.json``. A Google Drive file link is fetched with gdown.
A Google Drive folder is listed first. Only files inside ``afroxlmr/`` are
fetched (``model.pt`` plus the tokenizer files). ``last.pt.tmp``, ``reports/``,
and anything else in the folder are not downloaded. ``model.pt`` is then
hashed. If that file is already present and the digest matches, the download
is skipped.
Anything else is fetched with urllib, including ``file://`` for tests.

``weights.sha256`` is the usual sha256sum layout: ``<digest>  <filename>``,
with ``#`` comments ignored. One entry means the downloaded blob must match
and is saved under that name. Several entries mean the download is a zip and
each member is hashed before it is written. A folder URL requires exactly
one entry. A mismatch deletes the partial download and exits 1.

The committed digest is the Colab T4 ``model.pt`` (1,112,288,438 bytes). A
second T4 training reproduced it. The committed URL is the public Drive
folder that contains ``afroxlmr/model.pt``. A placeholder URL is refused
before any download.
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
from pathlib import Path, PurePosixPath

_PLACEHOLDER_MARKERS = ("placeholder", "replace_me", "todo_drive_url")
_DEFAULT_MEMBER = "afroxlmr/model.pt"


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


def is_google_drive_folder_url(url: str) -> bool:
    if not is_google_drive_url(url):
        return False
    return "/folders/" in urllib.parse.urlsplit(url).path


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
) -> tuple[str, Path, Path, str]:
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
        or _env_path(environ, "WEIGHTS_DIR")
        or _env_path(environ, "AFROXLMR_WEIGHTS_DIR")
        or _maybe_path(config.get("out_dir"))
        or (find_root() / "artifacts" / "transformer")
    )
    member = (
        environ.get("AFROXLMR_WEIGHTS_MEMBER") or config.get("member") or _DEFAULT_MEMBER
    ).strip()
    if chosen_url == "":
        fail("No AfroXLMR weights URL. Set AFROXLMR_WEIGHTS_URL or edit config/weights.json.")
    return chosen_url, _resolve(sha_raw), _resolve(out_raw), member


def afroxlmr_relative(path: str) -> str | None:
    """Path inside an ``afroxlmr`` directory, or None when the file is elsewhere."""
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if "afroxlmr" not in parts:
        return None
    rest = parts[parts.index("afroxlmr") + 1 :]
    if not rest or any(part in {"", ".", ".."} for part in rest):
        return None
    return "/".join(rest)


def select_afroxlmr_paths(paths: list[str]) -> list[str]:
    """Keep Drive paths that live under ``afroxlmr/`` and drop the rest."""
    return [path for path in paths if afroxlmr_relative(path) is not None]


def plan_afroxlmr_downloads(
    files: list[tuple[str, str]],
    member: str,
) -> tuple[tuple[str, str], list[tuple[str, str]]]:
    """Split a Drive listing into the checkpoint and the other afroxlmr files.

    ``files`` is ``(file_id, path)``. Paths outside ``afroxlmr/`` are ignored,
    including a sibling ``last.pt.tmp`` and a ``reports/`` directory.
    """
    chosen: list[tuple[str, str]] = []
    for file_id, path in files:
        relative = afroxlmr_relative(path)
        if relative is None:
            continue
        chosen.append((file_id, relative))
    want = afroxlmr_relative(member) or PurePosixPath(member).name
    models = [item for item in chosen if item[1] == want]
    if len(models) != 1:
        fail(
            f"Drive folder listing has {len(models)} files for {member}. "
            "Expected one afroxlmr checkpoint. Nothing was written."
        )
    others = [item for item in chosen if item[1] != want]
    return models[0], others


def find_folder_member(root: Path, member: str) -> Path:
    """Return the one file under ``root`` whose path ends with ``member``."""
    relative = Path(member)
    if relative.is_absolute() or ".." in relative.parts or relative.parts == ():
        fail(f"refusing unsafe Drive member {member!r}")
    matches = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.parts[-len(relative.parts) :] == relative.parts
    ]
    if not matches:
        fail(
            f"Drive folder has no {member}. "
            f"Expected {_DEFAULT_MEMBER} inside the folder. Nothing was written."
        )
    if len(matches) > 1:
        listed = ", ".join(str(path.relative_to(root)) for path in matches)
        fail(f"Drive folder has more than one {member}: {listed}. Nothing was written.")
    return matches[0]


def download(url: str, dest: Path, *, member: str = _DEFAULT_MEMBER) -> None:
    """Download one file. Folder URLs are handled in :func:`run`."""
    del member
    dest.parent.mkdir(parents=True, exist_ok=True)
    if is_google_drive_folder_url(url):
        fail("internal error: folder URLs are downloaded by run(), not download()")
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
    chosen_url, sha_path, destination, member = resolve_settings(
        url=url,
        sha256_path=sha256_path,
        out_dir=out_dir,
        config_path=config_path,
        environ=dict(os.environ) if environ is None else environ,
    )
    if is_placeholder_url(chosen_url):
        fail(
            "AfroXLMR weights URL is still the placeholder. Set AFROXLMR_WEIGHTS_URL "
            "to the public Drive folder or file link, or edit config/weights.json. "
            "Refusing to download."
        )
    entries = parse_sha256_file(sha_path)
    if len(entries) == 1 and _sha_matches(destination / entries[0].name, entries[0]):
        print(f"{entries[0].name} already matches the SHA-256; skipping download")
        return
    if is_google_drive_folder_url(chosen_url) and len(entries) != 1:
        fail(
            "A Drive folder download keeps one member. "
            "weights.sha256 must list exactly that file. Nothing was written."
        )
    destination.mkdir(parents=True, exist_ok=True)
    partial = destination / ".partial-download"
    try:
        siblings: list[tuple[str, str]] = []
        if is_google_drive_folder_url(chosen_url):
            model, siblings = plan_afroxlmr_downloads(_list_drive_files(chosen_url), member)
            _download_drive_file(model[0], partial)
        else:
            download(chosen_url, partial, member=member)
        install_download(partial, entries, destination)
        for file_id, relative in siblings:
            _download_drive_file(file_id, destination / relative)
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


def _list_drive_files(url: str) -> list[tuple[str, str]]:
    """List a Drive folder. This does not download file bytes."""
    try:
        import gdown
    except ImportError:
        fail("gdown is required to download a Google Drive folder. Install the package and retry.")
    try:
        listed = gdown.download_folder(url=url, skip_download=True, quiet=True)
    except Exception as exc:
        fail(f"gdown folder listing failed for {display_url(url)}: {type(exc).__name__}: {exc}")
    if not listed:
        fail(f"gdown listed no files in {display_url(url)}")
    return [(str(item.id), str(item.path).replace("\\", "/")) for item in listed]


def _download_drive_file(file_id: str, dest: Path) -> None:
    try:
        import gdown
    except ImportError:
        fail("gdown is required to download a Google Drive file. Install the package and retry.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = gdown.download(
            url="https://drive.google.com/uc?id=" + file_id,
            output=str(dest),
            quiet=True,
            fuzzy=True,
        )
    except Exception as exc:
        dest.unlink(missing_ok=True)
        fail(f"gdown failed for file {file_id}: {type(exc).__name__}: {exc}")
    if not result or not dest.is_file() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        fail(f"gdown did not write {dest.name}")


def _sha_matches(path: Path, entry: ShaEntry) -> bool:
    if not path.is_file():
        return False
    return hmac.compare_digest(sha256_file(path), entry.digest)


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
