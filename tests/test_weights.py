"""SHA-256 checks for the AfroXLMR download. No network and no real weights."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from capstone.weights import (
    ShaEntry,
    find_folder_member,
    is_google_drive_folder_url,
    is_google_drive_url,
    is_placeholder_url,
    main,
    parse_sha256_file,
    plan_afroxlmr_downloads,
    select_afroxlmr_paths,
)

DRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/1Ql5b645nf9j_UhWygj7kYqH0jaWZsDl9?usp=sharing"
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_google_drive_urls() -> None:
    assert is_google_drive_url("https://drive.google.com/file/d/abc/view?usp=sharing")
    assert is_google_drive_url("https://drive.google.com/uc?id=abc")
    assert is_google_drive_url("https://docs.google.com/uc?export=download&id=abc")
    assert not is_google_drive_url("https://example.com/afroxlmr.pt")
    assert is_google_drive_folder_url(DRIVE_FOLDER)
    assert not is_google_drive_folder_url("https://drive.google.com/file/d/abc/view")
    assert is_placeholder_url("https://drive.google.com/file/d/PLACEHOLDER/view")
    assert is_placeholder_url("TODO_DRIVE_URL")
    assert not is_placeholder_url(DRIVE_FOLDER)
    assert not is_placeholder_url("https://drive.google.com/file/d/abc123/view")


def test_committed_weights_record_the_t4_digest() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config" / "weights.json").read_text(encoding="utf-8"))
    assert config["url"] == DRIVE_FOLDER
    assert config["member"] == "afroxlmr/model.pt"
    assert is_google_drive_folder_url(config["url"])
    assert not is_placeholder_url(config["url"])
    entries = parse_sha256_file(root / config["sha256_file"])
    assert entries == [
        ShaEntry(
            digest="778246a31841ac554fbf574cfc42cc9579981b67fb85a8eb1814dbb689ea9eb6",
            name="model.pt",
        )
    ]
    assert config["sha256"] == entries[0].digest
    assert config["bytes"] == 1112288438
    assert config["filename"] == "model.pt"


def test_placeholder_url_fails_before_download(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sha = tmp_path / "weights.sha256"
    sha.write_text("0" * 64 + "  afroxlmr.pt\n", encoding="utf-8")
    config = tmp_path / "weights.json"
    config.write_text(
        json.dumps(
            {
                "url": "https://drive.google.com/file/d/PLACEHOLDER/view?usp=sharing",
                "sha256_file": str(sha),
                "out_dir": str(tmp_path / "out"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main(["--config", str(config)])
    assert exc.value.code == 1
    assert "placeholder" in capsys.readouterr().err.lower()
    assert not (tmp_path / "out").exists()


def test_matching_download_is_kept(tmp_path: Path) -> None:
    payload = b"afroxlmr-synthetic-bytes"
    source = tmp_path / "source.pt"
    source.write_bytes(payload)
    sha = tmp_path / "weights.sha256"
    sha.write_text(f"# comment\n{_sha(payload)}  afroxlmr.pt\n", encoding="utf-8")
    out = tmp_path / "out"
    main(["--url", source.as_uri(), "--sha256", str(sha), "--out", str(out)])
    assert (out / "afroxlmr.pt").read_bytes() == payload
    assert not (out / ".partial-download").exists()


def test_mismatch_is_loud_and_deletes_the_download(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"not-the-checkpoint")
    sha = tmp_path / "weights.sha256"
    sha.write_text(f"{'ab' * 32}  afroxlmr.pt\n", encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(["--url", source.as_uri(), "--sha256", str(sha), "--out", str(out)])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "mismatch" in err.lower()
    assert "ab" * 32 in err
    assert not (out / "afroxlmr.pt").exists()
    assert not (out / ".partial-download").exists()


def test_zero_digest_mismatch_names_the_placeholder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"real-bytes")
    sha = tmp_path / "weights.sha256"
    sha.write_text("0" * 64 + "  afroxlmr.pt\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--url", source.as_uri(), "--sha256", str(sha), "--out", str(tmp_path / "out")])
    assert "placeholder" in capsys.readouterr().err.lower()


def test_zip_members_are_verified_before_extract(tmp_path: Path) -> None:
    model = b"model-bytes"
    config = b'{"ok":true}'
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("subdir/model.pt", model)
        archive.writestr("config.json", config)
    sha = tmp_path / "weights.sha256"
    sha.write_text(
        f"{_sha(model)}  subdir/model.pt\n{_sha(config)}  config.json\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    main(["--url", archive_path.as_uri(), "--sha256", str(sha), "--out", str(out)])
    assert (out / "subdir" / "model.pt").read_bytes() == model
    assert (out / "config.json").read_bytes() == config


def test_zip_mismatch_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    archive_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("model.pt", b"one")
        archive.writestr("config.json", b"two")
    sha = tmp_path / "weights.sha256"
    sha.write_text(
        f"{_sha(b'one')}  model.pt\n{'cd' * 32}  config.json\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(["--url", archive_path.as_uri(), "--sha256", str(sha), "--out", str(out)])
    assert exc.value.code == 1
    assert "mismatch" in capsys.readouterr().err.lower()
    assert not (out / "model.pt").exists()
    assert not (out / "config.json").exists()


def _stage_folder(output: Path, payload: bytes, nested: str = "afroxlmr/model.pt") -> None:
    output.mkdir(parents=True, exist_ok=True)
    path = output / nested
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _patch_listing(
    monkeypatch: pytest.MonkeyPatch,
    files: list[tuple[str, str]],
    payloads: dict[str, bytes],
) -> list[str]:
    fetched: list[str] = []

    def fake_list(url: str) -> list[tuple[str, str]]:
        assert "folders" in url
        return files

    def fake_download(file_id: str, dest: Path) -> None:
        fetched.append(file_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payloads[file_id])

    monkeypatch.setattr("capstone.weights._list_drive_files", fake_list)
    monkeypatch.setattr("capstone.weights._download_drive_file", fake_download)
    return fetched


def test_folder_member_accepts_an_extra_parent_directory(tmp_path: Path) -> None:
    root = tmp_path / "download"
    _stage_folder(root / "Shared folder", b"weights", "afroxlmr/model.pt")
    found = find_folder_member(root, "afroxlmr/model.pt")
    assert found.read_bytes() == b"weights"


def test_select_afroxlmr_paths_drops_tmp_and_reports() -> None:
    paths = [
        "last.pt.tmp",
        "reports/results.md",
        "reports/figures/issue.png",
        "afroxlmr/model.pt",
        "afroxlmr/tokenizer.json",
        "afroxlmr/tokenizer_config.json",
        "afroxlmr/config.json",
        "Drive folder/afroxlmr/special_tokens_map.json",
    ]
    assert select_afroxlmr_paths(paths) == [
        "afroxlmr/model.pt",
        "afroxlmr/tokenizer.json",
        "afroxlmr/tokenizer_config.json",
        "afroxlmr/config.json",
        "Drive folder/afroxlmr/special_tokens_map.json",
    ]
    model, others = plan_afroxlmr_downloads(
        [("tmp", paths[0]), ("rep", paths[1]), ("model", paths[3]), ("tok", paths[4])],
        "afroxlmr/model.pt",
    )
    assert model == ("model", "model.pt")
    assert others == [("tok", "tokenizer.json")]


def test_folder_listing_downloads_only_afroxlmr_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"checkpoint-bytes"
    files = [
        ("tmp", "last.pt.tmp"),
        ("report", "reports/results.md"),
        ("model", "afroxlmr/model.pt"),
        ("tok", "afroxlmr/tokenizer.json"),
        ("cfg", "afroxlmr/config.json"),
    ]
    fetched = _patch_listing(
        monkeypatch,
        files,
        {"model": payload, "tok": b"tokenizer", "cfg": b"config"},
    )
    sha = tmp_path / "weights.sha256"
    sha.write_text(f"{_sha(payload)}  model.pt\n", encoding="utf-8")
    out = tmp_path / "out"
    main(["--url", DRIVE_FOLDER, "--sha256", str(sha), "--out", str(out)])
    assert fetched == ["model", "tok", "cfg"]
    assert (out / "model.pt").read_bytes() == payload
    assert (out / "tokenizer.json").read_bytes() == b"tokenizer"
    assert (out / "config.json").read_bytes() == b"config"
    assert not (out / "last.pt.tmp").exists()
    assert not (out / "reports").exists()


def test_folder_url_keeps_the_matching_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"checkpoint-bytes"
    _patch_listing(
        monkeypatch,
        [("model", "Shared folder/afroxlmr/model.pt")],
        {"model": payload},
    )
    sha = tmp_path / "weights.sha256"
    sha.write_text(f"{_sha(payload)}  model.pt\n", encoding="utf-8")
    out = tmp_path / "out"
    main(["--url", DRIVE_FOLDER, "--sha256", str(sha), "--out", str(out)])
    assert (out / "model.pt").read_bytes() == payload
    assert not (out / ".partial-download").exists()


def test_matching_checkpoint_skips_the_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"already-there"
    out = tmp_path / "out"
    out.mkdir()
    (out / "model.pt").write_bytes(payload)

    def fail_if_called(url: str) -> list[tuple[str, str]]:
        raise AssertionError(url)

    monkeypatch.setattr("capstone.weights._list_drive_files", fail_if_called)
    sha = tmp_path / "weights.sha256"
    sha.write_text(f"{_sha(payload)}  model.pt\n", encoding="utf-8")
    main(["--url", DRIVE_FOLDER, "--sha256", str(sha), "--out", str(out)])
    assert (out / "model.pt").read_bytes() == payload


def test_folder_url_rejects_a_bad_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_listing(
        monkeypatch,
        [("model", "afroxlmr/model.pt")],
        {"model": b"not-the-checkpoint"},
    )
    sha = tmp_path / "weights.sha256"
    sha.write_text(f"{'ab' * 32}  model.pt\n", encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(["--url", DRIVE_FOLDER, "--sha256", str(sha), "--out", str(out)])
    assert exc.value.code == 1
    assert "mismatch" in capsys.readouterr().err.lower()
    assert not (out / "model.pt").exists()


def test_folder_url_fails_when_the_member_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_listing(
        monkeypatch,
        [("tmp", "last.pt.tmp"), ("report", "reports/results.md")],
        {},
    )
    sha = tmp_path / "weights.sha256"
    sha.write_text(f"{'ab' * 32}  model.pt\n", encoding="utf-8")
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        main(["--url", DRIVE_FOLDER, "--sha256", str(sha), "--out", str(out)])
    assert exc.value.code == 1
    assert "afroxlmr/model.pt" in capsys.readouterr().err
    assert not (out / "model.pt").exists()


def test_unsafe_filename_is_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sha = tmp_path / "weights.sha256"
    sha.write_text(f"{'ab' * 32}  ../escape.pt\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        parse_sha256_file(sha)
    assert exc.value.code == 1
    assert "unsafe" in capsys.readouterr().err.lower()
