"""HTTP contract for the serving app. Fixture predictions are a harness check."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from capstone.labels import ISSUE_CLASSES, SENTIMENT_CLASSES, URGENCY_CLASSES
from capstone.pii import mask_pii
from capstone.serving.app import create_app
from capstone.serving.app import mask_pii as serving_mask
from capstone.serving.backends import ModelNotReady, _read_labels, load_tfidf, to_public
from capstone.serving.fixture import train_fixture
from capstone.serving.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "sample" / "labels"
PHONE = "0712345678"
EMAIL = "fake.user@example.com"


class Recording:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.seen: list[str] = []

    def predict(self, texts: list[str]) -> list[dict]:
        self.seen.extend(texts)
        return self.inner.predict(texts)


class Exploding:
    def predict(self, texts: list[str]) -> list[dict]:
        raise RuntimeError(f"secret {PHONE} {texts[0]}")


class Garbage:
    def predict(self, texts: list[str]) -> list[dict]:
        return [{"nope": texts[0]} for _ in texts]


@pytest.fixture
def tfidf_dir(tmp_path: Path) -> Path:
    return train_fixture(LABELS, tmp_path / "tfidf")


@pytest.fixture
def settings(tfidf_dir: Path) -> Settings:
    return Settings(
        backend="tfidf",
        tfidf_dir=tfidf_dir,
        max_text_chars=80,
        max_batch_size=3,
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings))


def test_mask_pii_is_the_training_function() -> None:
    assert serving_mask is mask_pii


def test_transformer_module_is_not_imported_for_tfidf(tfidf_dir: Path) -> None:
    # Other tests may already have imported the trainer. Drop it and prove the
    # TF-IDF load path does not import it again.
    sys.modules.pop("capstone.models.multitask", None)
    load_tfidf(tfidf_dir)
    assert "capstone.models.multitask" not in sys.modules


def test_health_reports_loaded_tfidf(client: TestClient, tfidf_dir: Path) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "status",
        "model_loaded",
        "backend",
        "device",
        "checkpoint",
        "issue_classes",
        "sent_classes",
        "urgency_labels",
        "error",
    }
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["backend"] == "tfidf"
    assert body["device"] == "cpu"
    assert body["checkpoint"] == str(tfidf_dir)
    assert body["issue_classes"] == 10
    assert body["sent_classes"] == 2
    assert body["urgency_labels"] == ["low", "medium", "emergency"]
    assert body["error"] is None


def test_predict_matches_frontend_schema(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"text": "My line has been down since morning, please help"},
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"issue", "sentiment", "urgency", "urgency_score", "confidence"}
    assert body["issue"] in ISSUE_CLASSES
    assert body["sentiment"] in SENTIMENT_CLASSES
    assert body["urgency"] in URGENCY_CLASSES
    assert isinstance(body["urgency_score"], float)
    assert 0.0 <= body["urgency_score"] <= 1.0
    assert set(body["confidence"]) == {"issue", "sentiment"}
    assert 0.0 <= body["confidence"]["issue"] <= 1.0
    assert 0.0 <= body["confidence"]["sentiment"] <= 1.0
    assert "My line" not in response.text


def test_batch_paths_return_an_aligned_array(client: TestClient) -> None:
    payload = {"texts": ["bundle haijaingia", "asante", "reverse this transfer"]}
    slash = client.post("/predict/batch", json=payload)
    underscore = client.post("/predict_batch", json=payload)
    assert slash.status_code == 200
    assert underscore.status_code == 200
    assert isinstance(slash.json(), list)
    assert len(slash.json()) == 3
    assert len(underscore.json()) == 3
    for item in slash.json():
        assert item["issue"] in ISSUE_CLASSES
        assert item["sentiment"] in SENTIMENT_CLASSES
        assert item["urgency"] in URGENCY_CLASSES


def test_text_and_batch_limits_are_422_and_do_not_echo_the_message(client: TestClient) -> None:
    long_text = PHONE + ("x" * 80)
    too_long = client.post("/predict", json={"text": long_text})
    assert too_long.status_code == 422
    assert "exceeds" in too_long.text
    assert PHONE not in too_long.text

    too_many = client.post("/predict_batch", json={"texts": ["ok", "ok", "ok", "ok"]})
    assert too_many.status_code == 422
    assert "batch exceeds" in too_many.text

    alias = client.post("/predict/batch", json={"texts": ["ok"] * 4})
    assert alias.status_code == 422

    empty = client.post("/predict_batch", json={"texts": []})
    assert empty.status_code == 422
    assert "at least one" in empty.text


def test_validation_error_does_not_echo_the_raw_value(client: TestClient) -> None:
    response = client.post("/predict", json={"text": {"phone": PHONE}})
    assert response.status_code == 422
    assert PHONE not in response.text
    assert "input" not in response.json()["detail"][0]


def test_pii_is_masked_before_the_model_sees_it(
    settings: Settings, tfidf_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    loaded = load_tfidf(tfidf_dir)
    recorder = Recording(loaded.predictor)
    client = TestClient(create_app(settings, predictor=recorder))
    raws = [f"call {PHONE} now", f"email {EMAIL} please"]
    caplog.set_level(logging.INFO, logger="capstone.serving")

    one = client.post("/predict", json={"text": raws[0]})
    batch = client.post("/predict_batch", json={"texts": raws})
    assert one.status_code == 200
    assert batch.status_code == 200
    assert recorder.seen == [mask_pii(raws[0]), mask_pii(raws[0]), mask_pii(raws[1])]
    for seen in recorder.seen:
        assert PHONE not in seen
        assert EMAIL not in seen
        assert "[PHONE]" in seen or "[USER]" in seen
    assert PHONE not in caplog.text
    assert EMAIL not in caplog.text
    assert PHONE not in one.text
    assert EMAIL not in batch.text


def test_missing_transformer_is_503_on_health_and_predict(tmp_path: Path) -> None:
    client = TestClient(
        create_app(Settings(backend="transformer", transformer_dir=tmp_path / "missing"))
    )
    health = client.get("/health")
    assert health.status_code == 503
    body = health.json()
    assert body["status"] == "unavailable"
    assert body["model_loaded"] is False
    assert body["backend"] == "transformer"
    assert "multitask" in body["error"]

    for path, payload in (
        ("/predict", {"text": f"call {PHONE}"}),
        ("/predict/batch", {"texts": [f"call {PHONE}"]}),
        ("/predict_batch", {"texts": [f"call {PHONE}"]}),
    ):
        response = client.post(path, json=payload)
        assert response.status_code == 503
        assert PHONE not in response.text
        assert "multitask" in response.text


def test_missing_tfidf_and_unknown_backend_are_503(tmp_path: Path) -> None:
    missing = TestClient(create_app(Settings(backend="tfidf", tfidf_dir=tmp_path / "empty")))
    health = missing.get("/health")
    assert health.status_code == 503
    assert health.json()["model_loaded"] is False
    assert "missing" in health.json()["error"]
    assert missing.post("/predict", json={"text": "hello"}).status_code == 503

    unknown = TestClient(create_app(Settings(backend="gpt")))
    response = unknown.get("/health")
    assert response.status_code == 503
    assert "MODEL_BACKEND" in response.json()["error"]


def test_from_env_cors_defaults_to_any_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    monkeypatch.delenv("CORS_ORIGIN_REGEX", raising=False)
    settings = Settings.from_env()
    assert settings.cors_origins == ("*",)
    assert settings.cors_origin_regex is None


def test_cors_allows_the_next_app_and_local_dev(client: TestClient) -> None:
    origins = [
        "https://multi-head.vercel.app",
        "https://multi-head-git-feature.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:4321",
    ]
    for origin in origins:
        response = client.options(
            "/predict",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert "POST" in response.headers["access-control-allow-methods"]

    blocked = client.post(
        "/predict",
        json={"text": "hello"},
        headers={"Origin": "https://evil.example"},
    )
    assert blocked.status_code == 200
    assert blocked.headers.get("access-control-allow-origin") is None


def test_inference_errors_do_not_echo_the_message(settings: Settings) -> None:
    exploding = TestClient(create_app(settings, predictor=Exploding()))
    failed = exploding.post("/predict", json={"text": f"call {PHONE}"})
    assert failed.status_code == 500
    assert failed.json()["detail"] == "inference failed"
    assert PHONE not in failed.text

    garbage = TestClient(create_app(settings, predictor=Garbage()))
    rejected = garbage.post("/predict", json={"text": f"call {PHONE}"})
    assert rejected.status_code == 503
    assert PHONE not in rejected.text


def test_rich_transformer_payload_maps_onto_the_public_schema() -> None:
    public = to_public(
        {
            "issue": {
                "label": "Network_Issues",
                "probs": {"Network_Issues": 0.7, "Complaint_General": 0.3},
            },
            "sentiment": {"label": "negative", "prob": 0.91},
            "urgency": {
                "level": "emergency",
                "probs": {"low": 0.1, "medium": 0.2, "emergency": 0.7},
                "emergency": True,
            },
        }
    )
    assert public["issue"] == "Network_Issues"
    assert public["sentiment"] == "negative"
    assert public["urgency"] == "emergency"
    assert public["confidence"]["issue"] == pytest.approx(0.7)
    assert public["confidence"]["sentiment"] == pytest.approx(0.91)
    assert public["urgency_score"] == pytest.approx(0.8)


def test_urgency_label_order_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(
        '{"issue":["Network_Issues"],"sentiment":["negative"],'
        '"urgency":["emergency","low","medium"]}',
        encoding="utf-8",
    )
    with pytest.raises(ModelNotReady, match="low, medium, emergency"):
        _read_labels(path)
