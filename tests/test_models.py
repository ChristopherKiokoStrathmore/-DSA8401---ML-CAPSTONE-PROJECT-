"""CORN, frozen layers, baselines, threshold, and a one-step tiny train."""

import ast
import csv
import json
from pathlib import Path

import torch
import yaml

from capstone.models.baselines import train_baselines
from capstone.models.corn import corn_class_probs, corn_loss, corn_predict
from capstone.models.metrics import choose_emergency_threshold
from capstone.models.multitask import (
    MODEL_ID,
    build_encoder,
    freeze_bottom,
    load_for_inference,
    masked_loss,
)
from capstone.models.train import train
from capstone.sample_labels import write_label_dir
from capstone.splits import build_manifests, write_manifests


def test_merged_split_keeps_rows_that_normalise_alike() -> None:
    from capstone.models.data import Example, merged_split

    issue_a = Example("Hello  there", "Network_Issues", 1.0, "issue.csv", 1)
    issue_b = Example("hello there", "Data_Bundle_Problems", 1.0, "issue.csv", 2)
    sent_a = Example("hello there", "negative", 1.0, "sentiment_hand.csv", 1)
    sent_b = Example("Hello there", "not_negative", 1.0, "sentiment_hand.csv", 2)
    splits = {
        "issue": {"train": [issue_a, issue_b], "val": [], "test": []},
        "sentiment": {"train": [sent_a, sent_b], "val": [], "test": []},
        "urgency": {"train": [], "val": [], "test": []},
    }
    merged = merged_split(splits, "train")
    assert len(merged) == 4
    assert {row.issue for row in merged if row.issue} == {
        "Network_Issues",
        "Data_Bundle_Problems",
    }
    assert {row.sentiment for row in merged if row.sentiment} == {"negative", "not_negative"}


def test_absent_transformer_is_not_evaluated(tmp_path: Path) -> None:
    from capstone.evaluate import _load_transformer

    missing = _load_transformer(tmp_path / "afroxlmr")
    assert missing == {
        "status": "not evaluated",
        "reason": "artifacts/afroxlmr is absent",
    }


def test_real_config_names_afroxlmr_and_freezes_eight_layers() -> None:
    config = yaml.safe_load(Path("configs/afroxlmr.yaml").read_text(encoding="utf-8"))
    assert config["model_name"] == MODEL_ID == "Davlan/afro-xlmr-base"
    assert config["freeze_bottom_layers"] == 8
    assert config["freeze_embeddings"] is True
    assert config["seed"] == 42
    assert config["tiny"] is False


def test_corn_probs_peak_on_the_labelled_rank() -> None:
    logits = torch.tensor(
        [
            [-8.0, -8.0],  # low
            [8.0, -8.0],  # medium
            [8.0, 8.0],  # emergency
        ]
    )
    labels = corn_predict(logits).tolist()
    assert labels == [0, 1, 2]
    probs = corn_class_probs(logits)
    assert torch.allclose(probs.sum(dim=1), torch.ones(3), atol=1e-4)
    assert probs[2, 2] > 0.9
    assert probs[0, 0] > 0.9


def test_corn_loss_is_lower_when_logits_match() -> None:
    labels = torch.tensor([0, 1, 2])
    good = torch.tensor([[-6.0, -6.0], [6.0, -6.0], [6.0, 6.0]])
    bad = torch.tensor([[6.0, 6.0], [-6.0, 6.0], [-6.0, -6.0]])
    assert float(corn_loss(good, labels)) < float(corn_loss(bad, labels))


def test_masked_loss_does_not_train_a_head_with_no_labels() -> None:
    encoder, _config = build_encoder(tiny=True)
    from capstone.models.multitask import MultiTaskEncoder

    model = MultiTaskEncoder(encoder, encoder.config.hidden_size)
    frozen = freeze_bottom(model.encoder, bottom_layers=2, freeze_embeddings=True)
    assert frozen == 2
    assert not any(parameter.requires_grad for parameter in model.encoder.embeddings.parameters())
    assert not model.encoder.encoder.layer[0].attention.output.dense.weight.requires_grad
    assert model.encoder.encoder.layer[-1].attention.output.dense.weight.requires_grad

    batch = 2
    issue_logits = model.issue(torch.zeros(batch, encoder.config.hidden_size))
    sentiment_logits = model.sentiment(torch.zeros(batch, encoder.config.hidden_size))
    urgency_logits = model.urgency(torch.zeros(batch, encoder.config.hidden_size))
    # Rebuild logits through the module so the graph includes the head weights.
    hidden = torch.randn(batch, encoder.config.hidden_size, requires_grad=True)
    issue_logits = model.issue(hidden)
    sentiment_logits = model.sentiment(hidden)
    urgency_logits = model.urgency(hidden)
    loss = masked_loss(
        issue_logits,
        sentiment_logits,
        urgency_logits,
        issue_y=torch.tensor([-100, -100]),
        sentiment_y=torch.tensor([-100, -100]),
        urgency_y=torch.tensor([0, 2]),
        urgency_weight=torch.tensor([1.0, 0.5]),
        class_weights={
            "issue": torch.ones(10),
            "sentiment": torch.ones(2),
            "urgency": torch.ones(3),
        },
    )
    loss.backward()
    assert model.issue.weight.grad is None
    assert model.sentiment.weight.grad is None
    assert model.urgency.weight.grad is not None


def test_threshold_uses_only_the_arrays_it_is_given() -> None:
    source = Path("src/capstone/models/metrics.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "choose_emergency_threshold"
    )
    names = [arg.arg for arg in function.args.args]
    assert "test" not in names
    gold = ["emergency", "emergency", "low", "low"]
    probs = [0.9, 0.8, 0.1, 0.2]
    chosen = choose_emergency_threshold(gold, probs, cap=0.20)
    assert chosen["threshold"] == 0.8
    assert chosen["meets_cap"] is True
    assert chosen["val_emergency_recall"] == 1.0
    assert chosen["val_false_emergency_rate"] == 0.0


def test_tfidf_pipeline_and_tiny_checkpoint_roundtrip(tmp_path: Path) -> None:
    labels = tmp_path / "labels"
    manifests = tmp_path / "manifests"
    write_label_dir(labels)
    write_manifests(build_manifests(labels, kind="harness"), manifests)
    tfidf_dir = tmp_path / "tfidf"
    selection = train_baselines(labels, manifests, tfidf_dir, kind="harness")
    assert set(selection["heads"]) == {"issue", "sentiment", "urgency"}
    for report in selection["heads"].values():
        assert report["chosen_on"] == "validation"
        assert "test_score" not in report
        assert report["chosen_c"] in (0.25, 1.0, 4.0)
    blob = json.dumps(selection)
    assert "test_score" not in blob
    import joblib

    pipeline = joblib.load(tfidf_dir / "issue.joblib")
    assert pipeline.named_steps["mask"].func.__module__ == "capstone.pii"
    probabilities = pipeline.predict_proba(["please reverse the bundle"])
    assert probabilities.shape[1] >= 2

    config = yaml.safe_load(Path("configs/tiny.yaml").read_text(encoding="utf-8"))
    out = tmp_path / "afroxlmr"
    first = train(config, labels, manifests, out, kind="harness")
    assert first["steps"] == 1
    assert (out / "sha256.txt").is_file()
    second = train(config | {"max_steps": 2}, labels, manifests, out, resume=True, kind="harness")
    assert second["steps"] == 2
    predictor = load_for_inference(out)
    rows = predictor.predict(["my line has been down since morning please help"])
    assert set(rows[0]) == {"issue", "sentiment", "urgency"}
    assert set(rows[0]["issue"]["probs"]) == set(
        json.loads((tfidf_dir / "labels.json").read_text(encoding="utf-8"))["issue"]
    )
    assert abs(sum(rows[0]["issue"]["probs"].values()) - 1) < 1e-4
    assert abs(sum(rows[0]["urgency"]["probs"].values()) - 1) < 1e-4
    assert "prob" in rows[0]["sentiment"]
    assert isinstance(rows[0]["urgency"]["emergency"], bool)


def test_evaluate_report_is_a_harness_and_hides_text(tmp_path: Path) -> None:
    from capstone.evaluate import evaluate

    labels = tmp_path / "labels"
    manifests = tmp_path / "manifests"
    write_label_dir(labels)
    write_manifests(build_manifests(labels, kind="harness"), manifests)
    train_baselines(labels, manifests, tmp_path / "tfidf", kind="harness")
    config = yaml.safe_load(Path("configs/tiny.yaml").read_text(encoding="utf-8"))
    train(config, labels, manifests, tmp_path / "afroxlmr", kind="harness")
    errors = tmp_path / "private" / "errors.csv"
    result = evaluate(
        label_dir=labels,
        manifest_dir=manifests,
        tfidf_dir=tmp_path / "tfidf",
        transformer_dir=tmp_path / "afroxlmr",
        baselines_dir=tmp_path / "baselines",
        out_dir=tmp_path / "reports",
        errors_path=errors,
        kind="harness",
    )
    report = (tmp_path / "reports" / "results.md").read_text(encoding="utf-8")
    assert "HARNESS CHECK" in report
    assert "my line has been down" not in report
    assert result["heads"]["issue"]["n"] > 0
    assert "n" in next(iter(result["heads"]["issue"]["models"]["tfidf"]["per_class"].values()))
    with errors.open(encoding="utf-8", newline="") as handle:
        dumped = list(csv.DictReader(handle))
    assert dumped
    assert "category_hint" in dumped[0]
    assert sum(row["taxonomy_pick"] == "true" for row in dumped) <= 30
    assert (tmp_path / "reports" / "figures" / "issue_tfidf.png").is_file()
    assert str(errors).endswith("errors.csv")
    # The default path used by the CLI lives under data/private.
    source = Path("src/capstone/evaluate.py").read_text(encoding="utf-8")
    assert 'default=Path("data/private/errors.csv")' in source
