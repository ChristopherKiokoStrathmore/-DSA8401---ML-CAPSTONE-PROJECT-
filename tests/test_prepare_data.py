"""Schema checks, masking, inclusion counts, and duplicate ids on a tiny frame."""

from pathlib import Path

import pandas as pd
import pytest

from capstone.generate_sample import generate_rows
from capstone.prepare_data import SchemaError, main, prepare_frame, render_audit, run


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _base(**overrides: object) -> dict:
    row = {
        "Channel": "Airtel Kenya",
        "Type": "Message",
        "Posted on": "2025-12-01",
        "Text": "My line has been down since morning",
        "Last action": "archived",
        "Action date": "2025-12-02",
        "Sentiment": None,
        "Tags": None,
        "Notes": None,
    }
    row.update(overrides)
    return row


def test_missing_column_fails_loudly() -> None:
    frame = _frame([_base()])
    frame = frame.drop(columns=["Text"])
    with pytest.raises(SchemaError, match="Text"):
        prepare_frame(frame)


def test_channel_and_type_keep_originals() -> None:
    frame = _frame(
        [
            _base(
                Channel="'@AIRTEL_KE",
                Type="Comment Mention",
                Text="Please check this line",
            ),
            _base(
                Channel="Airtel Kenya (+254 100 301301)",
                Type="message",
                Text="Another real complaint about data",
            ),
            _base(
                Channel="Airtel Kenya (254101301301-1006435192554682) (+254 101 301301)",
                Type="Message",
                Text="Third complaint about the sim",
            ),
            _base(
                Channel="Airtel Kenya Twitter Search",
                Type="direct message",
                Text="Fourth complaint from search",
            ),
            _base(Channel="'@airtel_kenya", Type="Comment", Text="Fifth complaint here"),
        ]
    )
    out, _audit = prepare_frame(frame)
    norms = set(
        zip(out["Channel"], out["Channel_norm"], out["Type"], out["Type_norm"], strict=True)
    )
    assert ("'@AIRTEL_KE", "airtel_ke", "Comment Mention", "comment mention") in norms
    assert ("Airtel Kenya (+254 100 301301)", "airtel_kenya", "message", "message") in norms
    assert (
        "Airtel Kenya (254101301301-1006435192554682) (+254 101 301301)",
        "airtel_kenya",
        "Message",
        "message",
    ) in norms
    assert (
        "Airtel Kenya Twitter Search",
        "airtel_kenya_twitter_search",
        "direct message",
        "direct message",
    ) in norms
    assert ("'@airtel_kenya", "airtel_kenya_handle", "Comment", "comment") in norms


def test_overflow_reported_then_dropped_and_text_masked() -> None:
    frame = _frame(
        [
            _base(
                Text="Call 0712345678 about the 1GB bundle",
                Notes="email me fake.user@example.com",
                **{"Unnamed: 9": "spill-a", "Unnamed: 28": "spill-b"},
            )
        ]
    )
    out, audit = prepare_frame(frame)
    assert "Unnamed: 9" not in out.columns
    assert "Unnamed: 28" not in out.columns
    assert audit["overflow"]["Unnamed: 9"] == 1
    assert audit["overflow"]["Unnamed: 28"] == 1
    assert out.loc[0, "Text"] == "Call [PHONE] about the 1GB bundle"
    assert out.loc[0, "text_norm"] == "call [phone] about the 1gb bundle"
    assert out.loc[0, "Notes"] == "email me [USER]"
    assert "0712345678" not in out.loc[0, "Text"]
    assert "1GB" in out.loc[0, "Text"]


def test_inclusion_counts_and_audit_omits_text() -> None:
    secret = "zebraunique bundle complaint about outages"
    frame = _frame(
        [
            _base(Text=secret),
            _base(Text=None),
            _base(Text="   "),
            _base(Text="😂😂"),
            _base(Text="12345"),
            _base(Text="???"),
            _base(Text="This is an automated message. Please do not reply."),
            _base(Text="Your request has been received."),
            _base(Text="Your recharge was successful."),
            _base(Text="Dear Customer, your bundle has been processed successfully."),
            _base(Text="Airtel Money: you have received KES 100. New balance is KES 20."),
            _base(Text="This message was deleted"),
            _base(Text="My airtel money transfer failed, please help"),
            _base(Text="Kwa nini mtandao ni mbaya?"),
            _base(Text="Still waiting on the line repair", **{"Posted on": "not-a-date"}),
            _base(Text="This started long ago on the line", **{"Posted on": "2020-01-15"}),
            _base(Text="This is scheduled far ahead on the line", **{"Posted on": "2030-01-01"}),
        ]
    )

    out, audit = prepare_frame(frame)
    assert audit["inclusion"]["empty"] == 2
    assert audit["inclusion"]["non_linguistic"] == 3
    assert audit["inclusion"]["automated_notice"] == 1
    assert audit["inclusion"]["ticket_acknowledgement"] == 1
    assert audit["inclusion"]["payment_or_bundle_receipt"] == 1
    assert audit["inclusion"]["dear_customer_template"] == 1
    assert audit["inclusion"]["airtel_money_receipt"] == 1
    assert audit["inclusion"]["deleted_or_unavailable"] == 1
    assert "My airtel money transfer failed, please help" in set(out["Text"])
    assert secret in set(out["Text"])
    assert audit["dates"]["Posted on"]["unparseable"] >= 1
    assert audit["dates"]["Posted on"]["before_window"] >= 1
    assert audit["dates"]["Posted on"]["after_window"] >= 1
    rendered = render_audit(audit, kind="harness")
    assert "HARNESS CHECK" in rendered
    assert "zebraunique" not in rendered
    assert secret not in rendered


def test_duplicate_groups_on_kept_rows() -> None:
    frame = _frame(
        [
            _base(Text="my line has been down since morning please help"),
            _base(Text="my line has been down since morning please help"),
            _base(Text="my line has been down since morning please help!"),
            _base(Text="router ya nyumba haionyeshi 5g kabisa"),
        ]
    )
    out, audit = prepare_frame(frame)
    groups = out["dup_group_id"].tolist()
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
    assert audit["duplicates"]["extra_copies"] == 1
    assert audit["duplicates"]["near_groups"] == 1


def test_csv_and_xlsx_round_trip(tmp_path: Path) -> None:
    frame = _frame(
        [
            _base(Text="Call 0711000000 about 1GB", **{"Unnamed: 9": "spill-a"}),
            _base(Text=""),
            _base(Text="😂"),
        ]
    )
    csv_path = tmp_path / "book.csv"
    xlsx_path = tmp_path / "book.xlsx"
    frame.to_csv(csv_path, index=False)
    frame.to_excel(xlsx_path, index=False)
    for source in (csv_path, xlsx_path):
        audit = run(
            source,
            tmp_path / source.stem,
            tmp_path / f"{source.stem}.md",
            kind="harness",
        )
        modelled = pd.read_parquet(tmp_path / source.stem / "messages.parquet")
        assert audit["rows_out"] == 1
        assert modelled.loc[0, "Text"] == "Call [PHONE] about 1GB"
        assert "Unnamed: 9" not in modelled.columns
        assert "HARNESS CHECK" in Path(audit["audit_path"]).read_text(encoding="utf-8")


def test_cli_missing_file_and_missing_column(tmp_path: Path) -> None:
    assert main([str(tmp_path / "nope.csv")]) == 1
    bad = tmp_path / "bad.csv"
    bad.write_text("Channel,Type\nAirtel Kenya,message\n", encoding="utf-8")
    assert main([str(bad), "--out", str(tmp_path / "out"), "--audit", str(tmp_path / "a.md")]) == 1


def test_synthetic_sample_pipeline_is_a_harness() -> None:
    rows = generate_rows(n=400, seed=42)
    frame = pd.DataFrame(rows)
    out, audit = prepare_frame(frame)
    rendered = render_audit(audit, kind="harness")
    blob = "\n".join(out["Text"].fillna("").tolist() + out["Notes"].fillna("").tolist())
    assert "0711000000" not in blob
    assert "0712 345 678" not in blob
    assert "fake.user@example.com" not in blob
    assert "[PHONE]" in blob
    assert "[USER]" in blob
    assert "[URL]" in blob
    assert "[ID]" in blob
    assert "1GB" in blob or "1gb" in blob.lower()
    assert audit["rows_out"] < audit["rows_in"]
    assert audit["inclusion"]["empty"] > 0
    assert audit["inclusion"]["non_linguistic"] > 0
    assert audit["overflow"]["Unnamed: 9"] > 0
    assert "Unnamed: 9" not in out.columns
    assert "imecheat" not in rendered
    base = "my line has been down since morning, please help"
    help_rows = out[out["text_norm"].isin([base, base + "!", base + " please"])]
    assert len(help_rows) >= 3
    assert help_rows["dup_group_id"].nunique() == 1
