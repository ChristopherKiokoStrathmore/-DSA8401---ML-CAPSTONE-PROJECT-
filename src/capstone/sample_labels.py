"""Fixed synthetic label files. Already masked. Not customer text.

The fixture plants a few exact and near overlaps so gold removal has something
to drop. The real sentiment test is exact-disjoint from the 1,500, and the
real urgency train file has no exact overlap with the urgency test. Those
real counts are in the label rubric. This generator does not copy them.
"""

from __future__ import annotations

import csv
from pathlib import Path

from capstone.labels import FILES, ISSUE_CLASSES

# Distinctive sentences the tests look up. Long enough to near-duplicate on a
# trailing " please" and not on the class templates below.
PLANTED_EXACT = "my line has been down since morning please help"
PLANTED_NEAR = "my line has been down since morning please help please"
STABLE_DUP = "what 1gb offers are available for kes 50 this week"
NEAR_KEEP_A = "the sim replacement has been pending for three full days"
NEAR_KEEP_B = "the sim replacement has been pending for three full days please"
ISSUE_OVERLAP = "please reverse the money sent to the wrong person yesterday"
ROUTER_A = "the home router shows no 5g light at night and calls fail"
ROUTER_B = "the home router shows no 5g light at night and calls fail please"
GEMINI_OVERLAP = "the rewards balance disappeared after the app update last week"
URGENCY_NEAR_TEST = "customers in this estate have had no signal since dawn today"
URGENCY_NEAR_TRAIN = "customers in this estate have had no signal since dawn today please"

_CLASS_LINE = {
    "Non_Actionable": "thanks the line is fine now and there is nothing else to do",
    "Network_Issues": "the mobile signal keeps dropping whenever it rains here",
    "Data_Bundle_Problems": "the 1gb bundle bought yesterday is still not reflecting",
    "Router_WiFi_5G": "wifi at the shop drops every evening after sunset",
    "SIM_Line_Services": "a replacement sim bought on monday is still inactive",
    "Product_Enquiry": "which weekly data offer suits light browsing and maps",
    "Airtel_Money_Transfer": "a shop payment is still pending after several hours",
    "App_Rewards": "points in the app went missing after the last update",
    "Complaint_General": "nobody has called back about the open service complaint",
    "Airtel_Money_Reversal": "a small transfer needs to be reversed to the sender",
}

_NEGATIVE = {
    "Network_Issues",
    "Data_Bundle_Problems",
    "Complaint_General",
    "Airtel_Money_Reversal",
}


def build_tables() -> dict[str, list[dict[str, str]]]:
    """Return the seven tables keyed by loader filename."""
    issue = _issue_rows()
    hand = []
    platform = []
    for row in issue:
        sentiment = "negative" if row["label"] in _NEGATIVE else "not_negative"
        hand.append({"text": row["text"], "sentiment": sentiment})
        platform.append(
            {
                "text": row["text"],
                "sentiment": _flip(sentiment) if _flip_platform(row["text"]) else sentiment,
            }
        )
    return {
        FILES["issue"]: issue,
        FILES["sentiment_hand"]: hand,
        FILES["sentiment_platform"]: platform,
        FILES["sentiment_test"]: _sentiment_test_rows(),
        FILES["urgency_train"]: _urgency_train_rows(),
        FILES["urgency_test"]: _urgency_test_rows(),
        FILES["gemini"]: _gemini_rows(),
    }


def write_label_dir(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tables = build_tables()
    columns = {
        FILES["issue"]: ["text", "label"],
        FILES["sentiment_hand"]: ["text", "sentiment"],
        FILES["sentiment_platform"]: ["text", "sentiment"],
        FILES["sentiment_test"]: ["row_id", "message", "your_sentiment"],
        FILES["urgency_train"]: ["base_idx", "text", "urg", "source", "weight"],
        FILES["urgency_test"]: ["base_idx", "text", "urg", "source", "weight"],
        FILES["gemini"]: ["text", "issue", "sentiment", "urgency"],
    }
    for name, rows in tables.items():
        _write(directory / name, columns[name], rows)


def _issue_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for class_index, name in enumerate(ISSUE_CLASSES):
        for copy in range(16):
            # Unique prefix so rows of one class are not near-duplicates.
            # Fully distinct text. A shared class sentence would glue the class into one group.
            text = _unique_line(name, class_index * 50 + copy, "q")
            rows.append({"text": f"{text} {_CLASS_LINE[name].split()[0]}", "label": name})
    rows.extend(
        [
            {"text": PLANTED_EXACT, "label": "Network_Issues"},
            {"text": PLANTED_EXACT, "label": "Network_Issues"},
            {"text": PLANTED_NEAR, "label": "Network_Issues"},
            {"text": STABLE_DUP, "label": "Product_Enquiry"},
            {"text": STABLE_DUP, "label": "Product_Enquiry"},
            {"text": NEAR_KEEP_A, "label": "SIM_Line_Services"},
            {"text": NEAR_KEEP_B, "label": "SIM_Line_Services"},
            {"text": ISSUE_OVERLAP, "label": "Airtel_Money_Reversal"},
            {"text": ROUTER_A, "label": "Router_WiFi_5G"},
            {
                "text": "please call [PHONE] about the 1gb bundle priced at kes 50",
                "label": "Data_Bundle_Problems",
            },
        ]
    )
    return rows


def _sentiment_test_rows() -> list[dict[str, str]]:
    rows = []
    for index in range(30):
        rows.append(
            {
                "row_id": str(5000 + index),
                "message": _unique_line("annotated", index, "s"),
                "your_sentiment": "negative" if index % 2 == 0 else "not_negative",
            }
        )
    rows.append(
        {
            "row_id": "5099",
            "message": PLANTED_EXACT,
            "your_sentiment": "negative",
        }
    )
    rows.append(
        {
            "row_id": "5100",
            "message": ROUTER_B,
            "your_sentiment": "negative",
        }
    )
    return rows


def _urgency_train_rows() -> list[dict[str, str]]:
    rows = []
    urg_cycle = ("0.0", "0.5", "1.0")
    for index in range(30):
        rows.append(
            {
                "base_idx": str(index),
                "text": _unique_line("goldfault", index, "g"),
                "urg": urg_cycle[index % 3],
                "source": "gold",
                "weight": "1.0",
            }
        )
    for index in range(50):
        rows.append(
            {
                "base_idx": str(100 + index),
                "text": _unique_line("distilledfault", index, "d"),
                "urg": urg_cycle[index % 3],
                "source": "distilled",
                "weight": "0.5",
            }
        )
    rows.append(
        {
            "base_idx": "900",
            "text": URGENCY_NEAR_TRAIN,
            "urg": "1.0",
            "source": "distilled",
            "weight": "0.5",
        }
    )
    return rows


def _urgency_test_rows() -> list[dict[str, str]]:
    rows = []
    urg_cycle = ("0.0", "0.5", "1.0")
    for index in range(18):
        rows.append(
            {
                "base_idx": str(index),
                "text": _unique_line("fixedgold", index, "t"),
                "urg": urg_cycle[index % 3],
                "source": "gold",
                "weight": "1.0",
            }
        )
    rows.extend(
        [
            {
                "base_idx": "80",
                "text": ISSUE_OVERLAP,
                "urg": "1.0",
                "source": "gold",
                "weight": "1.0",
            },
            {
                "base_idx": "81",
                "text": GEMINI_OVERLAP,
                "urg": "0.5",
                "source": "gold",
                "weight": "1.0",
            },
            {
                "base_idx": "82",
                "text": URGENCY_NEAR_TEST,
                "urg": "1.0",
                "source": "gold",
                "weight": "1.0",
            },
        ]
    )
    return rows


def _gemini_rows() -> list[dict[str, str]]:
    rows = []
    classes = list(ISSUE_CLASSES)
    for index in range(15):
        rows.append(
            {
                "text": _unique_line("geminisilver", index, "m"),
                "issue": classes[index % len(classes)],
                "sentiment": "negative" if index % 2 == 0 else "not_negative",
                "urgency": ("low", "medium", "emergency")[index % 3],
            }
        )
    rows.append(
        {
            "text": GEMINI_OVERLAP,
            "issue": "App_Rewards",
            "sentiment": "negative",
            "urgency": "medium",
        }
    )
    return rows


def _unique_line(stem: str, index: int, tag: str) -> str:
    """A line whose character 3-grams barely overlap the neighbouring index."""
    tokens = " ".join(f"{tag}{index}z{slot}" for slot in range(10))
    return f"{stem}-{index:02d} {tokens} end {index * 41 + 7}"


def _flip(sentiment: str) -> str:
    return "not_negative" if sentiment == "negative" else "negative"


def _flip_platform(text: str) -> bool:
    """Stable across duplicate copies of the same text."""
    total = sum(ord(char) for char in text)
    return total % 3 == 0


def _write(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
