"""Deterministic synthetic workbook in the real column layout.

Every message is invented. Seed 42. The file is safe to commit under
``data/sample/``. Real exports never go here.
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path

COLUMNS = [
    "Channel",
    "Type",
    "Posted on",
    "Text",
    "Last action",
    "Action date",
    "Sentiment",
    "Tags",
    "Notes",
    "Unnamed: 9",
    "Unnamed: 10",
    "Unnamed: 11",
]

CHANNELS = [
    "Airtel Kenya",
    "'@AIRTEL_KE",
    "Airtel Kenya (+254 100 301301)",
    "Airtel Kenya (254101301301-1006435192554682) (+254 101 301301)",
    "'@airtel_kenya",
    "Airtel Kenya Twitter Search",
]
TYPES = [
    "message",
    "comment",
    "Message",
    "direct message",
    "tweet",
    "reply",
    "Comment",
    "Comment Mention",
    "Story Mention Message",
    "Media Caption Mention",
    "post",
]
ACTIONS = ["archived", "unread", "removed", "prioritized", "deleted"]
BUNDLES = ["1GB", "1.5GB", "2GB", "500MB", "10GB"]
AMOUNTS = ["20", "50", "100", "200", "1500"]

# Invented English, Swahili, and Sheng lines. Not customer records.
TEMPLATES = [
    "My line has been down since morning, please help",
    "I bought {bundle} yesterday and it is not reflecting",
    "Why was I charged KES {amount} for a bundle I did not request?",
    "Kwa nini mtandao ni mbaya {days}?",
    "Nimeweka bundle ya {bundle} lakini haijaingia",
    "Naomba msaada, laini yangu imezimwa",
    "Boss, bundle imecheat, si unfix",
    "Msee data imeisha ovyo na bado niko na siku",
    "Router haionyeshi ishara ya 5G tangu {days}",
    "App ya rewards inasema points zimepotea",
    "Please reverse the wrong money payment of KES {amount}",
    "SIM mpya haijaactivate tangu nilinunua",
    "WiFi ya nyumba inakatika kila usiku",
    "What {bundle} offers do you have for KES {amount}?",
    "The agent sorted the line, this is only a thank you",
    "Habari, niko na swali kuhusu vocha ya {amount} bob",
    "Network 😩 tena, calls are dropping {days}",
    "Balance shows KES {amount} but {bundle} did not load",
    "I dialled *544# and the menu closed",
    "Posted around 2025-11-01 and still waiting",
]

PII_TAILS = [
    " Call 0711000000.",
    " My email is fake.user@example.com.",
    " Details at https://example.com/ticket/1.",
    " Ping @fakeuser when it is fixed.",
    " National id 12345678 on the form.",
    " Account 9988776655 was charged.",
    " Mpesa code QW12ER34TY is the one.",
    " Use +254 711 000 000 or 0111000000.",
]


def generate_rows(n: int = 400, seed: int = 42) -> list[dict[str, str]]:
    """Build ``n`` fictional rows. ``seed`` fixes the filler rows."""
    if n < 80:
        raise ValueError("n must be at least 80 so the fixed edge cases fit")
    rows = _fixed_rows()
    rng = random.Random(seed)
    reserved = 24
    while len(rows) < n - reserved:
        rows.append(_filler_row(rng, len(rows)))
    _append_duplicate_pairs(rows)
    if len(rows) != n:
        raise RuntimeError(f"expected {n} rows, built {len(rows)}")
    return rows


def write_sample(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the synthetic sample workbook and labels.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--out", type=Path, default=Path("data/sample/messages.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/sample/labels"))
    parser.add_argument(
        "--skip-labels",
        action="store_true",
        help="write only the workbook",
    )
    args = parser.parse_args(argv)
    rows = generate_rows(n=args.rows, seed=args.seed)
    write_sample(args.out, rows)
    print(f"wrote {len(rows)} synthetic rows to {args.out}")
    if not args.skip_labels:
        from capstone.sample_labels import write_label_dir

        write_label_dir(args.labels)
        print(f"wrote synthetic label files to {args.labels}")
    return 0


def _fixed_rows() -> list[dict[str, str]]:
    """Edge cases the pipeline must count: empties, junk, receipts, bad dates, PII."""
    specs: list[tuple[str, str, str]] = [
        ("", "2025-11-02", ""),
        ("", "2025-11-03", ""),
        ("   ", "2025-11-04", ""),
        ("😂😂😂", "2025-11-05", ""),
        ("???", "2025-11-06", ""),
        ("12345", "2025-11-07", ""),
        ("*544#", "2025-11-08", ""),
        ("This is an automated message. Please do not reply.", "2025-11-09", ""),
        ("Your request has been received.", "2025-11-10", ""),
        ("Your recharge was successful.", "2025-11-11", ""),
        (
            "Dear Customer, your bundle has been processed successfully. Reference number AB12.",
            "2025-11-12",
            "",
        ),
        (
            "Airtel Money: you have received KES 100. New balance is KES 500.",
            "2025-11-13",
            "",
        ),
        ("This message was deleted", "2025-11-14", ""),
        ("My line has been down since morning, please help", "2025-12-01", ""),
        ("My line has been down since morning, please help", "2025-12-01", ""),
        ("My line has been down since morning, please help!", "2025-12-02", ""),
        ("Please call 0712 345 678 about the 1GB bundle that cost KES 50", "2025-12-03", "1"),
        ("Email fake.user@example.com and see https://example.com/help", "2025-12-04", "0"),
        ("Ping @fakeuser, national id 23456789, account 1234567890", "2025-12-05", "-1"),
        ("Transaction code QW12ER34TY for KES 1,500 on 01/11/2025", "2025-12-06", ""),
        ("Nimeweka bundle ya 1GB lakini haijaingia", "not-a-date", ""),
        ("Kwa nini mtandao ni mbaya leo?", "2020-01-15", ""),
        ("Boss, bundle imecheat, si unfix", "2030-05-01", ""),
        ("Router haionyeshi ishara ya 5G tangu jana", "32/13/2025", ""),
        ("Naomba msaada, laini yangu imezimwa", "2026-04-17", ""),
        ("What 10 GB offers do you have?", "2026-04-18", ""),
        ("I have 2 lines and the balance is 250 bob", "07/12/2025", ""),
        ("Calls drop at 07:30 every evening", "2026-01-15", ""),
    ]
    rows = []
    for index, (text, posted, sentiment) in enumerate(specs):
        rows.append(
            _row(
                index=index,
                text=text,
                posted=posted,
                sentiment=sentiment,
                note="Synthetic note with 0711000000." if index == 16 else "",
                tag="network" if text else "",
                spill=index % 5 == 0,
            )
        )
    # Three more null-text rows (missing Text, not an empty string).
    for _extra in range(3):
        row = _row(
            index=len(rows),
            text="",
            posted="2025-11-20",
            sentiment="",
            note="",
            tag="",
            spill=False,
        )
        row["Text"] = ""
        rows.append(row)
    return rows


def _append_duplicate_pairs(rows: list[dict[str, str]]) -> None:
    """Add 12 exact copies and 12 one-word-longer near copies of kept lines."""
    sources = [
        row
        for row in rows
        if row["Text"].startswith("My line has been down")
        or row["Text"].startswith("Nimeweka bundle")
        or row["Text"].startswith("I bought")
    ]
    if len(sources) < 12:
        sources = [row for row in rows if any(char.isalpha() for char in row["Text"])]
    for index in range(12):
        source = sources[index % len(sources)]
        exact = dict(source)
        rows.append(exact)
        near = dict(source)
        near["Text"] = source["Text"].rstrip("!.") + " please"
        near["Posted on"] = "2026-02-02"
        rows.append(near)


def _filler_row(rng: random.Random, index: int) -> dict[str, str]:
    template = TEMPLATES[index % len(TEMPLATES)]
    text = template.format(
        bundle=rng.choice(BUNDLES),
        amount=rng.choice(AMOUNTS),
        days=rng.choice(["leo", "jana", "since morning", "for two days"]),
    )
    if rng.random() < 0.35:
        text += rng.choice(PII_TAILS)
    start = date(2025, 11, 1).toordinal()
    end = date(2026, 4, 17).toordinal()
    posted = date.fromordinal(rng.randint(start, end))
    sentiment_roll = rng.random()
    if sentiment_roll < 0.90:
        sentiment = ""
    elif sentiment_roll < 0.94:
        sentiment = "-1"
    elif sentiment_roll < 0.98:
        sentiment = "0"
    else:
        sentiment = "1"
    return _row(
        index=index,
        text=text,
        posted=posted.isoformat(),
        sentiment=sentiment,
        note="Follow up tomorrow." if rng.random() < 0.1 else "",
        tag=rng.choice(["", "network", "billing", "app"]),
        spill=index % 37 == 0,
        action_on=posted + timedelta(days=int(rng.randint(0, 3))),
    )


def _row(
    *,
    index: int,
    text: str,
    posted: str,
    sentiment: str,
    note: str,
    tag: str,
    spill: bool,
    action_on: date | None = None,
) -> dict[str, str]:
    if action_on is None:
        action_on = date(2025, 11, 2) + timedelta(days=index % 40)
    return {
        "Channel": CHANNELS[index % len(CHANNELS)],
        "Type": TYPES[index % len(TYPES)],
        "Posted on": posted,
        "Text": text,
        "Last action": ACTIONS[0] if index % 17 else ACTIONS[index % len(ACTIONS)],
        "Action date": action_on.isoformat(),
        "Sentiment": sentiment,
        "Tags": tag,
        "Notes": note,
        "Unnamed: 9": "spill-a" if spill else "",
        "Unnamed: 10": "2025-11-01 00:00:00" if spill else "",
        "Unnamed: 11": "",
    }


if __name__ == "__main__":
    raise SystemExit(main())
