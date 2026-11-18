"""Mask PII in customer messages.

The labelled training files already use exactly four placeholders. Serving must
emit the same tokens so the text the model sees at inference matches training.

    [PHONE]  Kenyan phone numbers
    [URL]    http(s) and www URLs
    [USER]   email addresses and @handles
    [ID]     national IDs, and account or transaction-like codes

Emails and handles both become [USER]. There is no [EMAIL] token. Account
numbers and M-Pesa-like confirmation codes become [ID], because they are
identifiers and the label set has no fifth placeholder.

The function is idempotent: text that already contains these placeholders is
unchanged by a second pass. Prices, dates, USSD codes, and bundle sizes such
as 1GB are left alone.
"""

from __future__ import annotations

import re

PLACEHOLDERS: tuple[str, ...] = ("[ID]", "[PHONE]", "[URL]", "[USER]")

# Separators seen between digits in Kenyan numbers. Slash is included so
# 0712/345/678 is masked; dates like 01/11/2025 are too short to match.
_SEP = r"[\s./\-\u00a0\u2010-\u2015\u2212]"

# Local 0[17]xxxxxxxx (10 digits) and 254 / +254 with an optional trunk 0.
_PHONE = re.compile(
    rf"(?<!\w)(?:"
    rf"(?:\+?254){_SEP}*\(0\){_SEP}*[17](?:{_SEP}*\d){{8}}"
    rf"|(?:\+?254){_SEP}*0?{_SEP}*[17](?:{_SEP}*\d){{8}}"
    rf"|\(0[17]\d{{2}}\)(?:{_SEP}*\d){{6}}"
    rf"|0[17](?:{_SEP}*\d){{8}}"
    rf")(?!\w)"
)

_URL = re.compile(r"(?i)(?<![\w@])(?:https?://|www\.)[^\s<>\[\]\"']+")

_EMAIL = re.compile(r"(?<![\w.])[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

_HANDLE = re.compile(r"(?<![\w@])@[A-Za-z][A-Za-z0-9_]{1,30}\b")

# Labelled Kenyan national ID: 7 or 8 digits. Bare numbers are not IDs.
_ID = re.compile(
    r"(?i)\b(?:national\s+id|nat\.?\s*id|kitambulisho|id\s*(?:no\.?|number|#)?)"
    r"\s*(?:is|ni|namba)?\s*[:#.\-]?\s*\d{7,8}\b"
)

# Labelled account / till / paybill. The token must contain a digit so
# "account manager" survives. Phone-shaped values are already [PHONE].
_ACCOUNT = re.compile(
    r"(?i)\b(?:account|acct|a/c|acc|paybill|till)\b"
    r"\s*(?:(?:no\.?|number|#)\s*)?"
    r"[:#.\-]?\s*"
    r"(?=[A-Za-z0-9\-]*\d)[A-Za-z0-9][A-Za-z0-9\-]{4,19}\b"
)

# "transaction code QW12ER34TY" and similar, any letter case.
_TXN_LABELLED = re.compile(
    r"(?i)\b(?:txn|transaction(?:\s+code)?|m-?pesa(?:\s+code)?|"
    r"confirmation(?:\s+code)?)\s*[:#.\-]?\s*[A-Za-z0-9]{8,12}\b"
)

# Bare M-Pesa-style codes: exactly 10 uppercase characters, at least three
# letters and three digits. Lowercase words and bundle tokens are not matched.
_TXN_BARE = re.compile(r"\b(?=(?:[A-Z]*\d){3,})(?=(?:\d*[A-Z]){3,})[A-Z0-9]{10}\b")

_TRAILING_URL_PUNCT = ".,;:!?)]}'\""


def mask_pii(text: str) -> str:
    """Replace PII in ``text`` with [ID], [PHONE], [URL], or [USER].

    ``text`` must be a ``str``. ``None`` returns an empty string so a missing
    field can be passed through at the edge of a caller. Any other type
    raises ``TypeError``.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        raise TypeError("mask_pii expects str")
    if text == "":
        return ""

    masked = _URL.sub(_url_token, text)
    masked = _EMAIL.sub("[USER]", masked)
    masked = _HANDLE.sub("[USER]", masked)
    masked = _PHONE.sub("[PHONE]", masked)
    masked = _ID.sub("[ID]", masked)
    masked = _ACCOUNT.sub("[ID]", masked)
    masked = _TXN_LABELLED.sub("[ID]", masked)
    masked = _TXN_BARE.sub("[ID]", masked)
    return masked


def _url_token(match: re.Match[str]) -> str:
    url = match.group(0)
    trimmed = url.rstrip(_TRAILING_URL_PUNCT)
    if not trimmed:
        return url
    return "[URL]" + url[len(trimmed) :]
