"""Contract for mask_pii. These examples are the spec for training and serving."""

import re

import pytest

from capstone.pii import PLACEHOLDERS, mask_pii

PHONES = [
    "0712345678",
    "0712 345 678",
    "0712-345-678",
    "0712.345.678",
    "0712/345/678",
    "0712 345678",
    "+254712345678",
    "+254 712 345 678",
    "+254-712-345-678",
    "254712345678",
    "254 712 345 678",
    "0112345678",
    "0112 345 678",
    "+254 112 345 678",
    "+254 (0) 712 345 678",
    "+2540712345678",
    "(0712) 345 678",
    "0712\u2013345\u2013678",
]


@pytest.mark.parametrize("phone", PHONES)
def test_phone_becomes_placeholder(phone: str) -> None:
    assert mask_pii(phone) == "[PHONE]"
    assert mask_pii(f"call {phone} now") == "call [PHONE] now"


def test_two_phones() -> None:
    assert mask_pii("0712345678 or 0112345678") == "[PHONE] or [PHONE]"


@pytest.mark.parametrize(
    "text",
    [
        "1GB",
        "1.5GB",
        "500MB",
        "10 GB",
        "10GBBUNDLE",
        "KES 1500",
        "KES 1,500",
        "100 bob",
        "sh 50",
        "balance is 250",
        "I have 2 lines",
        "2025-11-01",
        "01/11/2025",
        "17/04/2026",
        "07/12/2025",
        "07:30",
        "*544#",
        "*334*1#",
        "100",
        "12345678",
        "Nimeweka bundle ya 1GB lakini haijaingia",
        "Boss, bundle imecheat, si unfix",
        "Kwa nini mtandao ni mbaya leo?",
        "bundle ya 10GB ni KES 50 tu",
    ],
)
def test_non_pii_survives(text: str) -> None:
    assert mask_pii(text) == text


def test_phone_keeps_bundle_and_price() -> None:
    raw = "Nambari 0712345678, bundle 1GB ni KES 50"
    assert mask_pii(raw) == "Nambari [PHONE], bundle 1GB ni KES 50"


def test_long_digit_string_is_not_a_phone() -> None:
    assert mask_pii("1006435192554682") == "1006435192554682"


def test_email_and_handle_share_user_token() -> None:
    assert mask_pii("jane.doe@example.com") == "[USER]"
    assert mask_pii("mail me at jane.doe+test@example.co.ke thanks") == ("mail me at [USER] thanks")
    assert mask_pii("ping @fake_user please") == "ping [USER] please"
    assert mask_pii("@Airtel_KE") == "[USER]"
    assert "@" not in mask_pii("jane.doe@example.com")


def test_urls() -> None:
    assert mask_pii("https://example.com/path?q=1") == "[URL]"
    assert mask_pii("see www.example.com.") == "see [URL]."
    assert mask_pii("go to http://example.co.ke/a, then wait") == "go to [URL], then wait"
    assert mask_pii("(https://example.com/a)") == "([URL])"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ID 12345678", "[ID]"),
        ("ID: 32345678", "[ID]"),
        ("ID no 1234567", "[ID]"),
        ("ID no. 12345678", "[ID]"),
        ("national id 23456789", "[ID]"),
        ("my ID is 12345678 today", "my [ID] today"),
        ("kitambulisho 12345678", "[ID]"),
        ("kitambulisho namba 1234567", "[ID]"),
        ("balance 12345678", "balance 12345678"),
        ("KES 12345", "KES 12345"),
    ],
)
def test_national_id(raw: str, expected: str) -> None:
    assert mask_pii(raw) == expected


def test_account_and_transaction_codes_use_id_token() -> None:
    assert mask_pii("account 1234567890") == "[ID]"
    assert mask_pii("acc no. 99887766") == "[ID]"
    assert mask_pii("paybill 123456") == "[ID]"
    assert mask_pii("account manager will call") == "account manager will call"
    assert mask_pii("account 0712345678") == "account [PHONE]"
    assert mask_pii("mpesa code QW12ER34TY") == "[ID]"
    assert mask_pii("The code is QW12ER34TY thanks") == "The code is [ID] thanks"
    assert mask_pii("transaction code qw12er34ty") == "[ID]"
    assert mask_pii("qw12er34ty") == "qw12er34ty"


def test_mixed_message_uses_only_the_four_tokens() -> None:
    raw = (
        "Email jane@example.com or call 0712345678 see https://example.com/a "
        "and @jane_k id 12345678"
    )
    masked = mask_pii(raw)
    assert masked == "Email [USER] or call [PHONE] see [URL] and [USER] [ID]"
    found = set(re.findall(r"\[[A-Z]+\]", masked))
    assert found <= set(PLACEHOLDERS)
    assert "[EMAIL]" not in masked
    assert "[HANDLE]" not in masked
    assert "[TXN]" not in masked
    assert "[ACCOUNT]" not in masked


def test_idempotent_and_empty() -> None:
    assert mask_pii("") == ""
    assert mask_pii(None) == ""  # type: ignore[arg-type]
    already = "call [PHONE] about [USER] via [URL] ref [ID]"
    assert mask_pii(already) == already
    once = mask_pii("reach 0712345678 or jane@example.com")
    assert mask_pii(once) == once


def test_non_string_raises() -> None:
    with pytest.raises(TypeError):
        mask_pii(712345678)  # type: ignore[arg-type]
