"""Field-level redaction helpers for logs, traces, and handoff summaries."""

from __future__ import annotations

import copy
import re
from typing import Any


REDACTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("[ID_CARD]", re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)")),
    ("[PHONE]", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("[EMAIL]", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("[ORDER_ID]", re.compile(r"#\d{4,}|\bORDER[-_]?\d+\b", re.IGNORECASE)),
    (
        "[ADDRESS]",
        re.compile(r"[\u4e00-\u9fff]{2,}(?:省|市|区|县|镇|街道|路|号楼|单元|室)[\u4e00-\u9fff0-9A-Za-z\-]*"),
    ),
]

SENSITIVE_KEYS = {
    "phone",
    "mobile",
    "email",
    "address",
    "id_card",
    "identity",
    "order_id",
    "user_name",
    "name",
}


def redact_sensitive_fields(value: Any) -> Any:
    """Return a copy of *value* with common customer PII redacted."""
    if isinstance(value, str):
        redacted = value
        for replacement, pattern in REDACTION_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        redacted = _redact_chinese_name_hint(redacted)
        return redacted

    if isinstance(value, list):
        return [redact_sensitive_fields(item) for item in value]

    if isinstance(value, tuple):
        return tuple(redact_sensitive_fields(item) for item in value)

    if isinstance(value, dict):
        copied = copy.deepcopy(value)
        for key, item in list(copied.items()):
            key_text = str(key).lower()
            if key_text in SENSITIVE_KEYS or any(part in key_text for part in SENSITIVE_KEYS):
                copied[key] = _placeholder_for_key(key_text)
            else:
                copied[key] = redact_sensitive_fields(item)
        return copied

    return value


def _placeholder_for_key(key: str) -> str:
    if "order" in key:
        return "[ORDER_ID]"
    if "email" in key:
        return "[EMAIL]"
    if "phone" in key or "mobile" in key:
        return "[PHONE]"
    if "address" in key:
        return "[ADDRESS]"
    if "id" in key or "identity" in key:
        return "[ID_CARD]"
    return "[NAME]"


def _redact_chinese_name_hint(text: str) -> str:
    # Redact explicit name mentions while avoiding broad replacement of normal prose.
    return re.sub(r"(姓名|收件人|联系人)[:：]\s*[\u4e00-\u9fff]{2,4}", r"\1：[NAME]", text)
