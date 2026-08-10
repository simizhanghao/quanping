"""Deterministic parsers bound to OutputSpec.parser — no business field names."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple


def parse_identity(raw: str) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    """Treat raw text as a single free-form value under key ``value`` if needed.

    For text classification fixtures that already store structured ``parsed``,
    identity parse of raw is rarely used; when used, wrap as {\"text\": raw}.
    """
    text = (raw or "").strip()
    if not text:
        return None, False, "empty_raw_output"
    return {"text": text}, True, ""


def parse_json(raw: str) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    text = (raw or "").strip()
    if not text:
        return None, False, "empty_raw_output"
    # strip optional markdown fences
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # try first {...} object
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None, False, "json_decode_error"
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError as e:
            return None, False, f"json_decode_error:{e}"
    if not isinstance(obj, dict):
        return None, False, "json_not_object"
    return obj, True, ""


def parse_raw(raw: str, parser_name: str) -> Tuple[Optional[Dict[str, Any]], bool, str]:
    name = (parser_name or "json").strip().lower()
    if name in {"json", "json_object"}:
        return parse_json(raw)
    if name in {"identity", "text"}:
        return parse_identity(raw)
    return None, False, f"unknown_parser:{name}"
