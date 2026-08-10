"""Deterministic surface transforms (P2-B). No LLM; seed only for lineage."""

from __future__ import annotations

import re
import string
from typing import Optional

from linguaeval.core.schema import PerturbationSpec, SampleInput

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS_RE = re.compile(r"\s+")


def _get_text(inp: SampleInput) -> str:
    if inp.text is not None:
        return str(inp.text)
    if inp.messages:
        for m in reversed(inp.messages):
            if isinstance(m, dict) and m.get("content") is not None:
                return str(m["content"])
    return ""


def _set_text(inp: SampleInput, text: str) -> SampleInput:
    if inp.text is not None or not inp.messages:
        return SampleInput(text=text, messages=inp.messages)
    # rewrite last message content
    msgs = [dict(m) for m in inp.messages]
    for i in range(len(msgs) - 1, -1, -1):
        if "content" in msgs[i]:
            msgs[i] = {**msgs[i], "content": text}
            break
    return SampleInput(text=None, messages=msgs)


def apply_case_lower(inp: SampleInput, spec: PerturbationSpec) -> SampleInput:
    return _set_text(inp, _get_text(inp).lower())


def apply_strip_punctuation(inp: SampleInput, spec: PerturbationSpec) -> SampleInput:
    return _set_text(inp, _get_text(inp).translate(_PUNCT_TABLE))


def apply_collapse_whitespace(inp: SampleInput, spec: PerturbationSpec) -> SampleInput:
    t = _get_text(inp).strip()
    t = _WS_RE.sub(" ", t)
    return _set_text(inp, t)
