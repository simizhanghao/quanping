"""Explicit multiple-choice answer encodings — no 0/1-based heuristics."""

from __future__ import annotations

from typing import Any, FrozenSet

ENCODING_LETTER = "letter"
ENCODING_ZERO = "zero_based_index"
ENCODING_ONE = "one_based_index"
VALID_ENCODINGS = frozenset({ENCODING_LETTER, ENCODING_ZERO, ENCODING_ONE})

_LETTERS: FrozenSet[str] = frozenset({"A", "B", "C", "D", "E"})
_ZERO_BASED = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
_ONE_BASED = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}


class AnswerEncodingError(ValueError):
    """Raised when answer_encoding is missing or label cannot be mapped."""


def require_answer_encoding(raw: Any, *, where: str = "source.answer_encoding") -> str:
    if raw is None or str(raw).strip() == "":
        raise AnswerEncodingError(
            f"{where} is required; choose one of "
            f"{sorted(VALID_ENCODINGS)} (no silent 0/1-based guessing)"
        )
    enc = str(raw).strip().lower()
    if enc not in VALID_ENCODINGS:
        raise AnswerEncodingError(
            f"unsupported answer_encoding={raw!r}; expected one of {sorted(VALID_ENCODINGS)}"
        )
    return enc


def as_mc_letter(raw: Any, *, encoding: str) -> str:
    """Map gold/pred to A–E using an explicit encoding.

    Letter strings (A–E) are always accepted. Integer indices are interpreted
    only according to ``encoding`` — never guessed.
    """
    enc = require_answer_encoding(encoding, where="answer_encoding")
    if raw is None:
        raise AnswerEncodingError("empty answer/target")

    # Unwrap common nested lm-eval response shapes
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
        if isinstance(raw, (list, tuple)) and raw:
            raw = raw[0]

    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            raise AnswerEncodingError("empty answer string")
        if s.upper() in _LETTERS:
            return s.upper()
        head = s[0].upper()
        if head in _LETTERS and (len(s) == 1 or s[1] in {".", ")", ":", " "}):
            return head
        if enc == ENCODING_LETTER:
            raise AnswerEncodingError(f"expected letter A–E, got {raw!r}")
        # fall through to parse as int string under index encodings
        try:
            n = int(s)
        except ValueError as e:
            raise AnswerEncodingError(f"cannot map answer={raw!r} under encoding={enc}") from e
    elif isinstance(raw, bool):
        raise AnswerEncodingError(f"boolean answer not allowed: {raw!r}")
    elif isinstance(raw, (int, float)) and float(raw).is_integer():
        n = int(raw)
    else:
        try:
            n = int(str(raw).strip())
        except (TypeError, ValueError) as e:
            raise AnswerEncodingError(f"cannot map answer={raw!r} under encoding={enc}") from e

    if enc == ENCODING_LETTER:
        raise AnswerEncodingError(
            f"answer_encoding=letter rejects numeric label {n!r}; "
            "use zero_based_index or one_based_index"
        )
    if enc == ENCODING_ZERO:
        if n not in _ZERO_BASED:
            raise AnswerEncodingError(f"zero_based_index out of range: {n}")
        return _ZERO_BASED[n]
    # one_based_index
    if n not in _ONE_BASED:
        raise AnswerEncodingError(f"one_based_index out of range: {n}")
    return _ONE_BASED[n]
