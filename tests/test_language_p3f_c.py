"""P3-F-C: explicit answer_encoding; no silent 0/1-based guessing."""

from __future__ import annotations

import pytest

from linguaeval.adapters.dataset.answer_encoding import AnswerEncodingError, as_mc_letter
from linguaeval.adapters.dataset.native_mc import copal_row_to_sample, indommlu_row_to_sample


def test_zero_based_vs_one_based_not_guessed():
    assert as_mc_letter(1, encoding="zero_based_index") == "B"
    assert as_mc_letter(1, encoding="one_based_index") == "A"
    with pytest.raises(AnswerEncodingError, match="letter rejects numeric"):
        as_mc_letter(1, encoding="letter")


def test_missing_encoding_raises():
    with pytest.raises(AnswerEncodingError, match="required"):
        as_mc_letter("A", encoding="")


def test_indommlu_numeric_needs_encoding():
    with pytest.raises(AnswerEncodingError):
        indommlu_row_to_sample(
            {
                "id": "t1",
                "question": "Q?",
                "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
                "answer": 1,
            },
            language="ind",
            benchmark_id="toy",
            capability="local_knowledge",
            answer_encoding="letter",
        )
    s = indommlu_row_to_sample(
        {
            "id": "t1",
            "question": "Q?",
            "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "answer": 1,
        },
        language="ind",
        benchmark_id="toy",
        capability="local_knowledge",
        answer_encoding="zero_based_index",
    )
    assert s.gold["answer"] == "B"


def test_copal_zero_based_label():
    s = copal_row_to_sample(
        {"id": "x", "premise": "p", "choice1": "a", "choice2": "b", "label": 0},
        language="ind",
        benchmark_id="toy",
        capability="cultural_reasoning",
        answer_encoding="zero_based_index",
    )
    assert s.gold["answer"] == "A"
    # Wrong encoding would shift labels silently before; now one_based maps 0 → error
    with pytest.raises(AnswerEncodingError):
        copal_row_to_sample(
            {"id": "x", "premise": "p", "choice1": "a", "choice2": "b", "label": 0},
            language="ind",
            benchmark_id="toy",
            capability="cultural_reasoning",
            answer_encoding="one_based_index",
        )
