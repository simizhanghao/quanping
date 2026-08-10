"""P3-C native Indonesian capability adapters + multi-capability matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linguaeval.adapters.dataset.native_mc import copal_row_to_sample, indommlu_row_to_sample
from linguaeval.adapters.dataset.registry import list_adapters
from linguaeval.core.language_matrix_runner import run_offline_language_matrix

ROOT = Path(__file__).resolve().parents[1]


def test_native_adapters_registered():
    ids = list_adapters()
    assert "indommlu_jsonl" in ids
    assert "copal_jsonl" in ids


def test_indommlu_native_provenance():
    s = indommlu_row_to_sample(
        {
            "id": "t1",
            "question": "Q?",
            "choices": {"A": "a", "B": "b", "C": "c", "D": "d"},
            "answer": "B",
        },
        language="ind",
        benchmark_id="toy",
        capability="local_knowledge",
    )
    assert s.gold["answer"] == "B"
    assert s.meta["provenance"]["native_authored"] is True
    assert s.meta["provenance"]["translation"] == "native"


def test_copal_binary():
    s = copal_row_to_sample(
        {"id": "x", "premise": "p", "choice1": "a", "choice2": "b", "label": 1},
        language="ind",
        benchmark_id="toy",
        capability="cultural_reasoning",
    )
    assert s.gold["answer"] == "B"
    assert s.meta["provenance"]["culture_sensitive"] is True


def test_capability_matrix_three_pillars():
    out = run_offline_language_matrix(
        ROOT / "configs/examples/24_language_capability_ind_native_toy.yaml"
    )
    m = json.loads((out / "language_metrics.json").read_text(encoding="utf-8"))
    r = json.loads((out / "language_regression.json").read_text(encoding="utf-8"))
    caps = set(m["by_capability"])
    assert caps == {"reading_comprehension", "local_knowledge", "cultural_reasoning"}

    reading = m["by_capability"]["reading_comprehension"]["by_language"]["ind"]
    knowledge = m["by_capability"]["local_knowledge"]["by_language"]["ind"]
    culture = m["by_capability"]["cultural_reasoning"]["by_language"]["ind"]
    assert reading["native_authored"] is False  # parallel belebele
    assert knowledge["native_authored"] is True
    assert culture["native_authored"] is True
    assert knowledge["accuracy"] == pytest.approx(1.0)
    assert culture["accuracy"] == pytest.approx(1.0)

    assert r["by_capability"]["local_knowledge"]["by_language"]["ind"]["delta_accuracy"] == pytest.approx(
        0.5
    )
    assert r["by_capability"]["cultural_reasoning"]["by_language"]["ind"][
        "delta_accuracy"
    ] == pytest.approx(0.25)
    assert r["by_capability"]["reading_comprehension"]["by_language"]["ind"][
        "delta_accuracy"
    ] == pytest.approx(0.25)
