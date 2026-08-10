"""P3-B Belebele adapter + language matrix (non-N2S)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linguaeval.adapters.dataset.belebele import load_belebele_samples_jsonl, row_to_sample
from linguaeval.adapters.dataset.registry import get_adapter, list_adapters
from linguaeval.core.language_matrix_runner import run_offline_language_matrix
from linguaeval.core.runner import run_offline_score

ROOT = Path(__file__).resolve().parents[1]


def test_belebele_adapter_registered():
    assert "belebele_jsonl" in list_adapters()


def test_row_to_sample_letter_and_language_meta():
    s = row_to_sample(
        {
            "link": "x1",
            "flores_passage": "p",
            "question": "q",
            "mc_answer1": "a",
            "mc_answer2": "b",
            "mc_answer3": "c",
            "mc_answer4": "d",
            "correct_answer_num": "2",
        },
        language="ind",
        benchmark_id="toy",
    )
    assert s.gold["answer"] == "B"
    assert s.meta["language"] == "ind"
    assert "A." in (s.input.text or "")


def test_same_adapter_loads_ind_and_arb():
    ind = load_belebele_samples_jsonl(
        ROOT / "examples/toy_belebele/ind_Latn.jsonl",
        language="ind",
        benchmark_id="toy_belebele_ind",
    )
    arb = load_belebele_samples_jsonl(
        ROOT / "examples/toy_belebele/arb_Arab.jsonl",
        language="arb",
        benchmark_id="toy_belebele_arb",
    )
    assert len(ind) == len(arb) == 8
    assert {s.meta["link"] for s in ind} == {s.meta["link"] for s in arb}
    assert all(s.meta["language"] == "ind" for s in ind)
    assert all(s.meta["language"] == "arb" for s in arb)


def test_score_offline_via_belebele_adapter():
    out = run_offline_score(ROOT / "configs/examples/23_score_belebele_ind_toy.yaml")
    biz = json.loads((out / "business_metrics.json").read_text(encoding="utf-8"))
    assert biz["targets"]["answer"]["accuracy"] == pytest.approx(1.0)


def test_language_matrix_ind_gain_arb_regression():
    out = run_offline_language_matrix(
        ROOT / "configs/examples/22_language_matrix_belebele_toy.yaml"
    )
    metrics = json.loads((out / "language_metrics.json").read_text(encoding="utf-8"))
    reg = json.loads((out / "language_regression.json").read_text(encoding="utf-8"))
    assert metrics["capability"] == "reading_comprehension"
    assert set(metrics["by_language"]) == {"ind", "arb"}
    # candidate primary metric values when baseline/candidate present
    assert metrics["by_language"]["ind"]["value"] == pytest.approx(1.0)
    assert metrics["by_language"]["arb"]["value"] == pytest.approx(0.5)
    assert metrics["by_language"]["ind"]["metric_path"] == "targets.answer.accuracy"

    ind = reg["by_language"]["ind"]
    arb = reg["by_language"]["arb"]
    assert ind["metric_path"] == "targets.answer.accuracy"
    assert ind["baseline_value"] == pytest.approx(0.75)
    assert ind["candidate_value"] == pytest.approx(1.0)
    assert ind["delta"] == pytest.approx(0.25)
    assert arb["baseline_value"] == pytest.approx(0.625)
    assert arb["candidate_value"] == pytest.approx(0.5)
    assert arb["delta"] == pytest.approx(-0.125)
    assert "delta_accuracy" not in ind

    # adapter used (no n2s)
    adapter = get_adapter("belebele_jsonl")
    samples, preds = adapter(
        {
            "language": "ind",
            "samples": str(ROOT / "examples/toy_belebele/ind_Latn.jsonl"),
            "predictions": str(ROOT / "examples/toy_belebele/predictions_ind_base.jsonl"),
            "benchmark_id": "toy_belebele_ind",
        },
        ROOT / "configs/examples",
        {},
    )
    assert len(samples) == 8 and len(preds) == 8
