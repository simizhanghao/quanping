"""P2-D robustness compare — shared variant_fingerprint gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linguaeval.core.robustness_compare_runner import run_offline_robustness_compare
from linguaeval.core.schema import SampleInput, VariantRecord
from linguaeval.robustness.compare import (
    VariantFingerprintError,
    classify_model_robust_transition,
    compare_robustness_metrics,
    require_shared_variant_fingerprint,
)
from linguaeval.robustness.generate import variant_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def _one_variant() -> list[VariantRecord]:
    return [
        VariantRecord(
            variant_id="a",
            parent_sample_id="s1",
            perturbation_id="case_lower",
            input=SampleInput(text="x"),
            semantic_validity="VERIFIED",
        )
    ]


def test_classify_model_robust_transition():
    assert classify_model_robust_transition(True, True) == "stable_robust"
    assert classify_model_robust_transition(False, True) == "robustness_gain"
    assert classify_model_robust_transition(True, False) == "robustness_regression"
    assert classify_model_robust_transition(False, False) == "both_fragile"


def test_fingerprint_gate_rejects_expected_mismatch():
    v = _one_variant()
    fp = variant_fingerprint(v)
    with pytest.raises(VariantFingerprintError) as ei:
        require_shared_variant_fingerprint(variants=v, expected_fingerprint="deadbeef")
    assert ei.value.reason == "expected_mismatch"
    assert require_shared_variant_fingerprint(variants=v, expected_fingerprint=fp) == fp


def test_side_fingerprint_mismatch():
    with pytest.raises(VariantFingerprintError) as ei:
        require_shared_variant_fingerprint(
            variants=_one_variant(),
            baseline_fingerprint="aaa",
            candidate_fingerprint="bbb",
        )
    assert ei.value.reason == "side_mismatch"


def test_metric_delta_flip_rate():
    baseline = {"status": "AVAILABLE", "by_target": {"intent_class": {"flip_rate": 0.25}}}
    candidate = {"status": "AVAILABLE", "by_target": {"intent_class": {"flip_rate": 0.0}}}
    cmp = compare_robustness_metrics(baseline, candidate)
    assert cmp["by_target"]["intent_class"]["delta"]["flip_rate"] == pytest.approx(-0.25)


def test_robustness_compare_offline_toy():
    out = run_offline_robustness_compare(ROOT / "configs/examples/18_robustness_compare_toy.yaml")
    summary = json.loads((out / "robustness_compare_metrics.json").read_text(encoding="utf-8"))
    assert summary["status"] == "AVAILABLE"
    assert summary["variant_fingerprint"]
    assert summary["n_transition_eligible"] == 8
    trans = summary["transitions"]
    assert trans["stable_robust"] == 6
    assert trans["robustness_gain"] == 2
    assert trans["robustness_regression"] == 0
    assert trans["both_fragile"] == 0
    assert sum(trans.values()) == 8
    delta = summary["metrics_compare"]["by_target"]["intent_class"]["delta"]
    assert delta["flip_rate"] == pytest.approx(-0.25)
    gains = [
        json.loads(line)
        for line in (out / "robustness_gain_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(gains) == 2
