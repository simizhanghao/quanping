"""P1-D: protocol, comparability, metric applicability, gate support policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linguaeval.compare.applicability import (
    STATUS_NOT_APPLICABLE,
    metrics_with_applicability,
)
from linguaeval.compare.bootstrap import PairRow
from linguaeval.compare.gates import evaluate_gates
from linguaeval.compare.protocol import (
    ComparisonProtocolError,
    evaluate_comparability,
    validate_comparison_protocol,
)
from linguaeval.core.compare_runner import run_offline_compare

ROOT = Path(__file__).resolve().parents[1]


def test_metric_f1_not_applicable_without_positives():
    rows = [
        PairRow(
            sample_id=f"s{i}",
            unit_id="d0",
            gold=False,
            baseline_pred=False,
            candidate_pred=True if i < 2 else False,
            transition="gain" if i < 2 else "stable_correct",
        )
        for i in range(5)
    ]
    out = metrics_with_applicability(
        rows,
        list(range(5)),
        side="candidate",
        target_type="binary",
        metric_names=["f1", "recall", "accuracy", "specificity", "false_positive_rate"],
    )
    assert out["f1"]["status"] == STATUS_NOT_APPLICABLE
    assert out["f1"]["reason"] == "positive_support=0"
    assert out["recall"]["status"] == STATUS_NOT_APPLICABLE
    assert out["accuracy"]["status"] == "APPLICABLE"
    assert out["specificity"]["status"] == "APPLICABLE"
    assert out["false_positive_rate"]["status"] == "APPLICABLE"


def test_comparability_semantic_vs_efficiency():
    cfg = {
        "semantic": {
            "prompt_protocol": "p1",
            "context_protocol": "c1",
            "scoring_protocol": "s1",
        },
        "baseline": {"backend_family": "vllm", "decoding": {"temperature": 0.0}},
        "candidate": {"backend_family": "transformers", "decoding": {"temperature": 0.0}},
    }
    out = evaluate_comparability(
        cfg,
        baseline_side=cfg["baseline"],
        candidate_side=cfg["candidate"],
    )
    assert out["semantic_comparable"] is True
    assert out["efficiency_comparable"] is False
    assert "backend_family" in out["efficiency_mismatches"]


def test_golden_pair_rejects_unknown_baseline():
    with pytest.raises(ComparisonProtocolError) as ei:
        validate_comparison_protocol(
            {
                "protocol_id": "x",
                "allowed_pairs": [
                    {
                        "baseline_path_suffix": "good_base.json",
                        "candidate_path_suffix": "good_sft.json",
                    }
                ],
            },
            baseline_path="/data/foo/bad_base.json",
            candidate_path="/data/foo/good_sft.json",
            dataset_fingerprint="abc",
            task_spec_hash=None,
            output_spec_hash=None,
            metric_spec_hash=None,
            n_aligned=10,
            comparability={"semantic_comparable": True, "efficiency_comparable": False},
        )
    assert ei.value.code == "NOT_COMPARABLE"


def test_gate_insufficient_support_not_fail():
    ctx = {
        "support": {"n_samples": 32, "n_units": 32},
        "statistics": {"metrics": {"accuracy": {"delta": {"ci_low": -0.1}}}},
        "candidate_business": {"primary": {"value": 0.9}},
    }
    out = evaluate_gates(
        ctx,
        [
            {
                "id": "ci",
                "path": "statistics.metrics.accuracy.delta.ci_low",
                "op": ">=",
                "value": 0,
                "requirements": {"min_samples": 500},
            },
            {
                "id": "acc",
                "path": "candidate_business.primary.value",
                "op": ">=",
                "value": 0.5,
            },
        ],
    )
    assert out["gates"][0]["status"] == "INSUFFICIENT_SUPPORT"
    assert out["gates"][1]["status"] == "PASS"
    assert out["status"] == "INSUFFICIENT_SUPPORT"
    assert out["n_fail"] == 0


def test_toy_compare_p1d_gate_and_protocol_optional():
    out = run_offline_compare(ROOT / "configs/examples/05_compare_base_sft_toy.yaml")
    metrics = json.loads((out / "comparison_metrics.json").read_text(encoding="utf-8"))
    gate = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    assert metrics["comparability"]["semantic_comparable"] is True
    assert metrics["comparability"]["efficiency_comparable"] is True
    assert gate["status"] == "INSUFFICIENT_SUPPORT"
    by_id = {g["id"]: g for g in gate["gates"]}
    assert by_id["delta_accuracy_ci_lower"]["status"] == "INSUFFICIENT_SUPPORT"
    assert by_id["candidate_accuracy_min"]["status"] == "PASS"
