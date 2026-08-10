"""P1.5-B: calibration metrics on ConfidenceRecords — non-N2S primary acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linguaeval.confidence.extract import STATUS_AVAILABLE
from linguaeval.confidence.metrics import (
    STATUS_NOT_AVAILABLE,
    compute_calibration_metrics,
    expected_calibration_error,
    roc_auc_binary,
)
from linguaeval.core.confidence_runner import run_offline_confidence
from linguaeval.core.schema import ConfidenceRecord

ROOT = Path(__file__).resolve().parents[1]


def test_ece_perfectly_calibrated_zero():
    # ten samples at conf=0.7 with 7 correct → bin acc=conf → ECE=0
    conf = [0.7] * 10
    correct = [True] * 7 + [False] * 3
    ece, _ = expected_calibration_error(conf, correct, n_bins=10)
    assert ece < 1e-9


def test_roc_auc_perfect():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [False, False, True, True]
    assert roc_auc_binary(scores, labels) == pytest.approx(1.0)


def test_metrics_empty_not_available():
    out = compute_calibration_metrics([])
    assert out["status"] == STATUS_NOT_AVAILABLE
    assert out["metrics"]["ece"]["status"] == STATUS_NOT_AVAILABLE


def test_metrics_insufficient_support_small_n():
    recs = [
        ConfidenceRecord(
            sample_id=f"s{i}",
            target="intent_class",
            status=STATUS_AVAILABLE,
            gold="a" if i % 2 == 0 else "b",
            prediction="a" if i % 2 == 0 else "b",
            class_scores={"a": 0.8, "b": 0.2} if i % 2 == 0 else {"a": 0.2, "b": 0.8},
            confidence=0.8,
        )
        for i in range(4)
    ]
    out = compute_calibration_metrics(recs, min_samples=10)
    assert out["status"] == "INSUFFICIENT_SUPPORT"
    assert out["metrics"]["ece"]["status"] == "INSUFFICIENT_SUPPORT"
    assert out["metrics"]["brier"]["status"] == "AVAILABLE"
    assert out["metrics"]["nll"]["status"] == "AVAILABLE"


def test_kernel_metrics_has_no_business_tokens():
    text = (ROOT / "src/linguaeval/confidence/metrics.py").read_text(encoding="utf-8")
    for banned in ("n2s", "routing_skill", "banking", "BCA", "bca"):
        assert banned not in text


def test_toy_calibration_metrics_available():
    out = run_offline_confidence(ROOT / "configs/examples/07_confidence_toy_multiclass.yaml")
    cal = json.loads((out / "calibration_metrics.json").read_text(encoding="utf-8"))
    assert cal["status"] == "AVAILABLE"
    assert cal["n_usable"] == 16
    for key in ("ece", "brier", "nll", "auroc_ovr_macro", "accuracy"):
        assert cal["metrics"][key]["status"] == "AVAILABLE"
        assert cal["metrics"][key]["value"] is not None
    assert 0.0 <= cal["metrics"]["ece"]["value"] <= 1.0
    assert cal["metrics"]["brier"]["value"] >= 0.0
    assert cal["metrics"]["nll"]["value"] >= 0.0
    assert 0.0 <= cal["metrics"]["auroc_ovr_macro"]["value"] <= 1.0


def test_n2s_calibration_not_available():
    path = Path(
        "/data/hanchengcheng/hcc_1/LlamaFactory/tests/yewupingce/n2s_test/"
        "n2s_result/qwen3_4b_test3.json"
    )
    if not path.is_file():
        pytest.skip("N2S fixture missing")
    out = run_offline_confidence(
        ROOT / "configs/examples/08_confidence_n2s_unavailable.yaml"
    )
    cal = json.loads((out / "calibration_metrics.json").read_text(encoding="utf-8"))
    assert cal["status"] == "NOT_AVAILABLE"
    assert cal["n_usable"] == 0
    assert cal["metrics"]["ece"]["status"] == "NOT_AVAILABLE"
    assert cal["metrics"]["ece"]["reason"] == "confidence_source_unavailable"
