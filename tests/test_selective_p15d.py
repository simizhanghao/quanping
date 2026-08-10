"""P1.5-D: selective prediction / Risk-Coverage — non-N2S primary acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linguaeval.confidence.selective import (
    STATUS_AVAILABLE,
    STATUS_NOT_AVAILABLE,
    SelectiveRow,
    aurc_from_curve,
    compute_selective_metrics,
    coverage_at_risk,
    risk_at_coverage,
    risk_coverage_curve,
)
from linguaeval.core.schema import (
    ConfidenceRecord,
    SampleInput,
    SampleRecord,
    SelectiveSpec,
)
from linguaeval.core.selective_runner import run_offline_selective

ROOT = Path(__file__).resolve().parents[1]


def test_perfect_ranking_zero_risk_until_errors():
    rows = [
        SelectiveRow("c1", 0.99, True, "test"),
        SelectiveRow("c2", 0.98, True, "test"),
        SelectiveRow("c3", 0.97, True, "test"),
        SelectiveRow("e1", 0.10, False, "test"),
        SelectiveRow("e2", 0.05, False, "test"),
    ]
    curve = risk_coverage_curve(rows)
    assert curve[0]["risk"] == 0.0
    assert curve[2]["risk"] == 0.0
    assert curve[2]["coverage"] == pytest.approx(0.6)
    assert curve[3]["risk"] == pytest.approx(0.25)
    assert coverage_at_risk(curve, 0.0) == pytest.approx(0.6)
    assert risk_at_coverage(curve, 0.6) == 0.0


def test_aurc_positive():
    rows = [
        SelectiveRow(f"s{i}", 1.0 - 0.1 * i, i >= 3, "test") for i in range(5)
    ]
    # mixed — AURC defined and in [0,1]
    curve = risk_coverage_curve(rows)
    a = aurc_from_curve(curve)
    assert a is not None
    assert 0.0 <= a <= 1.0


def test_missing_confidence_not_available():
    samples = [
        SampleRecord(
            sample_id="s1",
            gold={"label": "fraud"},
            input=SampleInput(text="x"),
            meta={"split_role": "test"},
        )
    ]
    records = [
        ConfidenceRecord(
            sample_id="s1",
            target="label",
            status=STATUS_NOT_AVAILABLE,
            reason="confidence_source_unavailable",
            gold="fraud",
        )
    ]
    out = compute_selective_metrics(
        records,
        samples,
        SelectiveSpec(target="label", evaluate_on="test"),
    )
    assert out["status"] == STATUS_NOT_AVAILABLE


def test_kernel_no_business_tokens():
    text = (ROOT / "src/linguaeval/confidence/selective.py").read_text(encoding="utf-8")
    for banned in ("n2s", "routing_skill", "banking", "BCA", "bca"):
        assert banned not in text


def test_toy_selective_available():
    out = run_offline_selective(ROOT / "configs/examples/13_selective_toy_binary.yaml")
    m = json.loads((out / "selective_metrics.json").read_text(encoding="utf-8"))
    assert m["status"] == STATUS_AVAILABLE
    assert m["n_evaluate"] == 64
    assert m["aurc"] is not None
    assert 0.0 <= m["aurc"] <= 1.0
    assert "0.8" in m["risk_at_coverage"]
    assert "0.1" in m["coverage_at_risk"]
    curve = json.loads((out / "risk_coverage_curve.json").read_text(encoding="utf-8"))
    assert curve["n"] == 64


def test_n2s_selective_not_available():
    path = Path(
        "/data/hanchengcheng/hcc_1/LlamaFactory/tests/yewupingce/n2s_test/"
        "n2s_result/qwen3_4b_test3.json"
    )
    if not path.is_file():
        pytest.skip("N2S fixture missing")
    out = run_offline_selective(
        ROOT / "configs/examples/14_selective_n2s_unavailable.yaml"
    )
    m = json.loads((out / "selective_metrics.json").read_text(encoding="utf-8"))
    assert m["status"] == STATUS_NOT_AVAILABLE
