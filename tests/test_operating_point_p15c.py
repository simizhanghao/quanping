"""P1.5-C: operating-point / threshold selection — non-N2S primary acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linguaeval.confidence.operating_point import (
    STATUS_AVAILABLE,
    STATUS_NO_FEASIBLE,
    STATUS_NOT_AVAILABLE,
    STATUS_TEST_LEAKAGE,
    OperatingPointError,
    ScoredBinaryRow,
    metrics_at_threshold,
    select_operating_point,
    sweep_curve,
)
from linguaeval.core.operating_point_runner import run_offline_operating_point
from linguaeval.core.schema import (
    ConfidenceRecord,
    OperatingPointSpec,
    SampleInput,
    SampleRecord,
)

ROOT = Path(__file__).resolve().parents[1]


def _rec(sid: str, gold: str, score_pos: float, pos: str = "fraud") -> ConfidenceRecord:
    other = 1.0 - score_pos
    neg = "legit" if pos == "fraud" else "other"
    return ConfidenceRecord(
        sample_id=sid,
        target="label",
        status="AVAILABLE",
        gold=gold,
        prediction=pos if score_pos >= 0.5 else neg,
        class_scores={pos: score_pos, neg: other},
        confidence=score_pos,
    )


def _sample(sid: str, role: str, gold: str) -> SampleRecord:
    return SampleRecord(
        sample_id=sid,
        gold={"label": gold},
        input=SampleInput(text=sid),
        meta={"split_role": role},
    )


def test_constraint_precision_floor_enforced():
    rows = [
        ScoredBinaryRow("a", 0.9, True, "validation"),
        ScoredBinaryRow("b", 0.8, True, "validation"),
        ScoredBinaryRow("c", 0.7, False, "validation"),
        ScoredBinaryRow("d", 0.6, True, "validation"),
        ScoredBinaryRow("e", 0.4, False, "validation"),
    ]
    curve = sweep_curve(rows)
    for m in curve:
        if m["precision"] + 1e-15 < 0.9:
            # low-precision points exist
            pass
    # at high threshold only one pos → P=1
    m = metrics_at_threshold(rows, 0.9)
    assert m["precision"] >= 0.9


def test_no_feasible_operating_point():
    # Every threshold that catches a positive also catches a higher-scored negative.
    records = [
        _rec("v1", "fraud", 0.80),
        _rec("v2", "legit", 0.90),
        _rec("v3", "fraud", 0.70),
        _rec("v4", "legit", 0.85),
    ]
    samples = [
        _sample("v1", "validation", "fraud"),
        _sample("v2", "validation", "legit"),
        _sample("v3", "validation", "fraud"),
        _sample("v4", "validation", "legit"),
    ]
    spec = OperatingPointSpec(
        target="label",
        positive_class="fraud",
        optimize_on="validation",
        evaluate_on="test",
        mode="max_recall_at_precision",
        precision_min=0.99,
    )
    out = select_operating_point(records, samples, spec)
    assert out["status"] == STATUS_NO_FEASIBLE
    assert out["selected"] is None


def test_test_leakage_raises():
    spec = OperatingPointSpec(
        target="label",
        positive_class="fraud",
        optimize_on="test",
        mode="best_f1",
    )
    with pytest.raises(OperatingPointError) as ei:
        select_operating_point([], [], spec)
    assert ei.value.reason == STATUS_TEST_LEAKAGE


def test_missing_confidence_not_available():
    samples = [_sample("s1", "validation", "fraud")]
    records = [
        ConfidenceRecord(
            sample_id="s1",
            target="label",
            status=STATUS_NOT_AVAILABLE,
            reason="confidence_source_unavailable",
            gold="fraud",
        )
    ]
    spec = OperatingPointSpec(
        target="label",
        positive_class="fraud",
        optimize_on="validation",
        mode="best_f1",
    )
    out = select_operating_point(records, samples, spec)
    assert out["status"] == STATUS_NOT_AVAILABLE


def test_target_rename_only_config():
    """Same scores; changing target name must not require kernel edits."""
    records = [_rec("v1", "fraud", 0.9), _rec("v2", "legit", 0.1)]
    samples = [
        _sample("v1", "validation", "fraud"),
        _sample("v2", "validation", "legit"),
    ]
    a = select_operating_point(
        records,
        samples,
        OperatingPointSpec(
            target="fraud_flag",
            positive_class="fraud",
            optimize_on="validation",
            mode="best_f1",
        ),
    )
    b = select_operating_point(
        records,
        samples,
        OperatingPointSpec(
            target="escalation",
            positive_class="fraud",
            optimize_on="validation",
            mode="best_f1",
        ),
    )
    assert a["status"] == STATUS_AVAILABLE
    assert b["status"] == STATUS_AVAILABLE
    assert a["selected"]["threshold"] == b["selected"]["threshold"]
    assert a["target"] == "fraud_flag"
    assert b["target"] == "escalation"


def test_ovr_positive_class_uses_that_score():
    rec = ConfidenceRecord(
        sample_id="x",
        target="intent",
        status="AVAILABLE",
        gold="refund",
        prediction="refund",
        class_scores={"refund": 0.7, "shipping": 0.2, "account": 0.1},
        confidence=0.7,
    )
    sample = SampleRecord(
        sample_id="x",
        gold={"intent": "refund"},
        input=SampleInput(text="x"),
        meta={"split_role": "validation"},
    )
    out = select_operating_point(
        [rec],
        [sample],
        OperatingPointSpec(
            target="intent",
            positive_class="refund",
            optimize_on="validation",
            evaluate_on="validation",
            mode="best_f1",
        ),
    )
    assert out["status"] == STATUS_AVAILABLE
    assert out["selected"]["threshold"] is not None


def test_toy_known_threshold_max_recall_at_p90():
    out = run_offline_operating_point(ROOT / "configs/examples/09_operating_point_toy_binary.yaml")
    op = json.loads((out / "operating_points.json").read_text(encoding="utf-8"))
    assert op["status"] == STATUS_AVAILABLE
    assert op["mode"] == "max_recall_at_precision"
    assert abs(float(op["selected"]["threshold"]) - 0.48) < 1e-9
    assert float(op["selected"]["precision"]) + 1e-15 >= 0.9
    assert float(op["selected"]["recall"]) == pytest.approx(1.0)
    assert "precision" in op["test_evaluation"]


def test_cli_config_test_leakage_status():
    with pytest.raises(OperatingPointError) as ei:
        run_offline_operating_point(ROOT / "configs/examples/10_operating_point_test_leakage.yaml")
    assert ei.value.reason == STATUS_TEST_LEAKAGE
    # artifact still written
    p = ROOT / "results/10_operating_point_test_leakage/operating_points.json"
    assert p.is_file()
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["status"] == STATUS_TEST_LEAKAGE


def test_n2s_operating_point_not_available():
    path = Path(
        "/data/hanchengcheng/hcc_1/LlamaFactory/tests/yewupingce/n2s_test/"
        "n2s_result/qwen3_4b_test3.json"
    )
    if not path.is_file():
        pytest.skip("N2S fixture missing")
    out = run_offline_operating_point(
        ROOT / "configs/examples/11_operating_point_n2s_unavailable.yaml"
    )
    op = json.loads((out / "operating_points.json").read_text(encoding="utf-8"))
    assert op["status"] == STATUS_NOT_AVAILABLE


def test_toy_ovr_smoke_available():
    out = run_offline_operating_point(ROOT / "configs/examples/12_operating_point_toy_ovr.yaml")
    op = json.loads((out / "operating_points.json").read_text(encoding="utf-8"))
    assert op["status"] == STATUS_AVAILABLE
    assert op["positive_class"] == "refund"
    assert op["selected"]["threshold"] is not None


def test_kernel_has_no_business_tokens():
    text = (ROOT / "src/linguaeval/confidence/operating_point.py").read_text(encoding="utf-8")
    for banned in ("n2s", "routing_skill", "banking", "BCA", "bca"):
        assert banned not in text
