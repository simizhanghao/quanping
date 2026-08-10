"""P1.5-A: generic ConfidenceSpec / extractor — non-N2S primary acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linguaeval.confidence.extract import (
    STATUS_AVAILABLE,
    STATUS_NOT_AVAILABLE,
    extract_one,
)
from linguaeval.core.confidence_runner import run_offline_confidence
from linguaeval.core.schema import (
    ConfidenceSpec,
    ConfidenceSourceSpec,
    FormatStatus,
    PredictionRecord,
    SampleInput,
    SampleRecord,
    TaskSpec,
    TargetSpec,
)

ROOT = Path(__file__).resolve().parents[1]


def _toy_task() -> TaskSpec:
    return TaskSpec(
        name="t",
        task_type="classification",
        targets=[
            TargetSpec(
                name="intent_class",
                type="multiclass",
                path="$.intent_class",
                labels=["refund", "shipping", "account"],
            )
        ],
    )


def test_extract_probabilities_available():
    sample = SampleRecord(
        sample_id="x1",
        gold={"intent_class": "refund"},
        input=SampleInput(text="x"),
    )
    pred = PredictionRecord(
        sample_id="x1",
        model_id="m",
        parsed={"intent_class": "refund"},
        scores={"intent_class": {"refund": 0.7, "shipping": 0.2, "account": 0.1}},
        format=FormatStatus(parse_ok=True, schema_ok=True),
    )
    spec = ConfidenceSpec(
        target="intent_class",
        source=ConfidenceSourceSpec(type="probabilities", path="scores.intent_class"),
    )
    rec = extract_one(sample, pred, spec=spec, task=_toy_task())
    assert rec.status == STATUS_AVAILABLE
    assert abs(rec.confidence - 0.7) < 1e-9
    assert rec.class_scores is not None
    assert set(rec.class_scores) == {"refund", "shipping", "account"}


def test_extract_missing_scores_not_available():
    sample = SampleRecord(
        sample_id="x1",
        gold={"intent_class": "refund"},
        input=SampleInput(text="x"),
    )
    pred = PredictionRecord(
        sample_id="x1",
        model_id="m",
        parsed={"intent_class": "refund"},
        scores={},
    )
    spec = ConfidenceSpec(
        target="intent_class",
        source=ConfidenceSourceSpec(type="probabilities", path="scores.intent_class"),
    )
    rec = extract_one(sample, pred, spec=spec, task=_toy_task())
    assert rec.status == STATUS_NOT_AVAILABLE
    assert rec.reason == "confidence_source_unavailable"


def test_binary_is_two_class_multiclass():
    """Binary uses the same contract as multiclass (K=2)."""
    task = TaskSpec(
        name="bin",
        task_type="classification",
        targets=[
            TargetSpec(
                name="toxic",
                type="binary",
                path="$.toxic",
                labels=["false", "true"],
            )
        ],
    )
    sample = SampleRecord(
        sample_id="b1",
        gold={"toxic": "true"},
        input=SampleInput(text="x"),
    )
    pred = PredictionRecord(
        sample_id="b1",
        model_id="m",
        parsed={"toxic": "true"},
        scores={"toxic": {"false": 0.2, "true": 0.8}},
        format=FormatStatus(parse_ok=True, schema_ok=True),
    )
    spec = ConfidenceSpec(
        target="toxic",
        source=ConfidenceSourceSpec(type="probabilities", path="scores.toxic"),
    )
    rec = extract_one(sample, pred, spec=spec, task=task)
    assert rec.status == STATUS_AVAILABLE
    assert abs(rec.confidence - 0.8) < 1e-9
    assert rec.meta.get("n_classes") == 2


def test_logits_softmax_available():
    sample = SampleRecord(
        sample_id="x1",
        gold={"intent_class": "refund"},
        input=SampleInput(text="x"),
    )
    pred = PredictionRecord(
        sample_id="x1",
        model_id="m",
        parsed={"intent_class": "refund"},
        scores={"intent_class": {"refund": 2.0, "shipping": 0.0, "account": 0.0}},
        format=FormatStatus(parse_ok=True, schema_ok=True),
    )
    spec = ConfidenceSpec(
        target="intent_class",
        source=ConfidenceSourceSpec(type="logits", path="scores.intent_class"),
    )
    rec = extract_one(sample, pred, spec=spec, task=_toy_task())
    assert rec.status == STATUS_AVAILABLE
    assert rec.class_scores is not None
    assert abs(sum(rec.class_scores.values()) - 1.0) < 1e-9
    assert rec.confidence == rec.class_scores["refund"]


def test_kernel_source_has_no_n2s_branch_strings():
    # Guardrail: extractor module must not hardcode business tokens
    text = (ROOT / "src/linguaeval/confidence/extract.py").read_text(encoding="utf-8")
    for banned in ("n2s", "routing_skill", "banking", "BCA", "bca"):
        assert banned not in text


def test_toy_confidence_offline_available():
    out = run_offline_confidence(ROOT / "configs/examples/07_confidence_toy_multiclass.yaml")
    audit = json.loads((out / "confidence_audit.json").read_text(encoding="utf-8"))
    assert audit["n_records"] == 16
    assert audit["counts"]["AVAILABLE"] == 16
    assert audit["counts"]["NOT_AVAILABLE"] == 0
    line = (out / "confidence_records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    assert row["status"] == "AVAILABLE"
    assert row["confidence"] is not None


def test_n2s_free_gen_confidence_not_available():
    path = Path(
        "/data/hanchengcheng/hcc_1/LlamaFactory/tests/yewupingce/n2s_test/"
        "n2s_result/qwen3_4b_test3.json"
    )
    if not path.is_file():
        pytest.skip("N2S fixture missing")
    out = run_offline_confidence(
        ROOT / "configs/examples/08_confidence_n2s_unavailable.yaml"
    )
    audit = json.loads((out / "confidence_audit.json").read_text(encoding="utf-8"))
    assert audit["n_records"] > 0
    assert audit["counts"]["AVAILABLE"] == 0
    assert audit["counts"]["NOT_AVAILABLE"] == audit["n_records"]
    assert "confidence_source_unavailable" in audit["reason_counts"]
