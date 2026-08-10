"""P2-E D8 consistency + D6 context ablation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linguaeval.consistency.aggregate import aggregate_consistency, build_consistency_records
from linguaeval.context.aggregate import aggregate_context_ablation, build_context_ablation_records
from linguaeval.core.consistency_runner import run_offline_consistency
from linguaeval.core.context_runner import run_offline_context
from linguaeval.core.schema import (
    FormatStatus,
    PredictionRecord,
    SampleInput,
    SampleRecord,
    TaskSpec,
    TargetSpec,
)

ROOT = Path(__file__).resolve().parents[1]


def _task() -> TaskSpec:
    return TaskSpec(
        name="t",
        task_type="classification",
        targets=[TargetSpec(name="intent_class", type="multiclass", path="$.intent_class")],
    )


def test_consistency_pairwise_and_all_agree():
    samples = [
        SampleRecord(sample_id="a", gold={"intent_class": "refund"}, input=SampleInput(text="x")),
        SampleRecord(sample_id="b", gold={"intent_class": "shipping"}, input=SampleInput(text="y")),
    ]
    preds = [
        PredictionRecord("a", "m", parsed={"intent_class": "refund"}, format=FormatStatus()),
        PredictionRecord("a", "m", parsed={"intent_class": "refund"}, format=FormatStatus()),
        PredictionRecord("b", "m", parsed={"intent_class": "shipping"}, format=FormatStatus()),
        PredictionRecord("b", "m", parsed={"intent_class": "account"}, format=FormatStatus()),
    ]
    rows = build_consistency_records(samples, preds, _task(), target="intent_class")
    metrics = aggregate_consistency(rows)
    assert metrics["status"] == "AVAILABLE"
    assert metrics["by_target"]["intent_class"]["all_agree_rate"] == pytest.approx(0.5)
    assert metrics["by_target"]["intent_class"]["pairwise_agreement_rate"] == pytest.approx(0.5)


def test_consistency_offline_toy():
    out = run_offline_consistency(ROOT / "configs/examples/19_consistency_toy_intent.yaml")
    m = json.loads((out / "consistency_metrics.json").read_text(encoding="utf-8"))
    assert m["status"] == "AVAILABLE"
    b = m["by_target"]["intent_class"]
    assert b["n"] == 4
    assert b["all_agree_rate"] == pytest.approx(0.75)
    assert b["pairwise_agreement_rate"] == pytest.approx(0.75)
    assert b["majority_accuracy"] == pytest.approx(1.0)


def test_context_ablation_gain():
    samples = [
        SampleRecord(
            sample_id="d1",
            gold={"intent_class": "refund"},
            input=SampleInput(text="x"),
            conversation={"dialogue_id": "g", "turn_id": 1},
        )
    ]
    without = [
        PredictionRecord("d1", "m", parsed={"intent_class": "shipping"}, format=FormatStatus())
    ]
    with_ctx = [
        PredictionRecord("d1", "m", parsed={"intent_class": "refund"}, format=FormatStatus())
    ]
    rows = build_context_ablation_records(
        samples=samples,
        without_preds=without,
        with_preds=with_ctx,
        task=_task(),
        target="intent_class",
    )
    metrics = aggregate_context_ablation(rows)
    b = metrics["by_target"]["intent_class"]
    assert b["delta_accuracy"] == pytest.approx(1.0)
    assert b["context_gain_rate"] == pytest.approx(1.0)
    assert b["prediction_flip_rate"] == pytest.approx(1.0)


def test_context_offline_toy():
    out = run_offline_context(ROOT / "configs/examples/20_context_toy_intent.yaml")
    m = json.loads((out / "context_metrics.json").read_text(encoding="utf-8"))
    assert m["status"] == "AVAILABLE"
    b = m["by_target"]["intent_class"]
    assert b["n"] == 8
    assert b["accuracy_without_context"] == pytest.approx(0.75)
    assert b["accuracy_with_context"] == pytest.approx(1.0)
    assert b["delta_accuracy"] == pytest.approx(0.25)
    assert b["context_gain_rate"] == pytest.approx(0.25)
    assert b["prediction_flip_rate"] == pytest.approx(0.25)
    gains = [
        json.loads(line)
        for line in (out / "context_gain_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(gains) == 2
