"""P2-A: metamorphic contract / invariance offline — non-N2S primary acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linguaeval.core.robustness_runner import run_offline_robustness
from linguaeval.core.schema import (
    MetamorphicRelationSpec,
    PerturbationSpec,
    PredictionRecord,
    SampleInput,
    SampleRecord,
    TaskSpec,
    TargetSpec,
    VariantRecord,
)
from linguaeval.robustness.aggregate import aggregate_robustness, build_robustness_records
from linguaeval.robustness.registry import (
    apply_perturbation,
    get_perturbation,
    list_perturbations,
)
from linguaeval.robustness.relations import relation_satisfied

ROOT = Path(__file__).resolve().parents[1]


def test_registry_surface_apply_works():
    ids = list_perturbations()
    assert "case_lower" in ids
    spec = get_perturbation("case_lower")
    assert isinstance(spec, PerturbationSpec)
    out = apply_perturbation(SampleInput(text="Hi There"), spec)
    assert out.text == "hi there"


def test_invariance_on_targets_not_raw_strings():
    rel = MetamorphicRelationSpec(type="invariance", targets=["intent_class"])
    assert relation_satisfied(rel, clean_pred="refund", variant_pred="refund") is True
    assert relation_satisfied(rel, clean_pred="refund", variant_pred="shipping") is False


def test_invalid_variant_excluded_from_denominator():
    task = TaskSpec(
        name="t",
        task_type="classification",
        targets=[TargetSpec(name="intent_class", type="multiclass", path="$.intent_class")],
    )
    samples = [
        SampleRecord(sample_id="s1", gold={"intent_class": "refund"}, input=SampleInput(text="A"))
    ]
    clean = [
        PredictionRecord(sample_id="s1", model_id="m", parsed={"intent_class": "refund"})
    ]
    variants = [
        VariantRecord(
            variant_id="s1__bad",
            parent_sample_id="s1",
            perturbation_id="case_lower",
            input=SampleInput(text="a"),
            semantic_validity="INVALID",
        ),
        VariantRecord(
            variant_id="s1__ok",
            parent_sample_id="s1",
            perturbation_id="case_lower",
            input=SampleInput(text="a"),
            semantic_validity="VERIFIED",
        ),
    ]
    vpreds = [
        PredictionRecord(sample_id="s1__bad", model_id="m", parsed={"intent_class": "account"}),
        PredictionRecord(sample_id="s1__ok", model_id="m", parsed={"intent_class": "refund"}),
    ]
    recs = build_robustness_records(
        samples=samples,
        clean_preds=clean,
        variants=variants,
        variant_preds=vpreds,
        task=task,
        relation=MetamorphicRelationSpec(type="invariance", targets=["intent_class"]),
    )
    metrics = aggregate_robustness(recs)
    assert metrics["coverage"]["n_applicable"] == 1
    assert metrics["by_target"]["intent_class"]["flip_rate"] == 0.0


def test_target_rename_only_config():
    task = TaskSpec(
        name="t",
        task_type="classification",
        targets=[TargetSpec(name="escalation", type="multiclass", path="$.intent_class")],
    )
    samples = [
        SampleRecord(sample_id="s1", gold={"intent_class": "refund"}, input=SampleInput(text="A"))
    ]
    clean = [PredictionRecord(sample_id="s1", model_id="m", parsed={"intent_class": "refund"})]
    variants = [
        VariantRecord(
            variant_id="s1__v",
            parent_sample_id="s1",
            perturbation_id="case_lower",
            input=SampleInput(text="a"),
            semantic_validity="VERIFIED",
        )
    ]
    vpreds = [PredictionRecord(sample_id="s1__v", model_id="m", parsed={"intent_class": "refund"})]
    recs = build_robustness_records(
        samples=samples,
        clean_preds=clean,
        variants=variants,
        variant_preds=vpreds,
        task=task,
        relation=MetamorphicRelationSpec(type="invariance", targets=["escalation"]),
    )
    assert recs[0].target == "escalation"
    assert recs[0].relation_satisfied is True


def test_kernel_no_business_tokens():
    for rel in (
        "src/linguaeval/robustness/aggregate.py",
        "src/linguaeval/robustness/relations.py",
        "src/linguaeval/robustness/registry.py",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for banned in ("n2s", "routing_skill", "banking", "BCA", "bca"):
            assert banned not in text


def test_toy_invariance_known_rates():
    out = run_offline_robustness(ROOT / "configs/examples/15_robustness_toy_invariance.yaml")
    m = json.loads((out / "robustness_metrics.json").read_text(encoding="utf-8"))
    assert m["status"] == "AVAILABLE"
    assert m["coverage"]["n_applicable"] == 8
    assert m["coverage"]["n_excluded"] == 1
    block = m["by_target"]["intent_class"]
    assert block["flip_rate"] == pytest.approx(0.25)
    assert block["metamorphic_violation_rate"] == pytest.approx(0.25)
    assert block["accuracy_clean"] == pytest.approx(1.0)
    assert block["accuracy_perturbed"] == pytest.approx(0.75)
    assert block["delta_accuracy"] == pytest.approx(-0.25)
    assert block["variant_all_correct_rate"] == pytest.approx(0.75)
    assert block["end_to_end_robust_success_rate"] == pytest.approx(0.75)
    assert "metrics" in m and "clean" in m["metrics"]
    assert "intent_class" in m["metrics"]["clean"]["targets"]
    assert "macro_f1" in m["metrics"]["clean"]["targets"]["intent_class"]
    assert block["transitions"].get("robustness_regression") == 2
    assert block["transitions"].get("stable_correct") == 6

