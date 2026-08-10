"""P2-C0: robustness semantics hardening."""

from __future__ import annotations

from linguaeval.core.schema import (
    MetamorphicRelationSpec,
    PredictionRecord,
    SampleInput,
    SampleRecord,
    TaskSpec,
    TargetSpec,
    VariantRecord,
)
from linguaeval.robustness.aggregate import build_robustness_records
from linguaeval.robustness.generate import generate_variants, perturbation_applies
from linguaeval.core.schema import PerturbationSpec


def test_score_record_drives_correctness_not_string_hack():
    task = TaskSpec(
        name="t",
        task_type="classification",
        targets=[TargetSpec(name="intent_class", type="multiclass", path="$.intent_class")],
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
        relation=MetamorphicRelationSpec(type="invariance", targets=["intent_class"]),
    )
    assert recs[0].clean_correct is True
    assert recs[0].variant_correct is True
    assert recs[0].transition == "stable_correct"


def test_end_to_end_requires_clean_correct():
    task = TaskSpec(
        name="t",
        task_type="classification",
        targets=[TargetSpec(name="intent_class", type="multiclass", path="$.intent_class")],
    )
    samples = [
        SampleRecord(sample_id="s1", gold={"intent_class": "refund"}, input=SampleInput(text="A"))
    ]
    # clean wrong, variant correct
    clean = [PredictionRecord(sample_id="s1", model_id="m", parsed={"intent_class": "shipping"})]
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
    from linguaeval.robustness.aggregate import aggregate_robustness

    recs = build_robustness_records(
        samples=samples,
        clean_preds=clean,
        variants=variants,
        variant_preds=vpreds,
        task=task,
        relation=MetamorphicRelationSpec(type="invariance", targets=["intent_class"]),
    )
    m = aggregate_robustness(recs)
    block = m["by_target"]["intent_class"]
    assert block["variant_all_correct_rate"] == 1.0
    assert block["end_to_end_robust_success_rate"] == 0.0
    assert recs[0].transition == "perturbation_gain"


def test_applies_to_skips_code_input():
    sample = SampleRecord(
        sample_id="c1",
        gold={"intent_class": "refund"},
        input=SampleInput(text="x = 1"),
        meta={"input_type": "code"},
    )
    spec = PerturbationSpec(id="case_lower", applies_to={"input_type": "natural_text"})
    assert perturbation_applies(sample, spec) is False
    variants = generate_variants([sample], [spec], seed=1)
    assert variants[0].semantic_validity == "NOT_APPLICABLE"
    assert variants[0].meta.get("exclusion") == "not_applicable"
