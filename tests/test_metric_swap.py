from linguaeval.core.schema import (
    FormatStatus,
    MetricSpec,
    PredictionRecord,
    SampleInput,
    SampleRecord,
    TaskSpec,
    TargetSpec,
)
from linguaeval.metrics.classification import score_targets


def test_binary_f1_to_f2_without_reinference():
    samples = [
        SampleRecord(sample_id="a", gold={"y": True}, input=SampleInput(text="a")),
        SampleRecord(sample_id="b", gold={"y": True}, input=SampleInput(text="b")),
        SampleRecord(sample_id="c", gold={"y": False}, input=SampleInput(text="c")),
        SampleRecord(sample_id="d", gold={"y": False}, input=SampleInput(text="d")),
        SampleRecord(sample_id="e", gold={"y": True}, input=SampleInput(text="e")),
    ]
    preds = [
        PredictionRecord("a", "m", parsed={"y": True}, format=FormatStatus(True, True)),
        PredictionRecord("b", "m", parsed={"y": False}, format=FormatStatus(True, True)),  # FN
        PredictionRecord("c", "m", parsed={"y": True}, format=FormatStatus(True, True)),  # FP
        PredictionRecord("d", "m", parsed={"y": False}, format=FormatStatus(True, True)),
        PredictionRecord("e", "m", parsed={"y": True}, format=FormatStatus(True, True)),
    ]
    task = TaskSpec(
        name="bin",
        task_type="classification",
        targets=[TargetSpec(name="y", type="binary", path="$.y")],
    )
    m1 = MetricSpec(metrics={"y": ["precision", "recall", "f1", "f2"]}, round_digits=4)
    m2 = MetricSpec(metrics={"y": ["f2"]}, round_digits=4)
    s1 = score_targets(samples, preds, task, m1)
    s2 = score_targets(samples, preds, task, m2)
    assert "f1" in s1["targets"]["y"]
    assert "f2" in s1["targets"]["y"]
    assert "f2" in s2["targets"]["y"]
    # same predictions → same f2
    assert s1["targets"]["y"]["f2"] == s2["targets"]["y"]["f2"]
