import pytest

from linguaeval.metrics.aggregate import build_business_metrics


def test_primary_requires_config_not_hardcoded_names():
    scored = {
        "task": "x",
        "task_type": "classification",
        "schema": {},
        "targets": {
            "fraud": {"f1": 0.5, "type": "binary"},
            "n2s": {"f1": 0.9, "type": "binary"},
        },
    }
    # Without report config → no primary (even if "n2s" exists)
    out = build_business_metrics(scored, report_cfg={})
    assert "primary" not in out

    out2 = build_business_metrics(
        scored, report_cfg={"primary_target": "fraud", "primary_metric": "f1"}
    )
    assert out2["primary"]["target"] == "fraud"
    assert out2["primary"]["value"] == 0.5


def test_primary_missing_target_errors():
    scored = {"targets": {"a": {"f1": 1.0}}}
    with pytest.raises(KeyError):
        build_business_metrics(scored, report_cfg={"primary_target": "missing"})
