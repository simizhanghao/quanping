import json
from pathlib import Path

from linguaeval.core.runner import run_offline_score

ROOT = Path(__file__).resolve().parents[1]


def test_extraction_strict_le_semantic_and_provenance():
    out = run_offline_score(ROOT / "configs/examples/04_score_json_extraction.yaml")
    business = json.loads((out / "business_metrics.json").read_text(encoding="utf-8"))
    audit = json.loads((out / "data_audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))

    cov = business["coverage"]
    assert cov["eligible_samples"] == 32
    assert cov["with_prediction"] == 32
    assert cov["format_ok_samples"] == 29
    # 29/32 = 0.90625 → MetricSpec.round_digits=4 → 0.9062
    assert cov["coverage_valid"] == 0.9062

    sem = business["metrics_by_mode"]["semantic"]["targets"]["amount"]["exact_match"]
    strict = business["metrics_by_mode"]["strict"]["targets"]["amount"]["exact_match"]
    assert sem is not None and strict is not None
    assert strict <= sem + 1e-9
    assert business["metrics_by_mode"]["strict"]["targets"]["amount"]["denominator"] == 32
    assert business["metrics_by_mode"]["semantic"]["targets"]["amount"]["denominator"] == 29

    assert "dataset_fingerprint" in manifest["provenance"]
    assert "config_hash" in manifest["provenance"]
    assert audit["coverage"]["format_ok_samples"] == 29


def test_n2s_semantic_unchanged_when_all_format_ok():
    out = run_offline_score(ROOT / "configs/examples/03_score_n2s_offline_replay.yaml")
    business = json.loads((out / "business_metrics.json").read_text(encoding="utf-8"))
    assert business["primary"]["f1"] == 0.8
    # all format ok => semantic f1 ~= strict f1
    sem = business["metrics_by_mode"]["semantic"]["targets"]["n2s"]["f1"]
    strict = business["metrics_by_mode"]["strict"]["targets"]["n2s"]["f1"]
    assert sem == strict == 0.8
