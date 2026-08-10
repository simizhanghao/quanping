import json
from pathlib import Path

from linguaeval.core.runner import run_offline_score

ROOT = Path(__file__).resolve().parents[1]


def test_json_extraction_from_raw_smoke():
    out = run_offline_score(ROOT / "configs/examples/04_score_json_extraction.yaml")
    business = json.loads((out / "business_metrics.json").read_text(encoding="utf-8"))
    assert business["parse"]["mode"] == "from_raw"
    schema = business["schema"]
    # 32 samples; at least e15 missing key, e20 invalid json, e25 type mismatch => format fails
    assert schema["eval_sample_count"] == 32
    assert schema["format_fail_count"] >= 3
    assert "amount" in business["targets"]
    assert business["primary"]["target"] == "amount"
    scores = (out / "scores.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(scores) == 32
    # one score record has joint structure
    row = json.loads(scores[0])
    assert "targets" in row and "amount" in row["targets"] and "time" in row["targets"]
