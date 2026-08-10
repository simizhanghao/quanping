from pathlib import Path
import json

from linguaeval.core.runner import run_offline_score

ROOT = Path(__file__).resolve().parents[1]


def test_toy_multiclass_offline_accuracy():
    out = run_offline_score(ROOT / "configs/examples/01_score_toy_multiclass.yaml")
    business = json.loads((out / "business_metrics.json").read_text(encoding="utf-8"))
    # target renamed to intent_class — Kernel must not care about the old name "label"
    block = business["targets"]["intent_class"]
    assert block["support"] == 32
    assert abs(block["accuracy"] - 0.9375) < 1e-6
    assert block["macro_f1"] > 0.9
    assert business["primary"]["target"] == "intent_class"
    assert business["primary"]["metric"] == "macro_f1"
    assert business["primary"]["value"] == block["macro_f1"]
