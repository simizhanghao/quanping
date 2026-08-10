from pathlib import Path

from linguaeval.core.runner import run_offline_score

ROOT = Path(__file__).resolve().parents[1]


def test_toy_multiclass_offline_accuracy():
    out = run_offline_score(ROOT / "configs/examples/toy_multiclass_offline.yaml")
    import json

    business = json.loads((out / "business_metrics.json").read_text(encoding="utf-8"))
    label = business["targets"]["label"]
    # 30/32 correct in fixture
    assert label["support"] == 32
    assert abs(label["accuracy"] - 0.9375) < 1e-6
    assert label["macro_f1"] > 0.9
