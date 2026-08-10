from pathlib import Path
import json

import pytest

from linguaeval.core.runner import run_offline_score

ROOT = Path(__file__).resolve().parents[1]
N2S_PRED = Path(
    "/data/hanchengcheng/hcc_1/LlamaFactory/tests/yewupingce/n2s_test/"
    "n2s_result/qwen3_4b_test3.json"
)
N2S_METRICS = Path(
    "/data/hanchengcheng/hcc_1/LlamaFactory/tests/yewupingce/n2s_test/"
    "n2s_result/qwen3_4b_test3_metrics.json"
)


@pytest.mark.skipif(not N2S_PRED.is_file(), reason="N2S prediction fixture missing")
def test_n2s_offline_replay_matches_legacy_f1():
    out = run_offline_score(ROOT / "configs/examples/n2s_offline_replay.yaml")
    business = json.loads((out / "business_metrics.json").read_text(encoding="utf-8"))
    legacy = json.loads(N2S_METRICS.read_text(encoding="utf-8"))
    primary = business["primary"]
    assert primary["TP"] == legacy["TP"]
    assert primary["TN"] == legacy["TN"]
    assert primary["FP"] == legacy["FP"]
    assert primary["FN"] == legacy["FN"]
    assert primary["precision"] == legacy["precision"]
    assert primary["recall"] == legacy["recall"]
    assert primary["f1"] == legacy["f1"]
    assert business["schema"]["format_match_rate"] == legacy["format_match_rate"]
