"""P1-A acceptance: A toy transitions, B rename via YAML, C N2S, D align fail, E applicable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linguaeval.compare.alignment import AlignmentError, require_strict_alignment
from linguaeval.compare.transitions import build_comparison_records, classify_transition
from linguaeval.core.compare_runner import run_offline_compare
from linguaeval.core.schema import ScoreRecord, TargetScore

ROOT = Path(__file__).resolve().parents[1]


def test_classify_transition_table():
    assert classify_transition(True, True) == "stable_correct"
    assert classify_transition(False, True) == "gain"
    assert classify_transition(True, False) == "regression"
    assert classify_transition(False, False) == "both_wrong"


def test_a_toy_known_transitions_and_b_intent_class_name():
    out = run_offline_compare(ROOT / "configs/examples/05_compare_base_sft_toy.yaml")
    metrics = json.loads((out / "comparison_metrics.json").read_text(encoding="utf-8"))
    trans = metrics["transitions"]
    assert metrics["compare"]["target"] == "intent_class"
    assert trans["stable_correct"] == 20
    assert trans["gain"] == 6
    assert trans["regression"] == 2
    assert trans["both_wrong"] == 4
    assert trans["net_gain"] == 4
    assert (
        trans["stable_correct"]
        + trans["gain"]
        + trans["regression"]
        + trans["both_wrong"]
        == trans["transition_eligible"]
        == 32
    )
    # metric delta present
    assert "accuracy" in metrics["metric_deltas"]["metrics"]
    assert (out / "gain_cases.jsonl").is_file()
    assert (out / "regression_cases.jsonl").is_file()
    gains = (out / "gain_cases.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(gains) == 6


def test_d_alignment_failure_on_missing_candidate():
    class _P:
        def __init__(self, sid: str):
            self.sample_id = sid

    with pytest.raises(AlignmentError) as ei:
        require_strict_alignment([_P("a"), _P("b")], [_P("a")])
    assert "candidate missing sample_id=b" in str(ei.value)


def test_e_applicable_invariant_excludes_na():
    # 5 applicable + 3 N/A → 4-cell sums to applicable eligible (5)
    def _score(sid: str, applicable: bool, correct: bool, fmt_ok: bool = True) -> ScoreRecord:
        return ScoreRecord(
            sample_id=sid,
            model_id="x",
            parse_ok=fmt_ok,
            schema_ok=fmt_ok,
            targets={
                "intent": TargetScore(
                    gold="a" if applicable else None,
                    pred="a" if correct else "b",
                    correct=correct if applicable else None,
                    applicable=applicable,
                )
            },
        )

    # applicable: 2 stable, 1 gain, 1 regression, 1 both_wrong = 5
    baseline = [
        _score("1", True, True),
        _score("2", True, True),
        _score("3", True, False),
        _score("4", True, True),
        _score("5", True, False),
        _score("6", False, False),
        _score("7", False, False),
        _score("8", False, False),
    ]
    candidate = [
        _score("1", True, True),
        _score("2", True, True),
        _score("3", True, True),  # gain
        _score("4", True, False),  # regression
        _score("5", True, False),  # both_wrong
        _score("6", False, False),
        _score("7", False, False),
        _score("8", False, False),
    ]
    records, counts = build_comparison_records(
        baseline, candidate, target="intent", denominator="semantic"
    )
    assert counts["not_applicable_samples"] == 3
    assert counts["applicable_samples"] == 5
    assert counts["transition_eligible"] == 5
    assert counts["stable_correct"] == 2
    assert counts["gain"] == 1
    assert counts["regression"] == 1
    assert counts["both_wrong"] == 1
    assert (
        counts["stable_correct"]
        + counts["gain"]
        + counts["regression"]
        + counts["both_wrong"]
        == counts["applicable_samples"]
    )
    assert sum(1 for r in records if r.exclusion == "not_applicable") == 3


def test_c_n2s_reference_pair():
    base_path = Path(
        "/data/hanchengcheng/hcc_1/LlamaFactory/tests/yewupingce/n2s_test/"
        "n2s_result/content_Indonesian_multi_skill_qwen3_4b_base_en.json"
    )
    sft_path = Path(
        "/data/hanchengcheng/hcc_1/LlamaFactory/tests/yewupingce/n2s_test/"
        "n2s_result/qwen3_4b_test3.json"
    )
    if not base_path.is_file() or not sft_path.is_file():
        pytest.skip("N2S prediction fixtures not present on this machine")

    out = run_offline_compare(ROOT / "configs/examples/06_compare_base_sft_n2s.yaml")
    metrics = json.loads((out / "comparison_metrics.json").read_text(encoding="utf-8"))
    assert metrics["compare"]["target"] == "n2s"
    cand = metrics["candidate_business"]["primary"]
    base = metrics["baseline_business"]["primary"]
    assert cand["f1"] == 0.8
    assert cand["precision"] == 0.86
    assert cand["recall"] == 0.75
    # legacy REPORT_base_vs_sft: 4B base F1 0.32 / P 0.19 / R 0.96
    assert base["f1"] == 0.32
    assert base["precision"] == 0.19
    assert base["recall"] == 0.96
    assert metrics["metric_deltas"]["metrics"]["f1"]["delta"] == pytest.approx(0.48)
    trans = metrics["transitions"]
    assert (
        trans["stable_correct"]
        + trans["gain"]
        + trans["regression"]
        + trans["both_wrong"]
        == trans["transition_eligible"]
    )
    assert trans["gain"] + trans["regression"] > 0
