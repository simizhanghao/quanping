"""P1-C: fixed slices + CI-aware gates."""

from __future__ import annotations

import json
from pathlib import Path

from linguaeval.compare.gates import evaluate_gates
from linguaeval.compare.slices import apply_buckets, resolve_slice_key
from linguaeval.core.compare_runner import run_offline_compare
from linguaeval.core.schema import (
    SampleInput,
    SampleRecord,
    ScoreRecord,
    TargetScore,
)

ROOT = Path(__file__).resolve().parents[1]


def test_resolve_slice_sources_generic():
    sample = SampleRecord(
        sample_id="x1",
        gold={"flag": True, "routing_skill": "banking"},
        input=SampleInput(text="hello world"),
        meta={"language": "en"},
        conversation={"turn_id": 4, "dialogue_id": 9},
    )
    sb = ScoreRecord(
        sample_id="x1",
        parse_ok=True,
        schema_ok=True,
        targets={"intent": TargetScore(gold="refund", pred="refund", correct=True)},
    )
    sc = ScoreRecord(
        sample_id="x1",
        parse_ok=False,
        schema_ok=False,
        targets={"intent": TargetScore(gold="refund", pred="shipping", correct=False)},
    )
    assert resolve_slice_key(sample, sb, sc, target="intent", source="meta.language") == "en"
    assert resolve_slice_key(sample, sb, sc, target="intent", source="target.gold") == "refund"
    assert resolve_slice_key(sample, sb, sc, target="intent", source="gold.routing_skill") == "banking"
    assert resolve_slice_key(sample, sb, sc, target="intent", source="format.both_ok") == "false"
    assert resolve_slice_key(sample, sb, sc, target="intent", source="input.text.length") == "11"
    assert apply_buckets("4", {"buckets": [3, 10], "labels": ["early", "mid"]}) == "mid"


def test_gate_pass_and_fail():
    ctx = {
        "candidate_business": {"primary": {"value": 0.8}},
        "statistics": {"metrics": {"f1": {"delta": {"ci_low": 0.1}}}},
    }
    ok = evaluate_gates(
        ctx,
        [
            {"id": "a", "path": "candidate_business.primary.value", "op": ">=", "value": 0.75},
            {"id": "b", "path": "statistics.metrics.f1.delta.ci_low", "op": ">=", "value": 0},
        ],
    )
    assert ok["status"] == "PASS"
    bad = evaluate_gates(
        ctx,
        [{"id": "a", "path": "candidate_business.primary.value", "op": ">=", "value": 0.99}],
    )
    assert bad["status"] == "FAIL"
    assert bad["gates"][0]["status"] == "FAIL"


def test_toy_compare_slices_and_gate():
    out = run_offline_compare(ROOT / "configs/examples/05_compare_base_sft_toy.yaml")
    slices = json.loads((out / "slice_comparison.json").read_text(encoding="utf-8"))
    gate = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    assert "language" in slices["slices"]
    assert "gold_label" in slices["slices"]
    gold_vals = slices["slices"]["gold_label"]["values"]
    assert set(gold_vals) >= {"refund", "shipping", "account"}
    # P1-D: CI gate on n=32 → INSUFFICIENT_SUPPORT (must not look like FAIL)
    assert gate["status"] == "INSUFFICIENT_SUPPORT", gate
    assert gate["n_fail"] == 0
    by_id = {g["id"]: g for g in gate["gates"]}
    assert by_id["candidate_accuracy_min"]["status"] == "PASS"
    assert by_id["delta_accuracy_min"]["status"] == "PASS"
    assert by_id["delta_accuracy_ci_lower"]["status"] == "INSUFFICIENT_SUPPORT"
