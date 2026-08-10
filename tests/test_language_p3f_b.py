"""P3-F-B: language matrix reuses P1 paired compare + bootstrap CI."""

from __future__ import annotations

import json
from pathlib import Path

from linguaeval.core.language_matrix_runner import run_offline_language_matrix

ROOT = Path(__file__).resolve().parents[1]


def test_language_matrix_uses_p1_paired_engine_and_ci():
    out = run_offline_language_matrix(
        ROOT / "configs/examples/22_language_matrix_belebele_toy.yaml"
    )
    reg = json.loads((out / "language_regression.json").read_text(encoding="utf-8"))
    assert reg["engine"] == "p1_paired_compare"
    ind = reg["by_language"]["ind"]
    assert ind["engine"] == "p1_paired_compare"
    assert ind["delta"] == 0.25 or abs(float(ind["delta"]) - 0.25) < 1e-9
    assert ind["delta_ci_low"] is not None
    assert ind["delta_ci_high"] is not None
    assert ind["delta_ci_low"] <= ind["delta"] <= ind["delta_ci_high"]
    assert ind["transitions"]["gain"] >= 0
    assert ind["statistics"]["enabled"] is True
    assert "accuracy" in ind["statistics"]["metrics"]
    assert ind["support"]["n_aligned"] == 8


def test_capability_gates_include_insufficient_support_ci():
    out = run_offline_language_matrix(
        ROOT / "configs/examples/26_language_capability_report_gates_toy.yaml"
    )
    gates = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    by_id = {g["id"]: g for g in gates["gates"]}
    assert by_id["target_ind_reading_min_gain"]["status"] == "PASS"
    assert by_id["other_arb_reading_max_drop"]["status"] == "FAIL"
    assert by_id["target_ind_delta_ci_lower"]["status"] == "INSUFFICIENT_SUPPORT"
    assert by_id["other_arb_delta_ci_lower"]["status"] == "INSUFFICIENT_SUPPORT"
    # FAIL still dominates overall status
    assert gates["status"] == "FAIL"

    md = (out / "report.md").read_text(encoding="utf-8")
    assert "p1_paired_compare" in md
    assert "Δ CI95" in md
