"""P3-E language × capability regression report + gates."""

from __future__ import annotations

import json
from pathlib import Path

from linguaeval.core.language_matrix_runner import run_offline_language_matrix

ROOT = Path(__file__).resolve().parents[1]


def test_capability_report_and_gates():
    out = run_offline_language_matrix(
        ROOT / "configs/examples/26_language_capability_report_gates_toy.yaml"
    )
    report = json.loads((out / "language_capability_report.json").read_text(encoding="utf-8"))
    gates = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    assert report["status"] == "AVAILABLE"
    assert "No multilingual total score" in report["note"]
    langs = {(r["language"], r["capability"]) for r in report["rows"]}
    assert ("ind", "reading_comprehension") in langs
    assert ("arb", "reading_comprehension") in langs

    by_id = {g["id"]: g for g in gates["gates"]}
    assert by_id["target_ind_reading_min_gain"]["status"] == "PASS"
    # arb Δ=-0.125 < -0.05 → FAIL
    assert by_id["other_arb_reading_max_drop"]["status"] == "FAIL"
    assert gates["status"] == "FAIL"

    md = (out / "report.md").read_text(encoding="utf-8")
    assert "Language Capability Report" in md
    assert "| Language | Capability |" in md
