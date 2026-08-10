"""P3-F-A: language matrix compares report.primary_metric, not hardcoded accuracy."""

from __future__ import annotations

import json
from pathlib import Path

from linguaeval.core.language_matrix_runner import run_offline_language_matrix

ROOT = Path(__file__).resolve().parents[1]


def test_regression_uses_metric_path_not_delta_accuracy():
    out = run_offline_language_matrix(
        ROOT / "configs/examples/22_language_matrix_belebele_toy.yaml"
    )
    reg = json.loads((out / "language_regression.json").read_text(encoding="utf-8"))
    report = json.loads((out / "language_capability_report.json").read_text(encoding="utf-8"))
    ind = reg["by_language"]["ind"]
    assert ind["primary_metric"] == "accuracy"
    assert ind["metric_path"] == "targets.answer.accuracy"
    assert "baseline_value" in ind and "delta" in ind
    assert "delta_accuracy" not in ind
    assert report["primary_metric"] == "accuracy"
    assert report["rows"][0]["metric_path"] == "targets.answer.accuracy"

    md = (out / "report.md").read_text(encoding="utf-8")
    assert "Metric" in md
    assert "targets.answer.accuracy" in md
