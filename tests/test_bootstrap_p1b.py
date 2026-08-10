"""P1-B: paired / cluster bootstrap."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from linguaeval.compare.bootstrap import (
    PairRow,
    build_cluster_map,
    resample_indices,
    resolve_unit_id,
    run_paired_bootstrap,
)
from linguaeval.core.compare_runner import run_offline_compare
from linguaeval.core.schema import SampleInput, SampleRecord

ROOT = Path(__file__).resolve().parents[1]


def test_cluster_map_groups_turns():
    units = ["d1", "d1", "d2", "d2", "d2", "d3"]
    m = build_cluster_map(units)
    assert m["d1"] == [0, 1]
    assert m["d2"] == [2, 3, 4]
    assert m["d3"] == [5]


def test_cluster_resample_does_not_draw_individual_turns():
    unit_ids = ["d1", "d1", "d2", "d2", "d3"]

    class _ClusterOnlyRng:
        def __init__(self):
            self.choices = ["d1", "d2", "d1"]

        def choice(self, keys):
            assert set(keys) == {"d1", "d2", "d3"}
            return self.choices.pop(0)

        def randrange(self, n):
            raise AssertionError(
                "cluster bootstrap must resample units, not per-turn randrange"
            )

    idxs = resample_indices(5, unit_ids=unit_ids, rng=_ClusterOnlyRng())
    # 3 units drawn: d1,d2,d1 → rows [0,1] + [2,3] + [0,1]
    assert idxs == [0, 1, 2, 3, 0, 1]


def test_sample_unit_uses_row_bootstrap():
    class _RowRng:
        def __init__(self):
            self.n = 0

        def randrange(self, n):
            assert n == 4
            self.n += 1
            return 0

        def choice(self, keys):
            raise AssertionError("unique units should not use cluster choice path")

    # unique unit ids ⇒ ordinary bootstrap
    idxs = resample_indices(4, unit_ids=["a", "b", "c", "d"], rng=_RowRng())
    assert idxs == [0, 0, 0, 0]


def test_resolve_unit_id_from_conversation():
    s = SampleRecord(
        sample_id="640_3",
        gold={"n2s": True},
        input=SampleInput(text="x"),
        conversation={"dialogue_id": 640, "turn_id": 3},
    )
    assert resolve_unit_id(s, "dialogue_id") == "640"
    assert resolve_unit_id(s, "sample") == "640_3"


def test_percentile_ci_and_bootstrap_delta_smoke():
    # 10 rows: baseline always wrong on first 6, candidate fixes them
    rows = []
    for i in range(10):
        gold = True
        b_pred = False if i < 6 else True
        c_pred = True
        rows.append(
            PairRow(
                sample_id=f"s{i}",
                unit_id=f"d{i // 2}",  # 5 dialogues × 2 turns
                gold=gold,
                baseline_pred=b_pred,
                candidate_pred=c_pred,
                transition="gain" if i < 6 else "stable_correct",
            )
        )
    stats = run_paired_bootstrap(
        rows,
        target_type="binary",
        metric_names=["f1", "recall"],
        n_bootstrap=300,
        confidence_level=0.95,
        seed=0,
        round_digits=4,
    )
    assert stats["cluster_mode"] is True
    assert stats["n_units"] == 5
    d = stats["metrics"]["f1"]["delta"]
    assert d["ci_low"] is not None and d["ci_high"] is not None
    assert d["ci_low"] <= d["point"] <= d["ci_high"]
    assert stats["transitions"]["net_gain"]["point"] == 6


def test_toy_compare_writes_statistics():
    out = run_offline_compare(ROOT / "configs/examples/05_compare_base_sft_toy.yaml")
    stats = json.loads((out / "statistics.json").read_text(encoding="utf-8"))
    assert stats["enabled"] is True
    assert stats["bootstrap_unit"] == "sample"
    assert stats["cluster_mode"] is False
    assert "accuracy" in stats["metrics"]
    d = stats["metrics"]["accuracy"]["delta"]
    assert d["ci_low"] <= d["point"] <= d["ci_high"]


def test_n2s_cluster_bootstrap_ci_smoke():
    base_path = Path(
        "/data/hanchengcheng/hcc_1/LlamaFactory/tests/yewupingce/n2s_test/"
        "n2s_result/content_Indonesian_multi_skill_qwen3_4b_base_en.json"
    )
    if not base_path.is_file():
        pytest.skip("N2S fixtures missing")
    cfg_path = ROOT / "configs/examples/06_compare_base_sft_n2s.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["statistics"]["n_bootstrap"] = 80  # keep CI smoke fast; example YAML stays 1000
    cfg["output_dir"] = str(ROOT / "results/06_compare_base_sft_n2s_bootstrap_smoke")
    # temp config lives outside repo — rewrite relative spec paths to absolute
    for key in ("task_spec", "output_spec", "metric_spec"):
        rel = cfg.get(key)
        if rel:
            cfg[key] = str((cfg_path.parent / rel).resolve())
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "compare_n2s_boot.yaml"
        tmp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        out = run_offline_compare(tmp)
    stats = json.loads((out / "statistics.json").read_text(encoding="utf-8"))
    assert stats["bootstrap_unit"] == "dialogue_id"
    assert stats["cluster_mode"] is True
    assert stats["n_units"] < stats["n_rows"]
    d = stats["metrics"]["f1"]["delta"]
    assert d["point"] == 0.48
    assert d["ci_low"] > 0
    assert d["ci_high"] >= d["ci_low"]
