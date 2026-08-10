"""P2-B: deterministic surface transforms + perturb-offline."""

from __future__ import annotations

import json
from pathlib import Path

from linguaeval.core.perturb_runner import run_offline_perturb
from linguaeval.core.schema import PerturbationSpec, SampleInput, SampleRecord
from linguaeval.robustness.generate import generate_variants, variant_fingerprint
from linguaeval.robustness.registry import apply_perturbation, get_perturbation
from linguaeval.robustness.transforms import (
    apply_case_lower,
    apply_collapse_whitespace,
    apply_strip_punctuation,
)

ROOT = Path(__file__).resolve().parents[1]


def test_surface_transforms():
    spec = PerturbationSpec(id="x")
    assert apply_case_lower(SampleInput(text="AbC"), spec).text == "abc"
    assert apply_strip_punctuation(SampleInput(text="Hi, there!"), spec).text == "Hi there"
    assert apply_collapse_whitespace(SampleInput(text="  a   b\tc  "), spec).text == "a b c"


def test_generate_variants_count_and_fingerprint_stable():
    samples = [
        SampleRecord(sample_id="a", gold={"intent_class": "refund"}, input=SampleInput(text="Hello!")),
        SampleRecord(sample_id="b", gold={"intent_class": "shipping"}, input=SampleInput(text="Where?")),
    ]
    pids = ["case_lower", "strip_punctuation", "collapse_whitespace"]
    v1 = generate_variants(samples, pids, seed=42)
    v2 = generate_variants(samples, pids, seed=42)
    assert len(v1) == 6
    assert variant_fingerprint(v1) == variant_fingerprint(v2)
    lower = next(v for v in v1 if v.perturbation_id == "case_lower" and v.parent_sample_id == "a")
    assert lower.input.text == "hello!"
    assert lower.semantic_validity == "AUTO_VALIDATED"


def test_perturb_offline_cli_toy():
    out = run_offline_perturb(ROOT / "configs/examples/16_perturb_toy_surface.yaml")
    man = json.loads((out / "variant_manifest.json").read_text(encoding="utf-8"))
    assert man["n_parents"] == 8
    assert man["n_generated"] == 24  # 8 * 3
    assert man["n_valid"] == 24
    assert man["variant_fingerprint"]
    lines = (out / "variants.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 24
