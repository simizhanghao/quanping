"""P2-C realistic perturbations — non-N2S smoke."""

from __future__ import annotations

import json
from pathlib import Path

from linguaeval.core.perturb_runner import run_offline_perturb
from linguaeval.core.schema import PerturbationSpec, SampleInput, SampleRecord
from linguaeval.robustness.generate import generate_variants
from linguaeval.robustness.registry import list_perturbations
from linguaeval.robustness.transforms_realistic import (
    apply_code_switch,
    apply_context_distractor,
    apply_typo,
)

ROOT = Path(__file__).resolve().parents[1]


def test_registry_includes_realistic_ids():
    ids = list_perturbations()
    assert "typo" in ids
    assert "code_switch" in ids
    assert "context_distractor" in ids


def test_typo_changes_text_and_respects_seed():
    spec = PerturbationSpec(id="typo", severity=3, seed=42, params={"edit_ratio": 0.2})
    a = apply_typo(SampleInput(text="Refund please now"), spec)
    b = apply_typo(SampleInput(text="Refund please now"), spec)
    assert a.text == b.text
    assert a.text != "Refund please now"


def test_typo_protected_tokens_not_required_to_change_those_chars():
    # With only protected content, may NO-OP at generate level
    sample = SampleRecord(
        sample_id="p",
        gold={"intent_class": "refund"},
        input=SampleInput(text="Refund"),
    )
    variants = generate_variants(
        [sample],
        [PerturbationSpec(id="typo", severity=1, seed=1, params={"protected_tokens": ["Refund"]})],
        seed=1,
    )
    assert variants[0].semantic_validity == "NOT_APPLICABLE"


def test_code_switch_uses_lexicon_params():
    spec = PerturbationSpec(
        id="code_switch",
        severity=1,
        seed=0,
        params={"lexicon": {"please": "pls", "package": "parcel"}, "max_swaps": 2},
    )
    out = apply_code_switch(SampleInput(text="please send package"), spec)
    assert "pls" in (out.text or "")
    assert "parcel" in (out.text or "")


def test_code_switch_empty_lexicon_noop():
    spec = PerturbationSpec(id="code_switch", seed=1, params={})
    out = apply_code_switch(SampleInput(text="hello world"), spec)
    assert out.text == "hello world"


def test_context_distractor_prefix():
    spec = PerturbationSpec(
        id="context_distractor",
        seed=2,
        params={"distractors": ["NOTE:"], "position": "prefix"},
    )
    out = apply_context_distractor(SampleInput(text="I want a refund"), spec)
    assert (out.text or "").startswith("NOTE:")


def test_perturb_offline_realistic_config():
    out = run_offline_perturb(ROOT / "configs/examples/17_perturb_toy_realistic.yaml")
    man = json.loads((out / "variant_manifest.json").read_text(encoding="utf-8"))
    assert man["n_parents"] == 8
    assert man["n_generated"] == 24
    assert man["n_valid"] + man["n_noop"] + man.get("n_not_applicable", 0) <= man["n_generated"]
    assert any(s["id"] == "typo" for s in man["perturbation_specs"])
    # at least some valid variants expected on English toy
    assert man["n_valid"] >= 1
