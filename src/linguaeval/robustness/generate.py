"""Generate VariantRecords from samples via registered perturbations (P2-B/C0)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from linguaeval.core.schema import PerturbationSpec, SampleInput, SampleRecord, VariantRecord
from linguaeval.robustness.registry import apply_perturbation, get_perturbation


def _variant_id(parent_id: str, perturbation_id: str, idx: int) -> str:
    return f"{parent_id}__{perturbation_id}__{idx:02d}"


def _input_fingerprint(inp: SampleInput) -> str:
    return json.dumps(asdict(inp), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def perturbation_applies(sample: SampleRecord, spec: PerturbationSpec) -> bool:
    """Applicability gate (P2-C0). Empty applies_to → applicable."""
    req = spec.applies_to or {}
    if not req:
        return True
    meta = sample.meta or {}
    if "input_type" in req:
        got = str(meta.get("input_type") or "natural_text")
        want = str(req["input_type"])
        if got != want:
            return False
    for k, v in req.items():
        if k == "input_type":
            continue
        if meta.get(k) != v:
            return False
    return True


def validate_variant(
    *,
    original: SampleInput,
    variant: SampleInput,
    spec: PerturbationSpec,
    default_validity: str = "AUTO_VALIDATED",
) -> str:
    """Per-variant validation. NO-OP → NOT_APPLICABLE (out of denominator)."""
    if _input_fingerprint(original) == _input_fingerprint(variant):
        return "NOT_APPLICABLE"
    # Deterministic transforms that changed input may be AUTO_VALIDATED.
    # Realistic packs still require allowlisted params (lexicon/distractors) at apply-time.
    return default_validity


def generate_variants(
    samples: Sequence[SampleRecord],
    specs: Sequence[PerturbationSpec],
    *,
    seed: int = 42,
    semantic_validity_if_changed: str = "AUTO_VALIDATED",
) -> List[VariantRecord]:
    """Apply each runtime PerturbationSpec once per sample."""
    out: List[VariantRecord] = []
    for s in samples:
        for raw_spec in specs:
            try:
                base = get_perturbation(raw_spec.id)
                spec = base.merged_with(raw_spec)
            except KeyError:
                spec = raw_spec
            # ensure seed on runtime spec for stochastic transforms
            if spec.seed is None:
                spec = PerturbationSpec(
                    id=spec.id,
                    category=spec.category,
                    severity=spec.severity,
                    seed=seed,
                    semantic_policy=spec.semantic_policy,
                    transform_version=spec.transform_version,
                    params=spec.params,
                    applies_to=spec.applies_to,
                )
            vid = _variant_id(s.sample_id, spec.id, 1)
            if not perturbation_applies(s, spec):
                out.append(
                    VariantRecord(
                        variant_id=vid,
                        parent_sample_id=s.sample_id,
                        perturbation_id=spec.id,
                        input=s.input,
                        severity=int(spec.severity),
                        seed=spec.seed,
                        semantic_policy=spec.semantic_policy,
                        semantic_validity="NOT_APPLICABLE",
                        transform_version=spec.transform_version,
                        meta={
                            "exclusion": "not_applicable",
                            "applies_to": spec.applies_to,
                            "params": spec.params,
                        },
                    )
                )
                continue
            new_inp = apply_perturbation(s.input, spec)
            validity = validate_variant(
                original=s.input,
                variant=new_inp,
                spec=spec,
                default_validity=semantic_validity_if_changed,
            )
            meta: Dict[str, Any] = {"params": spec.params, "parent_text": s.input.text}
            if validity == "NOT_APPLICABLE":
                meta["exclusion"] = "no_op"
            out.append(
                VariantRecord(
                    variant_id=vid,
                    parent_sample_id=s.sample_id,
                    perturbation_id=spec.id,
                    input=new_inp,
                    severity=int(spec.severity),
                    seed=spec.seed,
                    semantic_policy=spec.semantic_policy,
                    semantic_validity=validity,
                    transform_version=spec.transform_version,
                    meta=meta,
                )
            )
    return out


def variant_fingerprint(variants: Sequence[VariantRecord]) -> str:
    payload = [v.to_dict() for v in variants]
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def coverage_audit(
    samples: Sequence[SampleRecord],
    variants: Sequence[VariantRecord],
    *,
    requested_perturbations: Sequence[str],
) -> Dict[str, Any]:
    n_parent = len(samples)
    n_gen = len(variants)
    n_valid = sum(1 for v in variants if str(v.semantic_validity).upper() in {"VERIFIED", "AUTO_VALIDATED"})
    n_noop = sum(1 for v in variants if (v.meta or {}).get("exclusion") == "no_op")
    n_na = sum(
        1
        for v in variants
        if str(v.semantic_validity).upper() == "NOT_APPLICABLE"
        and (v.meta or {}).get("exclusion") != "no_op"
    )
    n_invalid = sum(1 for v in variants if str(v.semantic_validity).upper() == "INVALID")
    n_unverified = sum(1 for v in variants if str(v.semantic_validity).upper() == "UNVERIFIED")
    return {
        "n_parents": n_parent,
        "n_requested_perturbations": len(list(requested_perturbations)),
        "n_generated": n_gen,
        "n_valid": n_valid,
        "n_noop": n_noop,
        "n_not_applicable": n_na,
        "n_invalid": n_invalid,
        "n_unverified": n_unverified,
        "n_evaluated_eligible": n_valid,
        "expected_if_one_each": n_parent * len(list(requested_perturbations)),
    }
