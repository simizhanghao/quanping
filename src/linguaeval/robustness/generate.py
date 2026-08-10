"""Generate VariantRecords from samples via registered perturbations (P2-B)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

from linguaeval.core.schema import PerturbationSpec, SampleRecord, VariantRecord
from linguaeval.robustness.registry import apply_perturbation, get_perturbation


def _variant_id(parent_id: str, perturbation_id: str, idx: int) -> str:
    return f"{parent_id}__{perturbation_id}__{idx:02d}"


def generate_variants(
    samples: Sequence[SampleRecord],
    perturbation_ids: Sequence[str],
    *,
    seed: int = 42,
    severity: int = 1,
    semantic_validity: str = "AUTO_VALIDATED",
    specs: Optional[Dict[str, PerturbationSpec]] = None,
) -> List[VariantRecord]:
    """Apply each listed perturbation once per sample (deterministic surface)."""
    out: List[VariantRecord] = []
    for s in samples:
        for pid in perturbation_ids:
            spec = (specs or {}).get(pid) or get_perturbation(pid)
            # lineage seed: same run seed for all; transform itself is deterministic
            new_inp = apply_perturbation(s.input, spec)
            vid = _variant_id(s.sample_id, pid, 1)
            out.append(
                VariantRecord(
                    variant_id=vid,
                    parent_sample_id=s.sample_id,
                    perturbation_id=pid,
                    input=new_inp,
                    severity=int(spec.severity if spec.severity is not None else severity),
                    seed=seed,
                    semantic_policy=spec.semantic_policy,
                    semantic_validity=semantic_validity,
                    transform_version=spec.transform_version,
                    meta={"parent_text": s.input.text},
                )
            )
    return out


def variant_fingerprint(variants: Sequence[VariantRecord]) -> str:
    """Stable fingerprint so Base/SFT can share the same variant set (P2-D)."""
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
    valid = sum(
        1
        for v in variants
        if str(v.semantic_validity).upper() in {"VERIFIED", "AUTO_VALIDATED"}
    )
    return {
        "n_parents": n_parent,
        "n_requested_perturbations": len(list(requested_perturbations)),
        "n_generated": n_gen,
        "n_valid": valid,
        "n_invalid_or_unverified": n_gen - valid,
        "expected_if_one_each": n_parent * len(list(requested_perturbations)),
    }
