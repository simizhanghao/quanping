"""Golden comparison protocol + comparability (P1-D)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class ComparisonProtocolError(ValueError):
    """Raised when baseline/candidate are not formally comparable."""

    def __init__(self, code: str, message: str, *, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = details or {}


def _norm_path(p: Optional[str]) -> str:
    if not p:
        return ""
    return str(Path(p).resolve()) if Path(p).exists() else str(p).replace("\\", "/")


def _endswith_match(path: str, suffix: str) -> bool:
    p = path.replace("\\", "/")
    s = (suffix or "").replace("\\", "/")
    return bool(s) and p.endswith(s)


def evaluate_comparability(
    comparability_cfg: Dict[str, Any],
    *,
    baseline_side: Dict[str, Any],
    candidate_side: Dict[str, Any],
) -> Dict[str, Any]:
    """Derive semantic_comparable / efficiency_comparable from declared metadata.

    Semantic fields (must match for business-effect compare):
      prompt_protocol, context_protocol, scoring_protocol, dataset_protocol

    Efficiency fields (must match for latency/throughput compare):
      backend_family, hardware, dtype, concurrency, decoding.*
    """
    semantic_keys = list(
        comparability_cfg.get("semantic_keys")
        or [
            "prompt_protocol",
            "context_protocol",
            "scoring_protocol",
            "dataset_protocol",
        ]
    )
    efficiency_keys = list(
        comparability_cfg.get("efficiency_keys")
        or [
            "backend_family",
            "hardware",
            "dtype",
            "concurrency",
        ]
    )
    # shared semantic block optional
    shared = dict(comparability_cfg.get("semantic") or {})
    b = {**shared, **dict(baseline_side or {})}
    c = {**shared, **dict(candidate_side or {})}

    sem_mismatches: List[str] = []
    for k in semantic_keys:
        if k in b or k in c:
            if b.get(k) != c.get(k):
                sem_mismatches.append(k)

    eff_mismatches: List[str] = []
    for k in efficiency_keys:
        if k in b or k in c:
            if b.get(k) != c.get(k):
                eff_mismatches.append(k)
    # decoding dict compare if present on either side
    bd = b.get("decoding") or {}
    cd = c.get("decoding") or {}
    if bd or cd:
        if bd != cd:
            eff_mismatches.append("decoding")

    semantic_comparable = len(sem_mismatches) == 0
    efficiency_comparable = len(eff_mismatches) == 0
    # if no efficiency fields declared at all, efficiency is unknown → false
    declared_eff = any(k in b or k in c for k in efficiency_keys) or bool(bd or cd)
    if not declared_eff:
        efficiency_comparable = False

    return {
        "semantic_comparable": semantic_comparable,
        "efficiency_comparable": efficiency_comparable,
        "semantic_mismatches": sem_mismatches,
        "efficiency_mismatches": eff_mismatches,
        "baseline": {k: b.get(k) for k in sorted(set(semantic_keys) | set(efficiency_keys) | {"decoding"})},
        "candidate": {k: c.get(k) for k in sorted(set(semantic_keys) | set(efficiency_keys) | {"decoding"})},
    }


def validate_comparison_protocol(
    protocol_cfg: Dict[str, Any],
    *,
    baseline_path: Optional[str],
    candidate_path: Optional[str],
    dataset_fingerprint: str,
    task_spec_hash: Optional[str],
    output_spec_hash: Optional[str],
    metric_spec_hash: Optional[str],
    n_aligned: int,
    comparability: Dict[str, Any],
    require_semantic_comparable: bool = True,
) -> Dict[str, Any]:
    """Validate golden pair allowlist + protocol invariants.

    On failure raises ComparisonProtocolError with code NOT_COMPARABLE.
    """
    if not protocol_cfg:
        # protocol optional for toy; still emit audit stub
        return {
            "protocol_id": None,
            "enforced": False,
            "allowed_pair_matched": None,
            "comparability": comparability,
        }

    protocol_id = protocol_cfg.get("protocol_id")
    audit: Dict[str, Any] = {
        "protocol_id": protocol_id,
        "enforced": True,
        "dataset_fingerprint": dataset_fingerprint,
        "task_spec_hash": task_spec_hash,
        "output_spec_hash": output_spec_hash,
        "metric_spec_hash": metric_spec_hash,
        "n_aligned": n_aligned,
        "baseline_path": baseline_path,
        "candidate_path": candidate_path,
        "comparability": comparability,
    }

    # Expected fingerprints if declared
    expected_ds = protocol_cfg.get("dataset_fingerprint")
    if expected_ds and expected_ds != dataset_fingerprint:
        raise ComparisonProtocolError(
            "NOT_COMPARABLE",
            "dataset_fingerprint mismatch vs comparison_protocol",
            details={"expected": expected_ds, "observed": dataset_fingerprint},
        )
    for key, observed in (
        ("task_spec_hash", task_spec_hash),
        ("output_spec_hash", output_spec_hash),
        ("metric_spec_hash", metric_spec_hash),
    ):
        expected = protocol_cfg.get(key)
        if expected and observed and expected != observed:
            raise ComparisonProtocolError(
                "NOT_COMPARABLE",
                f"{key} mismatch vs comparison_protocol",
                details={"expected": expected, "observed": observed},
            )

    allowed = list(protocol_cfg.get("allowed_pairs") or [])
    matched = None
    if allowed:
        b_path = _norm_path(baseline_path)
        c_path = _norm_path(candidate_path)
        for pair in allowed:
            b_suf = str(pair.get("baseline_path_suffix") or pair.get("baseline") or "")
            c_suf = str(pair.get("candidate_path_suffix") or pair.get("candidate") or "")
            if _endswith_match(b_path, b_suf) and _endswith_match(c_path, c_suf):
                matched = {
                    "baseline_path_suffix": b_suf,
                    "candidate_path_suffix": c_suf,
                    "note": pair.get("note"),
                }
                break
        if matched is None:
            raise ComparisonProtocolError(
                "NOT_COMPARABLE",
                "baseline/candidate paths are not in allowed_pairs (golden reference)",
                details={
                    "baseline_path": b_path,
                    "candidate_path": c_path,
                    "allowed_pairs": allowed,
                },
            )
    audit["allowed_pair_matched"] = matched

    if require_semantic_comparable and not comparability.get("semantic_comparable", True):
        raise ComparisonProtocolError(
            "NOT_COMPARABLE",
            "semantic_comparable=false; business-effect compare refused",
            details={"mismatches": comparability.get("semantic_mismatches")},
        )

    # efficiency mismatch is allowed for business compare but recorded
    audit["status"] = "OK"
    return audit
