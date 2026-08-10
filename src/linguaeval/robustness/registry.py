"""Perturbation registry — Kernel never branches on perturbation names."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from linguaeval.core.schema import PerturbationSpec, SampleInput
from linguaeval.robustness import transforms as _transforms
from linguaeval.robustness import transforms_realistic as _realistic

ApplyFn = Callable[[SampleInput, PerturbationSpec], SampleInput]

_REGISTRY: Dict[str, PerturbationSpec] = {}
_APPLY: Dict[str, ApplyFn] = {}


def register_perturbation(spec: PerturbationSpec, apply_fn: Optional[ApplyFn] = None) -> None:
    _REGISTRY[spec.id] = spec
    if apply_fn is not None:
        _APPLY[spec.id] = apply_fn


def get_perturbation(perturbation_id: str) -> PerturbationSpec:
    if perturbation_id not in _REGISTRY:
        raise KeyError(f"unknown perturbation_id: {perturbation_id}")
    return _REGISTRY[perturbation_id]


def list_perturbations() -> List[str]:
    return sorted(_REGISTRY.keys())


def apply_perturbation(inp: SampleInput, spec: PerturbationSpec) -> SampleInput:
    fn = _APPLY.get(spec.id)
    if fn is None:
        raise NotImplementedError(
            f"perturbation {spec.id!r} is registered but apply() is not implemented"
        )
    return fn(inp, spec)


def ensure_builtin_perturbation_specs() -> None:
    """Register P2-B surface + P2-C realistic transforms."""
    surface = (
        ("case_lower", _transforms.apply_case_lower, "surface"),
        ("strip_punctuation", _transforms.apply_strip_punctuation, "surface"),
        ("collapse_whitespace", _transforms.apply_collapse_whitespace, "surface"),
    )
    for pid, fn, cat in surface:
        register_perturbation(
            PerturbationSpec(
                id=pid,
                category=cat,
                severity=1,
                semantic_policy="preserve",
                transform_version="1",
                applies_to={"input_type": "natural_text"},
            ),
            fn,
        )
    realistic = (
        ("typo", _realistic.apply_typo, "lexical"),
        ("code_switch", _realistic.apply_code_switch, "lexical"),
        ("context_distractor", _realistic.apply_context_distractor, "context"),
    )
    for pid, fn, cat in realistic:
        register_perturbation(
            PerturbationSpec(
                id=pid,
                category=cat,
                severity=1,
                semantic_policy="preserve",
                transform_version="1",
                applies_to={"input_type": "natural_text"},
            ),
            fn,
        )


ensure_builtin_perturbation_specs()
