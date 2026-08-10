"""Perturbation registry — Kernel never branches on perturbation names."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from linguaeval.core.schema import PerturbationSpec, SampleInput
from linguaeval.robustness import transforms as _transforms

# apply(text_or_input, spec) -> SampleInput
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
    """Register P2-B deterministic surface transforms."""
    builtins = (
        ("case_lower", _transforms.apply_case_lower),
        ("strip_punctuation", _transforms.apply_strip_punctuation),
        ("collapse_whitespace", _transforms.apply_collapse_whitespace),
    )
    for pid, fn in builtins:
        spec = PerturbationSpec(
            id=pid,
            category="surface",
            severity=1,
            semantic_policy="preserve",
            transform_version="1",
        )
        register_perturbation(spec, fn)


ensure_builtin_perturbation_specs()
