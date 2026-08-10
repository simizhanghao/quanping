"""Dataset adapter registry — Kernel looks up by name, never by business if."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from linguaeval.core.schema import PredictionRecord, SampleRecord

AdapterFn = Callable[
    [Dict[str, Any], Path, Dict[str, Any]],
    Tuple[List[SampleRecord], List[PredictionRecord]],
]

_REGISTRY: Dict[str, AdapterFn] = {}
_BUILTINS_LOADED = False


def register_adapter(name: str, fn: AdapterFn) -> None:
    key = name.strip()
    if not key:
        raise ValueError("adapter name must be non-empty")
    _REGISTRY[key] = fn


def get_adapter(name: str) -> AdapterFn:
    ensure_builtin_adapters()
    key = (name or "").strip()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown dataset adapter {key!r}. Registered: {known}")
    return _REGISTRY[key]


def list_adapters() -> List[str]:
    ensure_builtin_adapters()
    return sorted(_REGISTRY)


def ensure_builtin_adapters() -> None:
    """Lazy import to avoid circular imports at package load."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from linguaeval.adapters.dataset import (
        belebele,
        jsonl_samples,
        lm_eval_samples,
        n2s_dialogue,
        native_mc,
    )

    register_adapter("jsonl", jsonl_samples.load_from_config)
    register_adapter("n2s_dialogue_prediction", n2s_dialogue.load_from_config)
    register_adapter("belebele_jsonl", belebele.load_from_config)
    register_adapter("indommlu_jsonl", native_mc.load_indommlu_from_config)
    register_adapter("copal_jsonl", native_mc.load_copal_from_config)
    register_adapter("lm_eval_samples", lm_eval_samples.load_from_config)
    # Preferred alias — same MC-only loader (generation not supported).
    register_adapter("lm_eval_mc_samples", lm_eval_samples.load_from_config)
    _BUILTINS_LOADED = True
