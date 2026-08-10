"""P2-C realistic perturbations: typo / code_switch / context_distractor.

Lexicons and distractor allowlists come from PerturbationSpec.params (or lexicon_path),
never hard-coded business/language names in Kernel branches.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from linguaeval.core.schema import PerturbationSpec, SampleInput
from linguaeval.robustness.transforms import _get_text, _set_text

_SEVERITY_EDIT_RATIO = {1: 0.01, 2: 0.03, 3: 0.05}
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _edit_ratio(spec: PerturbationSpec) -> float:
    if "edit_ratio" in (spec.params or {}):
        return float(spec.params["edit_ratio"])
    return float(_SEVERITY_EDIT_RATIO.get(int(spec.severity or 1), 0.01))


def _protected_token_set(spec: PerturbationSpec) -> Set[str]:
    raw = (spec.params or {}).get("protected_spans") or (spec.params or {}).get("protected_tokens") or []
    return {str(x).lower() for x in raw}


def _seeded_rng(spec: PerturbationSpec, fallback_seed: int = 42) -> random.Random:
    seed = spec.seed if spec.seed is not None else fallback_seed
    id_mix = sum(ord(c) for c in str(spec.id))
    return random.Random(int(seed) * 1009 + int(spec.severity or 1) * 17 + id_mix)


def apply_typo(inp: SampleInput, spec: PerturbationSpec) -> SampleInput:
    """Character-level edits with edit budget; skip protected tokens."""
    text = _get_text(inp)
    if not text:
        return inp
    rng = _seeded_rng(spec)
    protected = _protected_token_set(spec)
    ratio = _edit_ratio(spec)
    chars = list(text)
    # candidate indices: alphanumeric not inside protected tokens (approx via token spans)
    spans: List[tuple] = []
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.lower() in protected:
            continue
        if tok.isalnum() and len(tok) >= 1:
            spans.append((m.start(), m.end()))
    editable: List[int] = []
    for a, b in spans:
        editable.extend(range(a, b))
    if not editable:
        return inp
    n_edits = max(1, int(round(len(text) * ratio)))
    n_edits = min(n_edits, len(editable))
    positions = rng.sample(editable, n_edits)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for pos in positions:
        ch = chars[pos]
        if not ch.isalpha():
            continue
        # substitute with different letter, preserve case
        pool = alphabet.upper() if ch.isupper() else alphabet
        choices = [c for c in pool if c != ch]
        if not choices:
            continue
        chars[pos] = rng.choice(choices)
    return _set_text(inp, "".join(chars))


def _load_lexicon(spec: PerturbationSpec) -> Dict[str, str]:
    params = spec.params or {}
    if isinstance(params.get("lexicon"), dict):
        return {str(k).lower(): str(v) for k, v in params["lexicon"].items()}
    path = params.get("lexicon_path")
    if path:
        p = Path(str(path))
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k).lower(): str(v) for k, v in data.items()}
    return {}


def apply_code_switch(inp: SampleInput, spec: PerturbationSpec) -> SampleInput:
    """Replace tokens using a lexicon from params / lexicon_path (Language Plugin data)."""
    text = _get_text(inp)
    lexicon = _load_lexicon(spec)
    if not text or not lexicon:
        return inp
    rng = _seeded_rng(spec)
    max_swaps = int(params_max_swaps(spec))
    swapped = 0

    def repl(match: re.Match) -> str:
        nonlocal swapped
        tok = match.group(0)
        key = tok.lower()
        if key in lexicon and swapped < max_swaps:
            # optional probabilistic skip still deterministic via rng order
            if rng.random() <= float((spec.params or {}).get("swap_prob", 1.0)):
                swapped += 1
                out = lexicon[key]
                if tok.isupper():
                    return out.upper()
                if tok[:1].isupper():
                    return out[:1].upper() + out[1:]
                return out
        return tok

    new_text = re.sub(r"[A-Za-z]+", repl, text)
    return _set_text(inp, new_text)


def params_max_swaps(spec: PerturbationSpec) -> int:
    if "max_swaps" in (spec.params or {}):
        return max(1, int(spec.params["max_swaps"]))
    # severity 1/2/3 → 1/2/3 swaps
    return max(1, int(spec.severity or 1))


def apply_context_distractor(inp: SampleInput, spec: PerturbationSpec) -> SampleInput:
    """Prepend/append an allowlisted distractor phrase (must not change task answer)."""
    text = _get_text(inp)
    params = spec.params or {}
    allow = list(params.get("distractors") or [])
    if not allow:
        return inp
    rng = _seeded_rng(spec)
    phrase = str(rng.choice(allow))
    position = str(params.get("position") or "prefix").lower()
    if position == "suffix":
        new_text = f"{text} {phrase}".strip() if text else phrase
    else:
        new_text = f"{phrase} {text}".strip() if text else phrase
    return _set_text(inp, new_text)
