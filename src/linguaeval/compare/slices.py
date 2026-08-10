"""Fixed-slice paired comparison (P1-C). Specs are config-driven; no business ifs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from linguaeval.compare.bootstrap import PairRow, _metric_on_indices
from linguaeval.core.paths import get_by_path
from linguaeval.core.schema import SampleRecord, ScoreRecord
from linguaeval.metrics.classification import _round_maybe


def _input_text_len(sample: SampleRecord) -> int:
    text = (sample.input.text if sample.input else None) or ""
    if not text and sample.input and sample.input.messages:
        text = " ".join(str(m.get("content") or "") for m in sample.input.messages)
    return len(text)


def _bucket(value: float, edges: Sequence[float], labels: Optional[Sequence[str]]) -> str:
    """edges are ascending upper bounds, length = n_buckets; labels optional length n."""
    for i, upper in enumerate(edges):
        if value <= float(upper):
            if labels and i < len(labels):
                return str(labels[i])
            if i == 0:
                return f"<= {upper}"
            return f"({edges[i - 1]}, {upper}]"
    if labels and len(labels) >= len(edges):
        return str(labels[-1])
    return f"> {edges[-1]}"


def resolve_slice_key(
    sample: SampleRecord,
    score_b: ScoreRecord,
    score_c: ScoreRecord,
    *,
    target: str,
    source: str,
) -> str:
    """Resolve a slice key from a declarative source string."""
    src = (source or "").strip()
    if src in {"target.gold", "gold_of_target"}:
        ts = score_b.targets.get(target)
        v = None if ts is None else ts.gold
        return "null" if v is None else str(v)
    if src == "format.both_ok":
        ok = bool(
            score_b.parse_ok
            and score_b.schema_ok
            and score_c.parse_ok
            and score_c.schema_ok
        )
        return "true" if ok else "false"
    if src in {"input.text.length", "input_length"}:
        return str(_input_text_len(sample))
    if src.startswith("meta."):
        v = get_by_path({"meta": sample.meta or {}}, src, default=None)
        return "unknown" if v is None else str(v)
    if src.startswith("conversation."):
        v = get_by_path({"conversation": sample.conversation or {}}, src, default=None)
        return "unknown" if v is None else str(v)
    if src.startswith("gold."):
        v = get_by_path(sample.gold or {}, src[5:], default=None)
        return "unknown" if v is None else str(v)
    # bare meta / conversation keys
    meta = sample.meta or {}
    conv = sample.conversation or {}
    if src in meta:
        return str(meta[src])
    if src in conv:
        return str(conv[src])
    return "unknown"


def apply_buckets(raw_key: str, spec: Dict[str, Any]) -> str:
    edges = spec.get("buckets")
    if not edges:
        return raw_key
    try:
        val = float(raw_key)
    except (TypeError, ValueError):
        return raw_key
    return _bucket(val, list(edges), spec.get("labels"))


def build_slice_comparison(
    samples: List[SampleRecord],
    scores_b: List[ScoreRecord],
    scores_c: List[ScoreRecord],
    rows: List[PairRow],
    *,
    target: str,
    target_type: str,
    metric_names: Sequence[str],
    slice_specs: Sequence[Dict[str, Any]],
    labels: Optional[List[str]] = None,
    round_digits: Optional[int] = None,
    min_support: int = 1,
) -> Dict[str, Any]:
    """Compute baseline/candidate/delta metrics per fixed slice."""
    sample_map = {s.sample_id: s for s in samples}
    b_map = {s.sample_id: s for s in scores_b}
    c_map = {s.sample_id: s for s in scores_c}
    row_index = {r.sample_id: i for i, r in enumerate(rows)}

    out_slices: Dict[str, Any] = {}
    for spec in slice_specs:
        name = str(spec.get("name") or spec.get("source") or "slice")
        source = str(spec.get("source") or "")
        if not source:
            continue
        groups: Dict[str, List[int]] = defaultdict(list)
        for r in rows:
            s = sample_map.get(r.sample_id)
            sb = b_map.get(r.sample_id)
            sc = c_map.get(r.sample_id)
            if s is None or sb is None or sc is None:
                continue
            raw = resolve_slice_key(s, sb, sc, target=target, source=source)
            key = apply_buckets(raw, spec)
            groups[key].append(row_index[r.sample_id])

        block: Dict[str, Any] = {
            "source": source,
            "min_support": min_support,
            "values": {},
        }
        for key, idxs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            if len(idxs) < min_support:
                continue
            mb = _metric_on_indices(
                rows, idxs, side="baseline", target_type=target_type, metric_names=metric_names, labels=labels
            )
            mc = _metric_on_indices(
                rows, idxs, side="candidate", target_type=target_type, metric_names=metric_names, labels=labels
            )
            metrics: Dict[str, Any] = {}
            for m in metric_names:
                if m not in mb or m not in mc:
                    continue
                bv, cv = mb[m], mc[m]
                metrics[m] = {
                    "baseline": _round_maybe(bv, round_digits) if round_digits is not None else bv,
                    "candidate": _round_maybe(cv, round_digits) if round_digits is not None else cv,
                    "delta": _round_maybe(cv - bv, round_digits) if round_digits is not None else (cv - bv),
                }
            gains = sum(1 for i in idxs if rows[i].transition == "gain")
            regs = sum(1 for i in idxs if rows[i].transition == "regression")
            block["values"][key] = {
                "support": len(idxs),
                "metrics": metrics,
                "transitions": {
                    "gain": gains,
                    "regression": regs,
                    "net_gain": gains - regs,
                },
            }
        out_slices[name] = block
    return {"target": target, "slices": out_slices}
