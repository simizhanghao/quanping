"""Deterministic classification scorers bound to TaskSpec target types."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from linguaeval.core.paths import condition_holds, get_by_path
from linguaeval.core.schema import MetricSpec, PredictionRecord, SampleRecord, TaskSpec, TargetSpec


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no", ""}:
        return False
    return None


def _round_maybe(x: float, digits: Optional[int]) -> float:
    if digits is None:
        return float(x)
    return round(float(x), digits)


def binary_confusion(golds: List[bool], preds: List[bool]) -> Dict[str, int]:
    tp = tn = fp = fn = 0
    for g, p in zip(golds, preds):
        if g and p:
            tp += 1
        elif (not g) and (not p):
            tn += 1
        elif (not g) and p:
            fp += 1
        else:
            fn += 1
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def metrics_from_confusion(
    cm: Dict[str, int],
    wanted: List[str],
    round_digits: Optional[int] = None,
    beta: float = 1.0,
) -> Dict[str, Any]:
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    n = tp + tn + fp + fn
    positive = tp + fn
    predicted = tp + fp
    precision = (tp / predicted) if predicted else 0.0
    recall = (tp / positive) if positive else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    b2 = beta * beta
    if (b2 * precision + recall) > 0:
        fb = (1 + b2) * precision * recall / (b2 * precision + recall)
    else:
        fb = 0.0
    accuracy = (tp + tn) / n if n else 0.0

    table = {
        "precision": _round_maybe(precision, round_digits),
        "recall": _round_maybe(recall, round_digits),
        "f1": _round_maybe(f1, round_digits),
        "f2": _round_maybe(fb if abs(beta - 2.0) < 1e-9 else (
            (1 + 4) * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0
        ), round_digits),
        "accuracy": _round_maybe(accuracy, round_digits),
        "confusion_matrix": cm,
        "support": n,
        "positive_support": positive,
    }
    # recompute f2 properly with beta=2
    table["f2"] = _round_maybe(
        ((1 + 4) * precision * recall / (4 * precision + recall)) if (4 * precision + recall) > 0 else 0.0,
        round_digits,
    )

    out: Dict[str, Any] = {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "support": n}
    for name in wanted:
        key = name.lower()
        if key in table:
            out[key] = table[key]
        elif key == "confusion_matrix":
            out["confusion_matrix"] = cm
    # always keep core confusion for debugging
    if "confusion_matrix" not in out:
        out["confusion_matrix"] = cm
    return out


def multiclass_metrics(
    golds: List[str],
    preds: List[str],
    wanted: List[str],
    labels: Optional[List[str]] = None,
    round_digits: Optional[int] = None,
) -> Dict[str, Any]:
    n = len(golds)
    correct = sum(1 for g, p in zip(golds, preds) if g == p)
    accuracy = correct / n if n else 0.0
    label_set = labels or sorted(set(golds) | set(preds))
    per_class: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    for lab in label_set:
        tp = sum(1 for g, p in zip(golds, preds) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(golds, preds) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(golds, preds) if g == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lab] = {
            "precision": _round_maybe(prec, round_digits),
            "recall": _round_maybe(rec, round_digits),
            "f1": _round_maybe(f1, round_digits),
            "support": tp + fn,
        }
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    table = {
        "accuracy": _round_maybe(accuracy, round_digits),
        "macro_f1": _round_maybe(macro_f1, round_digits),
        "per_class": per_class,
        "support": n,
    }
    out: Dict[str, Any] = {"support": n}
    for name in wanted:
        key = name.lower()
        if key in table:
            out[key] = table[key]
    # Always expose accuracy/macro_f1 when computed; keep per_class only if requested.
    out.setdefault("accuracy", table["accuracy"])
    out.setdefault("macro_f1", table["macro_f1"])
    return out


def _pair_values(
    samples: List[SampleRecord],
    preds: List[PredictionRecord],
    target: TargetSpec,
    metric_spec: MetricSpec,
) -> Tuple[List[Any], List[Any], int, int]:
    by_id = {p.sample_id: p for p in preds}
    golds: List[Any] = []
    pred_vals: List[Any] = []
    skipped_missing = 0
    skipped_format = 0
    for s in samples:
        p = by_id.get(s.sample_id)
        if p is None:
            skipped_missing += 1
            continue
        if metric_spec.exclude_format_fail and not (p.format.parse_ok and p.format.schema_ok):
            skipped_format += 1
            continue
        # condition on gold (task semantics)
        if not condition_holds(s.gold, target.condition):
            continue
        g = get_by_path(s.gold, target.path, default=None)
        v = get_by_path(p.parsed, target.path, default=None)
        golds.append(g)
        pred_vals.append(v)
    return golds, pred_vals, skipped_missing, skipped_format


def score_targets(
    samples: List[SampleRecord],
    preds: List[PredictionRecord],
    task: TaskSpec,
    metric_spec: MetricSpec,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "task": task.name,
        "task_type": task.task_type,
        "targets": {},
        "schema": {
            "eval_sample_count": 0,
            "format_ok_count": 0,
            "format_fail_count": 0,
            "format_match_rate": None,
        },
    }

    by_id = {p.sample_id: p for p in preds}
    aligned = [s for s in samples if s.sample_id in by_id]
    format_ok = sum(
        1
        for s in aligned
        if by_id[s.sample_id].format.parse_ok and by_id[s.sample_id].format.schema_ok
    )
    format_fail = len(aligned) - format_ok
    result["schema"] = {
        "eval_sample_count": len(aligned),
        "format_ok_count": format_ok,
        "format_fail_count": format_fail,
        "format_match_rate": (format_ok / len(aligned)) if aligned else None,
    }

    for target in task.targets:
        wanted = metric_spec.metrics.get(target.name, [])
        if not wanted:
            continue
        golds, pred_vals, miss, fmt_skip = _pair_values(samples, preds, target, metric_spec)
        block: Dict[str, Any] = {
            "type": target.type,
            "path": target.path,
            "skipped_missing_pred": miss,
            "skipped_format_fail": fmt_skip,
        }
        if target.type == "binary":
            g_bool: List[bool] = []
            p_bool: List[bool] = []
            for g, p in zip(golds, pred_vals):
                gb = _as_bool(g)
                pb = _as_bool(p)
                if gb is None or pb is None:
                    continue
                g_bool.append(gb)
                p_bool.append(pb)
            cm = binary_confusion(g_bool, p_bool)
            block.update(metrics_from_confusion(cm, wanted, metric_spec.round_digits))
        elif target.type == "multiclass":
            g_str = ["" if g is None else str(g) for g in golds]
            p_str = ["" if p is None else str(p) for p in pred_vals]
            block.update(
                multiclass_metrics(
                    g_str,
                    p_str,
                    wanted,
                    labels=target.labels,
                    round_digits=metric_spec.round_digits,
                )
            )
        else:
            # text exact match stub
            n = len(golds)
            exact = sum(1 for g, p in zip(golds, pred_vals) if g == p)
            block["exact_match"] = _round_maybe(exact / n if n else 0.0, metric_spec.round_digits)
            block["support"] = n
        result["targets"][target.name] = block

    # joint: all required targets exact (binary/multiclass equality)
    if "joint" in metric_spec.metrics and task.targets:
        joint_ok = 0
        joint_n = 0
        for s in aligned:
            p = by_id[s.sample_id]
            if metric_spec.exclude_format_fail and not (p.format.parse_ok and p.format.schema_ok):
                continue
            ok = True
            applicable = False
            for target in task.targets:
                if not condition_holds(s.gold, target.condition):
                    continue
                applicable = True
                g = get_by_path(s.gold, target.path, default=None)
                v = get_by_path(p.parsed, target.path, default=None)
                if target.type == "binary":
                    if _as_bool(g) != _as_bool(v):
                        ok = False
                        break
                else:
                    if ("" if g is None else str(g)) != ("" if v is None else str(v)):
                        ok = False
                        break
            if not applicable:
                continue
            joint_n += 1
            if ok:
                joint_ok += 1
        rate = joint_ok / joint_n if joint_n else 0.0
        result["joint"] = {
            "exact_joint_success": _round_maybe(rate, metric_spec.round_digits),
            "support": joint_n,
        }

    return result
