from __future__ import annotations

import argparse
import sys
from pathlib import Path

from linguaeval.compare.alignment import AlignmentError
from linguaeval.compare.protocol import ComparisonProtocolError
from linguaeval.core.compare_runner import run_offline_compare
from linguaeval.core.confidence_runner import run_offline_confidence
from linguaeval.core.consistency_runner import run_offline_consistency
from linguaeval.core.context_runner import run_offline_context
from linguaeval.core.language_matrix_runner import run_offline_language_matrix
from linguaeval.core.language_runner import run_offline_language_inspect
from linguaeval.core.operating_point_runner import run_offline_operating_point
from linguaeval.core.perturb_runner import run_offline_perturb
from linguaeval.core.robustness_compare_runner import run_offline_robustness_compare
from linguaeval.core.robustness_runner import run_offline_robustness
from linguaeval.core.selective_runner import run_offline_selective
from linguaeval.core.runner import run_offline_score
from linguaeval.confidence.operating_point import OperatingPointError
from linguaeval.language.registry import LanguageRegistryError
from linguaeval.robustness.compare import VariantFingerprintError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="linguaeval", description="LinguaEval CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score-offline", help="Score existing predictions without model inference")
    p_score.add_argument("config", type=str, help="Path to offline YAML config")

    p_cmp = sub.add_parser(
        "compare-offline",
        help="Paired baseline vs candidate regression without model inference",
    )
    p_cmp.add_argument("config", type=str, help="Path to compare YAML config")

    p_conf = sub.add_parser(
        "confidence-offline",
        help="Extract confidence + calibration metrics (ECE/Brier/NLL/AUROC)",
    )
    p_conf.add_argument("config", type=str, help="Path to confidence YAML config")

    p_op = sub.add_parser(
        "operating-point-offline",
        help="Select threshold / operating point (P1.5-C; never optimize on test)",
    )
    p_op.add_argument("config", type=str, help="Path to operating-point YAML config")

    p_sel = sub.add_parser(
        "selective-offline",
        help="Selective prediction Risk-Coverage / AURC (P1.5-D)",
    )
    p_sel.add_argument("config", type=str, help="Path to selective YAML config")

    p_pert = sub.add_parser(
        "perturb-offline",
        help="Generate deterministic surface variants (P2-B; no model inference)",
    )
    p_pert.add_argument("config", type=str, help="Path to perturb YAML config")

    p_rob = sub.add_parser(
        "robustness-offline",
        help="Metamorphic robustness eval on clean+variant predictions (P2-A/B)",
    )
    p_rob.add_argument("config", type=str, help="Path to robustness YAML config")

    p_rcmp = sub.add_parser(
        "robustness-compare-offline",
        help="Paired baseline vs candidate robustness (shared variant_fingerprint; P2-D)",
    )
    p_rcmp.add_argument("config", type=str, help="Path to robustness-compare YAML config")

    p_cons = sub.add_parser(
        "consistency-offline",
        help="Self-consistency on repeated predictions (D8 / P2-E)",
    )
    p_cons.add_argument("config", type=str, help="Path to consistency YAML config")

    p_ctx = sub.add_parser(
        "context-offline",
        help="Context ablation with_context vs without_context (D6 / P2-E)",
    )
    p_ctx.add_argument("config", type=str, help="Path to context YAML config")

    p_lang = sub.add_parser(
        "language-inspect-offline",
        help="Inspect LanguagePack / Benchmark registry availability (D4 / P3-A)",
    )
    p_lang.add_argument("config", type=str, help="Path to language-pack YAML config")

    p_lmat = sub.add_parser(
        "language-matrix-offline",
        help="Cross-language Belebele-style score + Base↔SFT deltas (D4 / P3-B)",
    )
    p_lmat.add_argument("config", type=str, help="Path to language-matrix YAML config")

    args = parser.parse_args(argv)
    if args.command == "score-offline":
        out = run_offline_score(Path(args.config))
        print(f"[linguaeval] offline score written to: {out}")
        print(f"[linguaeval] see: {out / 'business_metrics.json'} and {out / 'report.md'}")
        return 0
    if args.command == "compare-offline":
        try:
            out = run_offline_compare(Path(args.config))
        except AlignmentError as e:
            print(f"[linguaeval] compare FAILED (alignment): {e}", file=sys.stderr)
            return 1
        except ComparisonProtocolError as e:
            print(f"[linguaeval] compare FAILED (NOT_COMPARABLE): {e}", file=sys.stderr)
            return 2
        print(f"[linguaeval] offline compare written to: {out}")
        print(f"[linguaeval] see: {out / 'comparison_metrics.json'} and {out / 'report.md'}")
        return 0
    if args.command == "confidence-offline":
        out = run_offline_confidence(Path(args.config))
        print(f"[linguaeval] offline confidence written to: {out}")
        print(
            f"[linguaeval] see: {out / 'calibration_metrics.json'}, "
            f"{out / 'confidence_audit.json'}, {out / 'report.md'}"
        )
        return 0
    if args.command == "operating-point-offline":
        try:
            out = run_offline_operating_point(Path(args.config))
        except OperatingPointError as e:
            print(f"[linguaeval] operating-point FAILED ({e.reason}): {e}", file=sys.stderr)
            return 2
        print(f"[linguaeval] offline operating-point written to: {out}")
        print(
            f"[linguaeval] see: {out / 'operating_points.json'}, "
            f"{out / 'threshold_curve.json'}, {out / 'report.md'}"
        )
        return 0
    if args.command == "selective-offline":
        out = run_offline_selective(Path(args.config))
        print(f"[linguaeval] offline selective written to: {out}")
        print(
            f"[linguaeval] see: {out / 'selective_metrics.json'}, "
            f"{out / 'risk_coverage_curve.json'}, {out / 'report.md'}"
        )
        return 0
    if args.command == "perturb-offline":
        out = run_offline_perturb(Path(args.config))
        print(f"[linguaeval] offline perturb written to: {out}")
        print(f"[linguaeval] see: {out / 'variants.jsonl'}, {out / 'variant_manifest.json'}")
        return 0
    if args.command == "robustness-offline":
        out = run_offline_robustness(Path(args.config))
        print(f"[linguaeval] offline robustness written to: {out}")
        print(
            f"[linguaeval] see: {out / 'robustness_metrics.json'}, "
            f"{out / 'robustness_records.jsonl'}, {out / 'report.md'}"
        )
        return 0
    if args.command == "robustness-compare-offline":
        try:
            out = run_offline_robustness_compare(Path(args.config))
        except VariantFingerprintError as e:
            print(f"[linguaeval] robustness-compare FAILED ({e.reason}): {e}", file=sys.stderr)
            return 2
        print(f"[linguaeval] offline robustness-compare written to: {out}")
        print(
            f"[linguaeval] see: {out / 'robustness_compare_metrics.json'}, "
            f"{out / 'robustness_compare_records.jsonl'}, {out / 'report.md'}"
        )
        return 0
    if args.command == "consistency-offline":
        out = run_offline_consistency(Path(args.config))
        print(f"[linguaeval] offline consistency written to: {out}")
        print(
            f"[linguaeval] see: {out / 'consistency_metrics.json'}, "
            f"{out / 'consistency_records.jsonl'}, {out / 'report.md'}"
        )
        return 0
    if args.command == "context-offline":
        out = run_offline_context(Path(args.config))
        print(f"[linguaeval] offline context written to: {out}")
        print(
            f"[linguaeval] see: {out / 'context_metrics.json'}, "
            f"{out / 'context_records.jsonl'}, {out / 'report.md'}"
        )
        return 0
    if args.command == "language-inspect-offline":
        try:
            out = run_offline_language_inspect(Path(args.config))
        except LanguageRegistryError as e:
            print(f"[linguaeval] language-inspect FAILED ({e.reason}): {e}", file=sys.stderr)
            return 2
        print(f"[linguaeval] offline language-inspect written to: {out}")
        print(
            f"[linguaeval] see: {out / 'language_pack_audit.json'}, {out / 'report.md'}"
        )
        return 0
    if args.command == "language-matrix-offline":
        out = run_offline_language_matrix(Path(args.config))
        print(f"[linguaeval] offline language-matrix written to: {out}")
        print(
            f"[linguaeval] see: {out / 'language_metrics.json'}, "
            f"{out / 'language_regression.json'}, "
            f"{out / 'language_capability_report.json'}, "
            f"{out / 'gate.json'}, {out / 'report.md'}"
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
