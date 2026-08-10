from __future__ import annotations

import argparse
import sys
from pathlib import Path

from linguaeval.compare.alignment import AlignmentError
from linguaeval.compare.protocol import ComparisonProtocolError
from linguaeval.core.compare_runner import run_offline_compare
from linguaeval.core.confidence_runner import run_offline_confidence
from linguaeval.core.operating_point_runner import run_offline_operating_point
from linguaeval.core.selective_runner import run_offline_selective
from linguaeval.core.runner import run_offline_score
from linguaeval.confidence.operating_point import OperatingPointError


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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
