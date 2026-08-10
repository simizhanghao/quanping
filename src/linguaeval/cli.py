from __future__ import annotations

import argparse
import sys
from pathlib import Path

from linguaeval.compare.alignment import AlignmentError
from linguaeval.compare.protocol import ComparisonProtocolError
from linguaeval.core.compare_runner import run_offline_compare
from linguaeval.core.confidence_runner import run_offline_confidence
from linguaeval.core.runner import run_offline_score


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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
