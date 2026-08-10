from __future__ import annotations

import argparse
from pathlib import Path

from linguaeval.core.runner import run_offline_score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="linguaeval", description="LinguaEval CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score-offline", help="Score existing predictions without model inference")
    p_score.add_argument("config", type=str, help="Path to offline YAML config")

    args = parser.parse_args(argv)
    if args.command == "score-offline":
        out = run_offline_score(Path(args.config))
        print(f"[linguaeval] offline score written to: {out}")
        print(f"[linguaeval] see: {out / 'business_metrics.json'} and {out / 'report.md'}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
