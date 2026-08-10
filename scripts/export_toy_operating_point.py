#!/usr/bin/env python3
"""Export toy operating-point JSONL into examples/toy_operating_point/."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from linguaeval.confidence.operating_point import build_toy_binary_operating_point_rows  # noqa: E402

OUT = ROOT / "examples" / "toy_operating_point"


def main() -> None:
    samples, preds = build_toy_binary_operating_point_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "samples.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in samples) + "\n",
        encoding="utf-8",
    )
    (OUT / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in preds) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(samples)} rows to {OUT}")


if __name__ == "__main__":
    main()
