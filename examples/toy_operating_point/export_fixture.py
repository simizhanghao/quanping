"""Write deterministic toy binary operating-point JSONL (run from repo root)."""

from __future__ import annotations

import json
from pathlib import Path

from linguaeval.confidence.operating_point import build_toy_binary_operating_point_rows

HERE = Path(__file__).resolve().parent


def main() -> None:
    samples, preds = build_toy_binary_operating_point_rows()
    (HERE / "samples.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in samples) + "\n",
        encoding="utf-8",
    )
    (HERE / "predictions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in preds) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(samples)} samples -> {HERE}")


if __name__ == "__main__":
    main()
