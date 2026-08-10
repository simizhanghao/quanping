"""Deterministic multiclass OVR fixture for operating-point smoke."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main() -> None:
    # validation: 12 per class; test: 8 per class
    classes = ["refund", "shipping", "account"]
    samples = []
    preds = []

    def add(split: str, gold: str, scores: dict, i: int) -> None:
        sid = f"{split[:3]}_{gold[:3]}_{i:02d}"
        pred = max(scores, key=scores.get)
        samples.append(
            {
                "sample_id": sid,
                "input": {"text": f"toy {sid}"},
                "gold": {"intent_class": gold},
                "meta": {"split_role": split, "language": "en"},
            }
        )
        preds.append(
            {
                "sample_id": sid,
                "model_id": "toy_ovr",
                "parsed": {"intent_class": pred},
                "scores": {"intent_class": scores},
                "format": {"parse_ok": True, "schema_ok": True},
            }
        )

    i = 0
    for gold in classes:
        for k in range(12):
            # peak on gold class with varying confidence
            p = 0.55 + 0.03 * (k % 10)
            rest = (1.0 - p) / 2.0
            scores = {c: (p if c == gold else rest) for c in classes}
            add("validation", gold, scores, i)
            i += 1
    i = 0
    for gold in classes:
        for k in range(8):
            p = 0.50 + 0.04 * (k % 8)
            rest = (1.0 - p) / 2.0
            scores = {c: (p if c == gold else rest) for c in classes}
            add("test", gold, scores, i)
            i += 1

    (OUT / "samples.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in samples) + "\n",
        encoding="utf-8",
    )
    (OUT / "predictions.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in preds) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(samples)} samples to {OUT}")


if __name__ == "__main__":
    main()
