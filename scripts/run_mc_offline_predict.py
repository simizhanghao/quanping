#!/usr/bin/env python3
"""Offline MC prediction for LanguagePack real subset (P3-F-D).

Loads a local HF CausalLM, scores Belebele / IndoMMLU / COPAL-format JSONL via
LinguaEval adapters (prompt text from SampleRecord.input.text), greedily
decodes a short continuation, extracts A–E, writes PredictionRecord JSONL.

No lm_eval dependency. Requires: torch, transformers.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]


def _extract_letter(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.strip()
    m = re.search(r"\b([A-E])\b", s.upper())
    if m:
        return m.group(1)
    for ch in s.upper():
        if ch in {"A", "B", "C", "D", "E"}:
            return ch
    return None


def _load_samples(
    adapter: str,
    samples_path: Path,
    *,
    language: str,
    benchmark_id: str,
    answer_encoding: Optional[str],
) -> List[Any]:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from linguaeval.adapters.dataset.registry import get_adapter

    source: Dict[str, Any] = {
        "adapter": adapter,
        "samples": str(samples_path),
        "language": language,
        "benchmark_id": benchmark_id,
    }
    if answer_encoding:
        source["answer_encoding"] = answer_encoding
    samples, _ = get_adapter(adapter)(source, samples_path.parent, {})
    return samples


def _build_prompt(sample_text: str) -> str:
    return (
        "You are answering a multiple-choice question. "
        "Reply with only one letter: A, B, C, D, or E.\n\n"
        f"{sample_text}\n\nAnswer:"
    )


def run_predict(
    *,
    model_path: Path,
    samples: Sequence[Any],
    model_id: str,
    max_new_tokens: int,
    device: str,
) -> List[Dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map="auto" if device == "auto" else None,
        trust_remote_code=True,
    )
    if device not in {"auto", "cpu"} and not str(getattr(model, "device", "")).startswith("cuda"):
        model.to(device)
    model.eval()

    rows: List[Dict[str, Any]] = []
    for i, s in enumerate(samples):
        prompt = _build_prompt(s.input.text or "")
        inputs = tok(prompt, return_tensors="pt")
        if device == "auto":
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif device != "cpu":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        letter = _extract_letter(gen) or "A"
        rows.append(
            {
                "sample_id": s.sample_id,
                "model_id": model_id,
                "raw_output": gen,
                "parsed": {"answer": letter},
                "format": {"parse_ok": True, "schema_ok": bool(letter)},
                "meta": {"predict_backend": "transformers_greedy_mc"},
            }
        )
        if (i + 1) % 10 == 0 or (i + 1) == len(samples):
            print(f"[run_mc_offline_predict] {i + 1}/{len(samples)}", flush=True)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--model-id", type=str, required=True)
    p.add_argument("--adapter", type=str, required=True, choices=["belebele_jsonl", "indommlu_jsonl", "copal_jsonl"])
    p.add_argument("--samples", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--language", type=str, required=True)
    p.add_argument("--benchmark-id", type=str, required=True)
    p.add_argument("--answer-encoding", type=str, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args(list(argv) if argv is not None else None)

    samples_path = args.samples if args.samples.is_absolute() else (ROOT / args.samples)
    out_path = args.out if args.out.is_absolute() else (ROOT / args.out)
    model_path = args.model if args.model.is_absolute() else Path(args.model)

    samples = _load_samples(
        args.adapter,
        samples_path,
        language=args.language,
        benchmark_id=args.benchmark_id,
        answer_encoding=args.answer_encoding,
    )
    print(f"[run_mc_offline_predict] n_samples={len(samples)} model={model_path}")
    rows = run_predict(
        model_path=model_path,
        samples=samples,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[run_mc_offline_predict] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
