#!/usr/bin/env python3
"""Prepare a real-data LanguagePack subset for P3-F-D (offline).

Downloads (optional) Belebele / IndoMMLU / COPAL-ID and writes LinguaEval JSONL
under ``data/language_pack_real/``.

IndoMMLU / COPAL are loaded from CSV files on the Hub (datasets>=5 no longer
runs legacy ``*.py`` loading scripts). Belebele still uses ``load_dataset``.

Does not invent scores. Predictions are produced separately by
``scripts/run_mc_offline_predict.py``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "language_pack_real"

_INDOMMLU_CSV = "https://huggingface.co/datasets/indolem/IndoMMLU/resolve/main/IndoMMLU.csv"
_COPAL_CSV = "https://huggingface.co/datasets/haryoaw/COPAL/resolve/main/test_copal.csv"

# Optional mirror (set HF_ENDPOINT=https://hf-mirror.com)
def _resolve_url(url: str) -> str:
    endpoint = (os.environ.get("HF_ENDPOINT") or "").rstrip("/")
    if endpoint and "huggingface.co" in url:
        return url.replace("https://huggingface.co", endpoint).replace(
            "http://huggingface.co", endpoint
        )
    return url


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _take(rows: List[Dict[str, Any]], *, n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0 or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [rows[i] for i in sorted(idx[:n])]


def _download_to_file(url: str, dest: Path, *, timeout: int = 600, retries: int = 4) -> Path:
    """Chunked download with retries; skips if dest already non-empty."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"[prepare] reuse cache {dest} ({dest.stat().st_size} bytes)")
        return dest

    url = _resolve_url(url)
    last_err: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        partial = dest.with_suffix(dest.suffix + f".part{attempt}")
        try:
            print(f"[prepare] download {url} → {dest} (try {attempt}/{retries})")
            req = urllib.request.Request(
                url,
                headers={
                    "Accept-Encoding": "identity",
                    "User-Agent": "linguaeval-prepare/0.1",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp, partial.open("wb") as out:
                while True:
                    chunk = resp.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    out.write(chunk)
            partial.replace(dest)
            print(f"[prepare] saved {dest} ({dest.stat().st_size} bytes)")
            return dest
        except BaseException as e:  # noqa: BLE001 — network errors vary
            last_err = e
            print(f"[prepare] download failed try {attempt}: {type(e).__name__}: {e}")
            if partial.is_file():
                partial.unlink(missing_ok=True)
            time.sleep(min(2 ** attempt, 20))
    raise TimeoutError(f"failed to download {url} after {retries} tries: {last_err}")


def _download_csv_rows(url: str, *, cache_path: Path) -> List[Dict[str, str]]:
    """Download a Hub CSV into cache_path, then parse."""
    path = _download_to_file(url, cache_path)
    text = path.read_text(encoding="utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _parse_indommlu_choices(jawaban: Any) -> Dict[str, str]:
    choices: Dict[str, str] = {}
    if jawaban is None:
        return choices
    for line in str(jawaban).replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-E])[.)\:]\s*(.*)$", line)
        if m:
            choices[m.group(1)] = m.group(2).strip()
    return choices


def belebele_hf_row_to_lingua(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map facebook/belebele row → adapter-ready JSONL object."""
    link = str(row.get("link") or row.get("id") or "")
    if not link:
        raise ValueError("belebele row missing link")
    return {
        "link": link,
        "flores_passage": row.get("flores_passage") or row.get("passage") or "",
        "question": row.get("question") or "",
        "mc_answer1": row.get("mc_answer1"),
        "mc_answer2": row.get("mc_answer2"),
        "mc_answer3": row.get("mc_answer3"),
        "mc_answer4": row.get("mc_answer4"),
        "correct_answer_num": str(row.get("correct_answer_num")),
    }


def indommlu_hf_row_to_lingua(row: Dict[str, Any], *, idx: int) -> Dict[str, Any]:
    """Map IndoMMLU CSV/HF row → indommlu_jsonl object (letter gold)."""
    sid = str(row.get("id") or row.get("sample_id") or row.get("nomor") or f"indommlu_{idx}")
    question = str(
        row.get("question") or row.get("soal") or row.get("prompt") or ""
    ).strip()
    choices = row.get("choices") or row.get("options")
    mapped: Dict[str, Any] = {
        "id": sid,
        "question": question,
        "subject": row.get("subject"),
    }
    if isinstance(choices, dict) and choices:
        mapped["choices"] = {str(k).upper(): v for k, v in choices.items()}
    elif isinstance(choices, list) and choices:
        letters = ["A", "B", "C", "D", "E"]
        mapped["choices"] = {letters[i]: choices[i] for i in range(min(len(choices), 5))}
    else:
        parsed = _parse_indommlu_choices(row.get("jawaban"))
        if parsed:
            mapped["choices"] = parsed
        else:
            for i, letter in enumerate(["A", "B", "C", "D"], start=1):
                key = f"option_{letter}"
                if row.get(key) is not None:
                    mapped.setdefault("choices", {})[letter] = row[key]
                elif row.get(f"mc_answer{i}") is not None:
                    mapped.setdefault("choices", {})[letter] = row[f"mc_answer{i}"]

    answer = row.get("answer")
    if answer is None:
        answer = row.get("kunci")
    if answer is None:
        answer = row.get("correct_answer")
    if isinstance(answer, str):
        a = answer.strip().upper()
        if a and a[0] in {"A", "B", "C", "D", "E"}:
            mapped["answer"] = a[0]
        else:
            mapped["answer"] = answer
    else:
        mapped["answer"] = answer
    return mapped


def copal_hf_row_to_lingua(row: Dict[str, Any], *, idx: int) -> Dict[str, Any]:
    sid = str(row.get("id") or row.get("sample_id") or row.get("idx") or f"copal_{idx}")
    label = row.get("label")
    if label is None:
        label = row.get("answer")
    if isinstance(label, str) and label.strip().isdigit():
        label = int(label.strip())
    return {
        "id": sid,
        "premise": row.get("premise") or row.get("context") or "",
        "choice1": row.get("choice1") or row.get("option_A") or "",
        "choice2": row.get("choice2") or row.get("option_B") or "",
        "label": label,
        "variant": row.get("variant") or row.get("dialect") or "standard",
    }


def _load_hf(path_or_name: str, *, name: Optional[str], split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit(
            "HuggingFace datasets is required for Belebele --from-hf. "
            "pip install datasets  OR place JSONL under --out and skip download."
        ) from e
    kwargs: Dict[str, Any] = {}
    if name:
        kwargs["name"] = name
    ds = load_dataset(path_or_name, split=split, **kwargs)
    return [dict(r) for r in ds]


def prepare_belebele(
    out_dir: Path,
    *,
    langs: Sequence[str],
    n: int,
    seed: int,
    from_hf: bool,
) -> Dict[str, Path]:
    written: Dict[str, Path] = {}
    for lang in langs:
        dest = out_dir / "belebele" / f"{lang}.jsonl"
        if from_hf:
            if dest.is_file() and sum(1 for _ in dest.open(encoding="utf-8") if _.strip()) >= n:
                print(f"[prepare] reuse existing {dest}")
            else:
                rows = _load_hf("facebook/belebele", name=lang, split="test")
                mapped = [belebele_hf_row_to_lingua(r) for r in rows]
                mapped = _take(mapped, n=n, seed=seed)
                _write_jsonl(dest, mapped)
        elif not dest.is_file():
            raise FileNotFoundError(
                f"missing {dest}; run with --from-hf or place Belebele JSONL there"
            )
        written[lang] = dest
    return written


def prepare_indommlu(out_dir: Path, *, n: int, seed: int, from_hf: bool) -> Path:
    dest = out_dir / "indommlu" / "samples.jsonl"
    if from_hf:
        print("[prepare] download IndoMMLU CSV …")
        cache = out_dir / "_cache" / "IndoMMLU.csv"
        raw = _download_csv_rows(_INDOMMLU_CSV, cache_path=cache)
        mapped = [indommlu_hf_row_to_lingua(r, idx=i) for i, r in enumerate(raw)]
        # Drop rows without parseable choices / letter key
        mapped = [
            r
            for r in mapped
            if r.get("question")
            and isinstance(r.get("choices"), dict)
            and r.get("choices")
            and str(r.get("answer") or "").upper()[:1] in {"A", "B", "C", "D", "E"}
        ]
        mapped = _take(mapped, n=n, seed=seed)
        _write_jsonl(dest, mapped)
        print(f"[prepare] IndoMMLU wrote {len(mapped)} rows → {dest}")
    elif not dest.is_file():
        raise FileNotFoundError(f"missing {dest}; run with --from-hf or place IndoMMLU JSONL")
    return dest


def prepare_copal(out_dir: Path, *, n: int, seed: int, from_hf: bool) -> Path:
    dest = out_dir / "copal_id" / "samples.jsonl"
    if from_hf:
        print("[prepare] download COPAL CSV …")
        cache = out_dir / "_cache" / "test_copal.csv"
        raw = _download_csv_rows(_COPAL_CSV, cache_path=cache)
        mapped = [copal_hf_row_to_lingua(r, idx=i) for i, r in enumerate(raw)]
        mapped = [
            r
            for r in mapped
            if r.get("premise") and r.get("choice1") is not None and r.get("label") is not None
        ]
        mapped = _take(mapped, n=n, seed=seed)
        _write_jsonl(dest, mapped)
        print(f"[prepare] COPAL wrote {len(mapped)} rows → {dest}")
    elif not dest.is_file():
        raise FileNotFoundError(f"missing {dest}; run with --from-hf or place COPAL-ID JSONL")
    return dest


def write_manifest(out_dir: Path, meta: Dict[str, Any]) -> Path:
    path = out_dir / "pack_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n", type=int, default=64, help="subset size per benchmark/lang")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--from-hf", action="store_true", help="download via HuggingFace")
    p.add_argument(
        "--belebele-langs",
        nargs="+",
        default=["ind_Latn", "eng_Latn"],
        help="Belebele config names (ISO-ish)",
    )
    p.add_argument("--skip-indommlu", action="store_true")
    p.add_argument("--skip-copal", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    out_dir = args.out if args.out.is_absolute() else (ROOT / args.out)
    belebele = prepare_belebele(
        out_dir,
        langs=args.belebele_langs,
        n=args.n,
        seed=args.seed,
        from_hf=args.from_hf,
    )
    paths: Dict[str, Any] = {"belebele": {k: str(v) for k, v in belebele.items()}}
    if not args.skip_indommlu:
        paths["indommlu"] = str(
            prepare_indommlu(out_dir, n=args.n, seed=args.seed, from_hf=args.from_hf)
        )
    if not args.skip_copal:
        paths["copal_id"] = str(
            prepare_copal(out_dir, n=args.n, seed=args.seed, from_hf=args.from_hf)
        )

    man = write_manifest(
        out_dir,
        {
            "pack_id": "language_pack_real_p3fd",
            "n_per_cell": args.n,
            "seed": args.seed,
            "from_hf": bool(args.from_hf),
            "paths": paths,
            "sources": {
                "belebele": "facebook/belebele",
                "indommlu": _INDOMMLU_CSV,
                "copal_id": _COPAL_CSV,
            },
            "prediction_layout": {
                "base": str(out_dir / "predictions" / "base"),
                "sft": str(out_dir / "predictions" / "sft"),
            },
            "note": "No scores invented here — run run_mc_offline_predict.py then language-matrix-offline.",
        },
    )
    print(f"[prepare_language_real_subset] wrote pack under {out_dir}")
    print(f"[prepare_language_real_subset] manifest: {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
