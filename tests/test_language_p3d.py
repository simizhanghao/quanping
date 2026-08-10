"""P3-D lm-eval MC log_samples adapter — no lm_eval package required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linguaeval.adapters.dataset.answer_encoding import AnswerEncodingError
from linguaeval.adapters.dataset.lm_eval_samples import (
    load_from_config,
    load_lm_eval_samples_blob,
    sample_to_records,
)
from linguaeval.adapters.dataset.registry import list_adapters
from linguaeval.core.runner import run_offline_score

ROOT = Path(__file__).resolve().parents[1]


def test_lm_eval_adapter_registered():
    ids = list_adapters()
    assert "lm_eval_samples" in ids
    assert "lm_eval_mc_samples" in ids


def test_sample_mapping_letter_and_index_target():
    s, p = sample_to_records(
        {
            "doc_id": 9,
            "doc": {"question": "q"},
            "target": 2,
            "arguments": [["PROMPT"]],
            "filtered_resps": "C",
        },
        language="ind",
        benchmark_id="t",
        task_name="toy",
        model_id="m",
        capability="local_knowledge",
        sample_index=0,
        answer_encoding="zero_based_index",
    )
    assert s.gold["answer"] == "C"
    assert p.parsed["answer"] == "C"
    assert s.input.text == "PROMPT"
    assert s.meta["adapter_kind"] == "multiple_choice"
    assert s.meta["provenance"]["executor"] == "lm-eval"


def test_load_wrapped_samples_json():
    rows = load_lm_eval_samples_blob(
        ROOT / "examples/toy_lm_eval/samples_toy_mc_ind.json",
        task_name="toy_mc_ind",
    )
    assert len(rows) == 4


def test_adapter_requires_answer_encoding():
    with pytest.raises(AnswerEncodingError, match="answer_encoding"):
        load_from_config(
            {
                "adapter": "lm_eval_mc_samples",
                "task_name": "toy_mc_ind",
                "language": "ind",
                "samples": str(ROOT / "examples/toy_lm_eval/samples_toy_mc_ind.json"),
            },
            ROOT / "configs/examples",
            {},
        )


def test_adapter_and_score_offline_rescores():
    samples, preds = load_from_config(
        {
            "adapter": "lm_eval_mc_samples",
            "task_name": "toy_mc_ind",
            "language": "ind",
            "answer_encoding": "zero_based_index",
            "samples": str(ROOT / "examples/toy_lm_eval/samples_toy_mc_ind.json"),
        },
        ROOT / "configs/examples",
        {},
    )
    assert len(samples) == len(preds) == 4
    # lm-eval dump claimed acc=0 on doc1; LinguaEval must recompute from gold/pred
    out = run_offline_score(ROOT / "configs/examples/25_score_lm_eval_samples_toy.yaml")
    biz = json.loads((out / "business_metrics.json").read_text(encoding="utf-8"))
    assert biz["targets"]["answer"]["accuracy"] == pytest.approx(0.75)
    assert (out / "scores.jsonl").is_file()
