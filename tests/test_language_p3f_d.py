"""P3-F-D helpers: real-pack row mappers (no HF download in unit tests)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_prepare():
    path = ROOT / "scripts" / "prepare_language_real_subset.py"
    spec = importlib.util.spec_from_file_location("prepare_language_real_subset", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_belebele_mapper_keeps_one_indexed_num():
    mod = _load_prepare()
    row = mod.belebele_hf_row_to_lingua(
        {
            "link": "x",
            "flores_passage": "p",
            "question": "q",
            "mc_answer1": "a",
            "mc_answer2": "b",
            "mc_answer3": "c",
            "mc_answer4": "d",
            "correct_answer_num": 2,
        }
    )
    assert row["correct_answer_num"] == "2"
    assert row["link"] == "x"


def test_indommlu_list_choices_to_letters():
    mod = _load_prepare()
    row = mod.indommlu_hf_row_to_lingua(
        {"id": "1", "question": "Q?", "choices": ["a", "b", "c", "d"], "answer": "B"},
        idx=0,
    )
    assert row["choices"]["B"] == "b"
    assert row["answer"] == "B"


def test_indommlu_csv_soal_jawaban_kunci():
    mod = _load_prepare()
    row = mod.indommlu_hf_row_to_lingua(
        {
            "id": "0",
            "soal": "Hari raya Buddha?",
            "jawaban": "A. Galungan\nB. Waisak\nC. Nyepi\nD. Natal",
            "kunci": "B",
            "subject": "Sejarah",
        },
        idx=0,
    )
    assert row["question"].startswith("Hari")
    assert row["choices"]["B"] == "Waisak"
    assert row["answer"] == "B"


def test_copal_mapper_zero_based_label():
    mod = _load_prepare()
    row = mod.copal_hf_row_to_lingua(
        {"id": "c", "premise": "p", "choice1": "a", "choice2": "b", "label": 1},
        idx=0,
    )
    assert row["label"] == 1


def test_real_matrix_config_exists():
    cfg = ROOT / "configs/examples/27_language_capability_real_base_sft.yaml"
    text = cfg.read_text(encoding="utf-8")
    assert "eng_Latn" in text
    assert "indommlu" in text and "copal_id" in text
    assert "delta_ci_low" in text
    assert "results/27_language_capability_real_base_sft" in text
