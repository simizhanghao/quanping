"""P3-A LanguagePack contract — non-N2S registry smoke."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from linguaeval.core.language_runner import load_language_ecosystem, run_offline_language_inspect
from linguaeval.language.registry import (
    LanguageRegistryError,
    clear_registries,
    get_language,
    get_pack,
    resolve_pack_availability,
)
from linguaeval.language.spec import BenchmarkSpec, LanguagePackSpec, LanguageSpec

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/examples/21_language_pack_inspect.yaml"


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registries()
    yield
    clear_registries()


def test_language_spec_arb_macrolanguage():
    lang = LanguageSpec.from_dict(
        {"iso639_3": "arb", "name": "Standard Arabic", "macrolanguage": "ara", "script": "Arab"}
    )
    assert lang.iso639_3 == "arb"
    assert lang.macrolanguage == "ara"


def test_benchmark_requires_provenance_enums():
    with pytest.raises(ValueError):
        BenchmarkSpec.from_dict(
            {
                "id": "x",
                "capability": "reading_comprehension",
                "task_type": "multiple_choice",
                "language": "ind",
                "provenance": {"origin": "nope", "translation": "native"},
            }
        )


def test_unknown_language_no_english_fallback():
    register = __import__("linguaeval.language.registry", fromlist=["register_language"]).register_language
    register(LanguageSpec(iso639_3="ind", name="Indonesian"))
    with pytest.raises(LanguageRegistryError) as ei:
        get_language("zzz")
    assert ei.value.reason == "unknown_language"
    assert "fallback" in str(ei.value).lower() or "Registered" in str(ei.value)


def test_pack_requires_registered_language():
    with pytest.raises(LanguageRegistryError) as ei:
        from linguaeval.language.registry import register_pack

        register_pack(LanguagePackSpec(id="x", language="ind", capabilities={}))
    assert ei.value.reason == "language_not_registered"


def test_ind_arb_fixtures_native_vs_parallel():
    load_language_ecosystem(CFG, reset=True)
    assert get_language("ind").iso639_3 == "ind"
    assert get_language("arb").macrolanguage == "ara"
    assert get_pack("ind_v1").language == "ind"
    assert get_pack("arb_v1").language == "arb"

    ind = resolve_pack_availability("ind_v1")
    culture = ind["capabilities"]["cultural_reasoning"][0]
    assert culture["status"] == "AVAILABLE"
    assert culture["native_authored"] is True

    reading = ind["capabilities"]["reading_comprehension"][0]
    assert reading["status"] == "AVAILABLE"
    assert reading["translated_or_parallel"] is True
    assert reading["native_authored"] is False

    knowledge = ind["capabilities"]["local_knowledge"][0]
    assert knowledge["native_authored"] is True

    arb = resolve_pack_availability("arb_v1")
    assert arb["capabilities"]["reading_comprehension"][0]["status"] == "AVAILABLE"


def test_unavailable_benchmark_never_fills_zero():
    from linguaeval.language.registry import register_benchmark, register_language, register_pack

    register_language(LanguageSpec(iso639_3="ind", name="Indonesian"))
    register_benchmark(
        BenchmarkSpec.from_dict(
            {
                "id": "stub_x",
                "capability": "cultural_reasoning",
                "task_type": "multiple_choice",
                "language": "ind",
                "provenance": {"origin": "native_authored", "translation": "native"},
                "status": "NOT_AVAILABLE",
                "reason": "fixture_not_wired",
            }
        )
    )
    register_pack(
        LanguagePackSpec(
            id="stub_pack",
            language="ind",
            capabilities={"cultural_reasoning": ["stub_x"]},
        )
    )
    row = resolve_pack_availability("stub_pack")["capabilities"]["cultural_reasoning"][0]
    assert row["status"] == "NOT_AVAILABLE"
    assert row["reason"] == "fixture_not_wired"
    assert "score" not in row
    assert row.get("accuracy") is None


def test_language_inspect_offline():
    out = run_offline_language_inspect(CFG)
    audit = json.loads((out / "language_pack_audit.json").read_text(encoding="utf-8"))
    assert "ind" in audit["languages"]
    assert "arb" in audit["languages"]
    assert "ind_v1" in audit["packs"]
    assert "arb_v1" in audit["packs"]
    probe = audit["unknown_language_probe"]
    assert probe["status"] == "NOT_AVAILABLE"
    assert probe["reason"] == "unknown_language"
    ind_block = next(p for p in audit["resolved_packs"] if p["pack_id"] == "ind_v1")
    culture = ind_block["capabilities"]["cultural_reasoning"][0]
    assert culture["status"] == "AVAILABLE"
    assert culture["native_authored"] is True
    assert "score" not in culture
