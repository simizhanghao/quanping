"""Language / Benchmark / Pack registries — no language if/else in Kernel scorers."""

from __future__ import annotations

from typing import Any, Dict, List

from linguaeval.language.spec import BenchmarkSpec, LanguagePackSpec, LanguageSpec

_LANGUAGES: Dict[str, LanguageSpec] = {}
_BENCHMARKS: Dict[str, BenchmarkSpec] = {}
_PACKS: Dict[str, LanguagePackSpec] = {}


class LanguageRegistryError(KeyError):
    """Unknown / unavailable language, pack, or benchmark registration error."""

    def __init__(self, message: str, *, reason: str = "unknown"):
        super().__init__(message)
        self.reason = reason


def clear_registries() -> None:
    """Test helper — empty all registries."""
    _LANGUAGES.clear()
    _BENCHMARKS.clear()
    _PACKS.clear()


def register_language(spec: LanguageSpec) -> None:
    _LANGUAGES[spec.iso639_3] = spec


def register_benchmark(spec: BenchmarkSpec) -> None:
    _BENCHMARKS[spec.id] = spec


def register_pack(spec: LanguagePackSpec) -> None:
    if spec.language not in _LANGUAGES:
        raise LanguageRegistryError(
            f"LanguagePack {spec.id!r} references unregistered language {spec.language!r}",
            reason="language_not_registered",
        )
    _PACKS[spec.id] = spec


def get_language(iso639_3: str) -> LanguageSpec:
    key = (iso639_3 or "").strip().lower()
    if key not in _LANGUAGES:
        known = ", ".join(sorted(_LANGUAGES)) or "(none)"
        raise LanguageRegistryError(
            f"Unknown language {key!r}. Registered: {known}. "
            "No English fallback.",
            reason="unknown_language",
        )
    return _LANGUAGES[key]


def get_benchmark(benchmark_id: str) -> BenchmarkSpec:
    if benchmark_id not in _BENCHMARKS:
        known = ", ".join(sorted(_BENCHMARKS)) or "(none)"
        raise LanguageRegistryError(
            f"Unknown benchmark {benchmark_id!r}. Registered: {known}",
            reason="unknown_benchmark",
        )
    return _BENCHMARKS[benchmark_id]


def get_pack(pack_id: str) -> LanguagePackSpec:
    if pack_id not in _PACKS:
        known = ", ".join(sorted(_PACKS)) or "(none)"
        raise LanguageRegistryError(
            f"Unknown language pack {pack_id!r}. Registered: {known}",
            reason="unknown_pack",
        )
    return _PACKS[pack_id]


def list_languages() -> List[str]:
    return sorted(_LANGUAGES)


def list_benchmarks() -> List[str]:
    return sorted(_BENCHMARKS)


def list_packs() -> List[str]:
    return sorted(_PACKS)


def resolve_pack_availability(pack_id: str) -> Dict[str, Any]:
    """Resolve pack → language + per-benchmark availability (never invent 0 scores)."""
    pack = get_pack(pack_id)
    lang = get_language(pack.language)
    by_cap: Dict[str, list] = {}
    for cap, bids in pack.capabilities.items():
        rows = []
        for bid in bids:
            try:
                b = get_benchmark(bid)
            except LanguageRegistryError as e:
                rows.append(
                    {
                        "benchmark_id": bid,
                        "status": "NOT_AVAILABLE",
                        "reason": e.reason,
                    }
                )
                continue
            if b.language != pack.language:
                rows.append(
                    {
                        "benchmark_id": bid,
                        "status": "NOT_AVAILABLE",
                        "reason": "language_mismatch",
                        "benchmark_language": b.language,
                        "pack_language": pack.language,
                    }
                )
                continue
            rows.append(
                {
                    "benchmark_id": bid,
                    "status": b.status,
                    "reason": b.reason,
                    "capability": b.capability,
                    "task_type": b.task_type,
                    "provenance": b.provenance,
                    "version": b.version,
                    "revision": b.revision,
                    "metrics": b.metrics,
                    "native_authored": b.is_native_authored,
                    "translated_or_parallel": b.is_translated_or_parallel,
                }
            )
        by_cap[cap] = rows
    return {
        "pack_id": pack.id,
        "language": lang.to_dict(),
        "capabilities": by_cap,
        "status": "AVAILABLE",
    }
