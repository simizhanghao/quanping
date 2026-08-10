"""Language package (D4 / P3-A)."""

from linguaeval.language.registry import (
    LanguageRegistryError,
    get_benchmark,
    get_language,
    get_pack,
    list_benchmarks,
    list_languages,
    list_packs,
    register_benchmark,
    register_language,
    register_pack,
    resolve_pack_availability,
)
from linguaeval.language.spec import BenchmarkSpec, LanguagePackSpec, LanguageSpec

__all__ = [
    "BenchmarkSpec",
    "LanguagePackSpec",
    "LanguageRegistryError",
    "LanguageSpec",
    "get_benchmark",
    "get_language",
    "get_pack",
    "list_benchmarks",
    "list_languages",
    "list_packs",
    "register_benchmark",
    "register_language",
    "register_pack",
    "resolve_pack_availability",
]
