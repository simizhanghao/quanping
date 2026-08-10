"""D4 Language contracts — LanguageSpec / BenchmarkSpec / LanguagePackSpec."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# Allowed provenance.translation values (P3-A hard enum; unknown → validation error)
TRANSLATION_TYPES = frozenset(
    {"native", "parallel", "human_translated", "machine_translated", "mixed", "unknown"}
)
PROVENANCE_ORIGINS = frozenset({"parallel", "native_authored", "translated", "mixed", "unknown"})


@dataclass
class LanguageSpec:
    """ISO-aware language identity. Prefer iso639_3; macrolanguage optional (e.g. ara)."""

    iso639_3: str
    name: str
    script: Optional[str] = None
    locale: Optional[str] = None
    macrolanguage: Optional[str] = None
    variant: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LanguageSpec":
        if not d or not d.get("iso639_3"):
            raise ValueError("LanguageSpec.iso639_3 is required")
        return cls(
            iso639_3=str(d["iso639_3"]).strip().lower(),
            name=str(d.get("name") or d["iso639_3"]),
            script=(str(d["script"]) if d.get("script") is not None else None),
            locale=(str(d["locale"]) if d.get("locale") is not None else None),
            macrolanguage=(
                str(d["macrolanguage"]).strip().lower() if d.get("macrolanguage") else None
            ),
            variant=(str(d["variant"]) if d.get("variant") is not None else None),
            meta=dict(d.get("meta") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSpec:
    """Benchmark as a plugin descriptor — not a score world of its own."""

    id: str
    capability: str
    task_type: str
    language: str  # iso639_3
    provenance: Dict[str, Any] = field(default_factory=dict)
    version: Optional[str] = None
    revision: Optional[str] = None
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    scorer: str = "deterministic"
    status: str = "AVAILABLE"  # AVAILABLE | NOT_AVAILABLE
    reason: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkSpec":
        if not d or not d.get("id"):
            raise ValueError("BenchmarkSpec.id is required")
        for req in ("capability", "task_type", "language"):
            if not d.get(req):
                raise ValueError(f"BenchmarkSpec.{req} is required")
        prov = dict(d.get("provenance") or {})
        origin = str(prov.get("origin") or "unknown")
        translation = str(prov.get("translation") or "unknown")
        if origin not in PROVENANCE_ORIGINS:
            raise ValueError(f"BenchmarkSpec.provenance.origin invalid: {origin!r}")
        if translation not in TRANSLATION_TYPES:
            raise ValueError(f"BenchmarkSpec.provenance.translation invalid: {translation!r}")
        status = str(d.get("status") or "AVAILABLE").upper()
        if status not in {"AVAILABLE", "NOT_AVAILABLE"}:
            raise ValueError(f"BenchmarkSpec.status must be AVAILABLE|NOT_AVAILABLE, got {status}")
        metrics = d.get("metrics") or d.get("metric") or ["accuracy"]
        if isinstance(metrics, str):
            metrics = [metrics]
        return cls(
            id=str(d["id"]),
            capability=str(d["capability"]),
            task_type=str(d["task_type"]),
            language=str(d["language"]).strip().lower(),
            provenance={
                "origin": origin,
                "translation": translation,
                "native_authored": bool(prov.get("native_authored", origin == "native_authored")),
                "source_language": prov.get("source_language"),
                "culture_sensitive": prov.get("culture_sensitive"),
                **{
                    k: v
                    for k, v in prov.items()
                    if k
                    not in {
                        "origin",
                        "translation",
                        "native_authored",
                        "source_language",
                        "culture_sensitive",
                    }
                },
            },
            version=(str(d["version"]) if d.get("version") is not None else None),
            revision=(str(d["revision"]) if d.get("revision") is not None else None),
            metrics=[str(x) for x in metrics],
            scorer=str(d.get("scorer") or "deterministic"),
            status=status,
            reason=d.get("reason"),
            meta=dict(d.get("meta") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_native_authored(self) -> bool:
        return bool(self.provenance.get("native_authored")) or self.provenance.get("origin") == "native_authored"

    @property
    def is_translated_or_parallel(self) -> bool:
        return str(self.provenance.get("translation")) in {
            "parallel",
            "human_translated",
            "machine_translated",
        } or str(self.provenance.get("origin")) in {"parallel", "translated"}


@dataclass
class LanguagePackSpec:
    """Capability → benchmark_id list. Pack is organization protocol, not a score."""

    id: str
    language: str  # iso639_3 code
    capabilities: Dict[str, List[str]] = field(default_factory=dict)
    version: str = "1"
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LanguagePackSpec":
        if not d or not d.get("id"):
            raise ValueError("LanguagePackSpec.id is required")
        lang = d.get("language")
        if isinstance(lang, dict):
            lang_code = str(lang.get("iso639_3") or "").strip().lower()
        else:
            lang_code = str(lang or "").strip().lower()
        if not lang_code:
            raise ValueError("LanguagePackSpec.language is required")
        caps_raw = dict(d.get("capabilities") or {})
        caps: Dict[str, List[str]] = {}
        for cap, items in caps_raw.items():
            if items is None:
                caps[str(cap)] = []
            elif isinstance(items, str):
                caps[str(cap)] = [items]
            else:
                caps[str(cap)] = [str(x) for x in items]
        return cls(
            id=str(d["id"]),
            language=lang_code,
            capabilities=caps,
            version=str(d.get("version") or "1"),
            meta=dict(d.get("meta") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def all_benchmark_ids(self) -> List[str]:
        out: List[str] = []
        for ids in self.capabilities.values():
            out.extend(ids)
        return out
