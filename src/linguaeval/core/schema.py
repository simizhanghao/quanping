"""Seven core contracts — Kernel speaks only these shapes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class SampleInput:
    text: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None


@dataclass
class SampleRecord:
    sample_id: str
    gold: Dict[str, Any]
    input: SampleInput = field(default_factory=SampleInput)
    meta: Dict[str, Any] = field(default_factory=dict)
    conversation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "input": asdict(self.input),
            "gold": self.gold,
            "meta": self.meta,
            "conversation": self.conversation,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SampleRecord":
        inp = d.get("input") or {}
        return cls(
            sample_id=str(d["sample_id"]),
            gold=dict(d.get("gold") or {}),
            input=SampleInput(
                text=inp.get("text"),
                messages=inp.get("messages"),
            ),
            meta=dict(d.get("meta") or {}),
            conversation=d.get("conversation"),
        )


@dataclass
class TargetSpec:
    name: str
    type: str  # binary | multiclass | text
    path: str  # JSON-path-like: $.field or field
    labels: Optional[List[str]] = None
    condition: Optional[Dict[str, Any]] = None


@dataclass
class TaskSpec:
    name: str
    task_type: str
    targets: List[TargetSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskSpec":
        targets = [
            TargetSpec(
                name=t["name"],
                type=t["type"],
                path=t.get("path", f"$.{t['name']}"),
                labels=t.get("labels"),
                condition=t.get("condition"),
            )
            for t in (d.get("targets") or [])
        ]
        return cls(name=d["name"], task_type=d.get("task_type", "classification"), targets=targets)


@dataclass
class OutputSpec:
    format: str = "json"
    parser: str = "json"
    schema: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OutputSpec":
        return cls(
            format=d.get("format", "json"),
            parser=d.get("parser", "json"),
            schema=dict(d.get("schema") or {}),
            constraints=dict(d.get("constraints") or {}),
        )


@dataclass
class MetricSpec:
    metrics: Dict[str, List[str]] = field(default_factory=dict)
    round_digits: Optional[int] = None
    exclude_format_fail: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricSpec":
        return cls(
            metrics={k: list(v) for k, v in (d.get("metrics") or {}).items()},
            round_digits=d.get("round_digits"),
            exclude_format_fail=bool(d.get("exclude_format_fail", True)),
        )


@dataclass
class ModelEndpoint:
    backend: str = "offline"
    model: str = ""
    template: Optional[str] = None
    dtype: Optional[str] = None
    sampling: Dict[str, Any] = field(default_factory=dict)
    max_tokens: Optional[int] = None
    concurrency: int = 1


@dataclass
class ModelSpec:
    models: Dict[str, ModelEndpoint] = field(default_factory=dict)
    comparability_group: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelSpec":
        models: Dict[str, ModelEndpoint] = {}
        for mid, cfg in (d.get("models") or {}).items():
            models[mid] = ModelEndpoint(
                backend=cfg.get("backend", "offline"),
                model=cfg.get("model", ""),
                template=cfg.get("template"),
                dtype=cfg.get("dtype"),
                sampling=dict(cfg.get("sampling") or {}),
                max_tokens=cfg.get("max_tokens"),
                concurrency=int(cfg.get("concurrency", 1)),
            )
        return cls(models=models, comparability_group=d.get("comparability_group"))


@dataclass
class FormatStatus:
    parse_ok: bool = True
    schema_ok: bool = True
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionRecord:
    sample_id: str
    model_id: str
    raw_output: Optional[str] = None
    parsed: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, Any] = field(default_factory=dict)
    format: FormatStatus = field(default_factory=FormatStatus)
    usage: Dict[str, Any] = field(default_factory=dict)
    timing: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "model_id": self.model_id,
            "raw_output": self.raw_output,
            "parsed": self.parsed,
            "scores": self.scores,
            "format": asdict(self.format),
            "usage": self.usage,
            "timing": self.timing,
            "error": self.error,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionRecord":
        fmt = d.get("format") or {}
        return cls(
            sample_id=str(d["sample_id"]),
            model_id=str(d.get("model_id", "default")),
            raw_output=d.get("raw_output"),
            parsed=dict(d.get("parsed") or {}),
            scores=dict(d.get("scores") or {}),
            format=FormatStatus(
                parse_ok=bool(fmt.get("parse_ok", True)),
                schema_ok=bool(fmt.get("schema_ok", True)),
                details=dict(fmt.get("details") or {}),
            ),
            usage=dict(d.get("usage") or {}),
            timing=dict(d.get("timing") or {}),
            error=d.get("error"),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class TargetScore:
    """Per-target sample-level score (generic; target name comes from TaskSpec)."""

    gold: Any = None
    pred: Any = None
    correct: Optional[bool] = None
    applicable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScoreRecord:
    """Sample-level evaluation record between Prediction and Aggregator."""

    sample_id: str
    model_id: str = "default"
    targets: Dict[str, TargetScore] = field(default_factory=dict)
    parse_ok: bool = True
    schema_ok: bool = True
    joint_success: Optional[bool] = None
    slices: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "model_id": self.model_id,
            "targets": {k: v.to_dict() for k, v in self.targets.items()},
            "parse_ok": self.parse_ok,
            "schema_ok": self.schema_ok,
            "joint_success": self.joint_success,
            "slices": self.slices,
            "meta": self.meta,
        }


@dataclass
class RunManifest:
    run_id: str
    created_at: str = field(default_factory=_utc_now_iso)
    config_path: Optional[str] = None
    packs: List[str] = field(default_factory=list)
    artifact_index: Dict[str, str] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
