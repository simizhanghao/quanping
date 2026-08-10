"""Robustness package (P2 metamorphic reliability)."""

from linguaeval.robustness.aggregate import aggregate_robustness, build_robustness_records
from linguaeval.robustness.compare import VariantFingerprintError, compare_robustness_metrics
from linguaeval.robustness.registry import list_perturbations

__all__ = [
    "aggregate_robustness",
    "build_robustness_records",
    "compare_robustness_metrics",
    "list_perturbations",
    "VariantFingerprintError",
]
