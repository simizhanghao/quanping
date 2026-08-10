# Bootstrap Statistics — P1-B

**Status:** frozen for P1-B  
**Depends on:** P1-A paired compare  
**Code:** `src/linguaeval/compare/bootstrap.py`

## Goal

Upgrade point-estimate deltas to **statistically supported** deltas:

```text
ΔF1 = +0.48
95% CI = [lo, hi]
```

## Config

```yaml
statistics:
  enabled: true
  n_bootstrap: 1000
  confidence_level: 0.95
  seed: 42
  bootstrap_unit: sample          # ordinary paired bootstrap
  # bootstrap_unit: dialogue_id   # cluster bootstrap (N2S / multi-turn)
  # metrics: [f1, precision, recall]   # optional; else MetricSpec / primary
```

`bootstrap_unit` resolves from `SampleRecord.conversation.<key>` then `meta.<key>`,  
else falls back to `sample_id`. No business-specific defaults in Kernel.

## Cluster bootstrap

When multiple rows share a unit id:

1. Resample **units** with replacement (`n_units` draws)
2. Expand each drawn unit to **all member rows**
3. A unit drawn twice contributes its turns twice

This forbids treating turns inside one dialogue as independent samples.

Invariant for tests: cluster mode must not call per-row `randrange` for index selection.

## Artifacts

- `statistics.json`
- `comparison_metrics.json` → `statistics`
- `report.md` → CI lines

## Out of P1-B

Fixed slices, CI-aware gate (P1-C), IndoMMLU, robustness.
