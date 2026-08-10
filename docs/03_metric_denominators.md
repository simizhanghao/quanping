# Metric Denominators — Coverage / Semantic / Strict

**Status:** P0.5-C frozen definitions  
**Code:** `src/linguaeval/metrics/denominators.py`

## Coverage

| Symbol | Meaning |
|--------|---------|
| eligible | all loaded `SampleRecord`s |
| with_prediction | samples aligned to a `PredictionRecord` |
| format_ok | `parse_ok AND schema_ok` |
| coverage_prediction | with_prediction / eligible |
| coverage_valid | format_ok / eligible |

## Semantic metric

Business metrics computed **only on format_ok samples**.

Must always report `support` / `denominator` — never a bare F1 without saying on how many rows.

Closest to “among successfully parsed outputs, how good is the model?”

## Strict metric

**All** `with_prediction` samples enter the denominator.

Format/schema failure ⇒ that sample counts as **incorrect** for applicable targets.

Closest to production: a crash/bad JSON is a failed business outcome.

## Example

```text
Semantic F1 / EM = 0.82   (denom = format_ok)
Coverage_valid   = 0.95
Strict F1 / EM   = 0.78   (denom = all predicted)
```

## Artifacts

Every offline run writes:

- `manifest.json` → `provenance` (hashes, git_sha, counts)
- `data_audit.json` → coverage + light distributions + fingerprint refs
- `business_metrics.json` → `coverage` + `metrics_by_mode.{semantic,strict}`

Top-level `targets` remain **semantic** for backward compatibility (`metric_mode_default: semantic`).
