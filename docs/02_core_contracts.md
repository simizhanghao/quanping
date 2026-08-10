# Core Contracts v0.1 (Field Freeze)

对应代码：`src/linguaeval/core/schema.py`  
结构文档：`docs/01_project_structure.md`

## 1. SampleRecord

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| sample_id | str | yes | stable id |
| input.text | str\|null | one of text/messages | single-turn text |
| input.messages | list\|null | one of text/messages | chat messages |
| gold | object | yes | open dict; interpreted by TaskSpec |
| meta | object | no | language, domain, source, split… |
| conversation | object\|null | no | dialogue_id, turn_id, role, context_mode… |

## 2. TaskSpec

| Field | Type | Notes |
|-------|------|-------|
| name | str | |
| task_type | str | classification \| structured_multitask \| … |
| targets | list | name, type, path, labels?, condition? |

Target types (v0): `binary`, `multiclass`, `text`.

## 3. OutputSpec

| Field | Type | Notes |
|-------|------|-------|
| format | str | json \| text |
| parser | str | json \| identity |
| schema.required | list[str] | optional |
| constraints | object | e.g. no_markdown |

## 4. MetricSpec

| Field | Type | Notes |
|-------|------|-------|
| metrics | map target→list | e.g. n2s: [precision, recall, f1, f2] |
| round_digits | int\|null | N2S compat: 2 |
| exclude_format_fail | bool | default true for classification |

## 5. ModelSpec

| Field | Type | Notes |
|-------|------|-------|
| models.\<id\> | object | backend, model, sampling… |
| comparability_group | str\|null | for D3/D10 |

## 6. PredictionRecord

| Field | Type | Notes |
|-------|------|-------|
| sample_id | str | |
| model_id | str | |
| raw_output | str\|null | |
| parsed | object | |
| scores | object | for calibration later |
| format.parse_ok / schema_ok | bool | |
| usage / timing | object | |
| error | str\|null | |

## 7. ScoreRecord (P0.5-B)

Sample-level bridge between Prediction and Aggregator:

| Field | Notes |
|-------|------|
| sample_id / model_id | |
| targets.\<name\> | `{gold, pred, correct, applicable}` — names from TaskSpec |
| parse_ok / schema_ok | from OutputSpec pipeline |
| joint_success | all applicable targets correct |
| slices | optional diagnostics keys from meta/conversation |

## 8. Parse modes

| Mode | Behavior |
|------|----------|
| `from_parsed` | Validate existing `parsed` (legacy N2S replay) |
| `from_raw` | `raw_output` → Parser → Validator → `parsed` |

Config:

```yaml
parse:
  mode: from_raw   # or from_parsed
```

## 9. RunManifest

run_id, created_at, config_path, packs, artifact_index, notes, **provenance**.

### provenance (P0.5-C)

| Field | Notes |
|-------|------|
| git_sha | repo HEAD if available |
| config_hash / task_spec_hash / output_spec_hash / metric_spec_hash | file sha256 |
| dataset_fingerprint / prediction_fingerprint | canonical JSON sha256 of records |
| eligible_samples / prediction_rows | counts |
| adapter / parse_mode | from config |

## 10. Metric denominators (P0.5-C)

See `docs/03_metric_denominators.md`.

| Mode | Denominator | Format fail |
|------|-------------|-------------|
| semantic (default top-level) | format_ok only | excluded |
| strict | all with_prediction | counts as incorrect |

Artifacts: `data_audit.json`, `business_metrics.coverage`, `business_metrics.metrics_by_mode`.

## 11. ComparisonRecord (P1-A)

See `docs/04_paired_regression_p1a.md`.

Paired offline regression over **baseline** vs **candidate** (display may say Base/SFT).

| Field | Notes |
|-------|------|
| sample_id / target | single `compare.target` |
| applicable | from TaskSpec condition |
| baseline / candidate | `{pred, correct}` |
| transition | `stable_correct` \| `gain` \| `regression` \| `both_wrong` (eligible only) |

Alignment: default **strict** sample_id equality or run FAIL.

## 12. Comparability + protocol (P1-D)

See `docs/08_evaluation_semantics_p1d.md`.

| Flag | Meaning |
|------|---------|
| semantic_comparable | prompt/context/scoring/dataset protocols match → business compare OK |
| efficiency_comparable | backend/hardware/decoding match → latency compare OK |

Golden `comparison_protocol.allowed_pairs` rejects non-reference asset pairs (`NOT_COMPARABLE`).

Metric blocks may be `{status: NOT_APPLICABLE, reason: ...}` instead of fake `0.0`.  
Gates may return `INSUFFICIENT_SUPPORT` when `requirements.min_samples` / `min_clusters` unmet.

## 13. ConfidenceSpec / ConfidenceRecord (P1.5-A)

See `docs/09_confidence_contract_p15a.md`.

| Object | Notes |
|--------|------|
| ConfidenceSpec | `target` + `source.type/path` (+ optional `predicted_path`, `labels`) |
| ConfidenceRecord | `status` ∈ AVAILABLE \| NOT_AVAILABLE \| NOT_APPLICABLE；可选 `class_scores` + scalar `confidence` |

Prediction protocol 与 Confidence protocol **解耦**：自由生成 JSON 可以没有 scores。

## 14. Calibration metrics (P1.5-B)

See `docs/10_calibration_metrics_p15b.md`.

`compute_calibration_metrics(ConfidenceRecord[])` → `calibration_metrics.json`  
（ECE / Brier / NLL / AUROC OVR macro；无 score → pack `NOT_AVAILABLE`）。

## 15. Operating point (P1.5-C)

See `docs/11_operating_point_p15c.md`.

`OperatingPointSpec` + `operating-point-offline`：在 validation/calibration 上选阈值，在 test 上冻结评估；`optimize_on: test` → `TEST_LEAKAGE`。

## 16. Selective prediction (P1.5-D)

See `docs/12_selective_prediction_p15d.md`.

`SelectiveSpec` + `selective-offline`：Risk-Coverage / AURC / Risk@Coverage / Coverage@Risk。
