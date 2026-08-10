# Core Contracts v0.1 (Field Freeze)

对应代码：`src/linguaeval/core/schema.py`

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

## 7. RunManifest

run_id, created_at, config_path, packs, artifact_index, notes.
