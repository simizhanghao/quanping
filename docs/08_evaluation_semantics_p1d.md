# Evaluation Semantics Freeze — P1-D

**Code:** `compare/protocol.py`, `compare/applicability.py`, `compare/gates.py`

## 1. Golden comparison protocol

```yaml
comparison_protocol:
  protocol_id: bca_n2s_v1
  require_semantic_comparable: true
  allowed_pairs:
    - baseline_path_suffix: content_Indonesian_multi_skill_qwen3_4b_base_en.json
      candidate_path_suffix: qwen3_4b_test3.json
      note: historical Base↔SFT reference (3471)
```

Mismatch → `ComparisonProtocolError` / CLI exit 2 (`NOT_COMPARABLE`).

## 2. Comparability

```yaml
comparability:
  semantic:
    prompt_protocol: n2s_en_v1
    context_protocol: context_turn_3
    scoring_protocol: linguaeval_offline_v1
  baseline:
    backend_family: vllm
    decoding: {temperature: 0.0}
  candidate:
    backend_family: transformers
    decoding: {temperature: 0.0}
```

Outputs:

- `semantic_comparable` — business-effect compare allowed
- `efficiency_comparable` — latency/throughput compare allowed

## 3. Metric applicability

Binary slice with `positive_support=0`:

```json
{"f1": {"status": "NOT_APPLICABLE", "reason": "positive_support=0"}}
```

Prefer `accuracy` / `specificity` / `false_positive_rate` on negative-only slices.

## 4. Gate support policy

```yaml
gates:
  - id: delta_f1_ci_lower
    path: statistics.metrics.f1.delta.ci_low
    op: ">="
    value: 0
    requirements:
      min_samples: 500
      min_clusters: 50
```

Statuses: `PASS | FAIL | ERROR | INSUFFICIENT_SUPPORT | NOT_APPLICABLE`.
