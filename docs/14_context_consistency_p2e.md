# P2-E Context (D6) + Consistency (D8)

**Principle (P11):** Kernel 不写死对话业务字段；`conversation.*` 只作可选元数据。

## Scope (this slice)

| Dimension | CLI | Offline meaning |
|-----------|-----|-----------------|
| **D8 Consistency** | `consistency-offline` | 同 `sample_id` 多条 replicate → pairwise / all_agree / majority_accuracy |
| **D6 Context** | `context-offline` | `without_context` vs `with_context` 消融 → Δacc / gain / flip |

Out of this slice: online ConversationRunner prompt assembly, live multi-turn decode.

## Consistency (D8)

```text
predictions.jsonl  may repeat sample_id (meta.replicate optional)
        ↓
ConsistencyRecord per sample
        ↓
pairwise_agreement_rate / all_agree_rate / majority_accuracy
```

Denominators: samples with `n_replicates < min_replicates` → excluded (`insufficient_replicates`).

## Context ablation (D6)

```text
samples (+ optional conversation.{dialogue_id,turn_id,context_mode})
  + predictions_without_context
  + predictions_with_context
        ↓
ScoreRecord reuse (D1)
        ↓
transitions: stable_correct | gain | regression | both_wrong
+ prediction_flip_rate
```

Roles are mode names, not Base/SFT.

## Artifacts

```text
# consistency
consistency_metrics.json
consistency_records.jsonl
report.md

# context
context_metrics.json
context_records.jsonl
context_gain_cases.jsonl
context_regression_cases.jsonl
report.md
```

## Toy acceptance

```bash
PYTHONPATH=src python -m linguaeval consistency-offline configs/examples/19_consistency_toy_intent.yaml
PYTHONPATH=src python -m linguaeval context-offline configs/examples/20_context_toy_intent.yaml
```

Expected:
- consistency: `all_agree_rate=0.75`, `pairwise_agreement_rate=0.75`
- context: `delta_accuracy=0.25`, `context_gain_rate=0.25`
