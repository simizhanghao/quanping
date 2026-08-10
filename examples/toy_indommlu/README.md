# Toy Native Indonesian capability fixtures (P3-C)

Not full IndoMMLU / COPAL-ID downloads — format-compatible fixtures only.

| Capability | Adapter | Acc (base→sft) |
|------------|---------|----------------|
| local_knowledge | `indommlu_jsonl` | 0.5 → 1.0 (Δ+0.5) |
| cultural_reasoning | `copal_jsonl` | 0.75 → 1.0 (Δ+0.25) |

Provenance: `native_authored=true` (contrast Belebele parallel).

Required encodings:

```yaml
indommlu_jsonl: answer_encoding: letter
copal_jsonl:    answer_encoding: zero_based_index   # label 0/1 → A/B
```
