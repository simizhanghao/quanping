# Paired Regression — P1-A (Point-estimate)

**Status:** frozen for P1-A  
**CLI:** `python -m linguaeval compare-offline <config.yaml>`  
**Code:** `src/linguaeval/compare/`

## Goal

Explain what changed between two offline prediction sets on the **same samples**:

```text
baseline PredictionRecord  +  candidate PredictionRecord
        → sample_id strict align
        → ComparisonRecord (single compare.target)
        → transition counts + metric deltas + case dumps
```

Roles are **`baseline` / `candidate`** (not hardcoded Base/SFT). Display names may say Base→SFT.

## Frozen decisions

| # | Decision |
|---|----------|
| Q1 | Single `compare.target` per run |
| Q2 | `denominator: semantic` default; `strict` optional |
| Q3 | Alignment policy **strict**: id set mismatch → FAIL |
| Q4 | N2S ref: `content_Indonesian_multi_skill_qwen3_4b_base_en.json` vs `qwen3_4b_test3.json`（同 3471 评测集；`qwen3_4b_base_test3.json` 为不完整 run，严格对齐会拒） |
| Q5 | Emit mini `metric_deltas` (point estimate only) |
| Q6 | Artifacts `05_` / `06_` |
| Q7 | `applicable=false` excluded from 4-cell; audit only |

**Out of P1-A (done in later slices):** bootstrap CI / cluster → `docs/05_bootstrap_statistics_p1b.md`（P1-B）；fixed slices + CI-aware gate（P1-C）；IndoMMLU.

## Denominator

- **semantic:** transition only if both sides `format_ok` and target `applicable`
- **strict:** all `applicable` rows; format fail already `correct=false` on ScoreRecord

Invariant:

```text
gain + regression + stable_correct + both_wrong == transition_eligible
```

## ComparisonRecord (minimal)

```json
{
  "sample_id": "t01",
  "target": "intent_class",
  "applicable": true,
  "baseline": {"pred": "refund", "correct": false},
  "candidate": {"pred": "shipping", "correct": true},
  "transition": "gain"
}
```

`transition ∈ {stable_correct, gain, regression, both_wrong}` when eligible;  
`not_applicable` / `excluded_format` for audit-only rows (optional on record).

## Artifacts

```text
manifest.json
comparison_metrics.json
comparison_records.jsonl
gain_cases.jsonl
regression_cases.jsonl
both_wrong_cases.jsonl
stable_correct_cases.jsonl   # optional dump; always counted
alignment_audit.json
report.md
```

## Acceptance (P1-A)

A. Toy known transition counts  
B. Target rename via YAML only (`intent_class`)  
C. N2S baseline/candidate P/R/F1 align with legacy replay  
D. Missing candidate id → FAIL  
E. applicable invariant (4-cell sum == applicable eligible)
