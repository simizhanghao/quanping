# Operating Point / Threshold Selection — P1.5-C

**Status:** frozen for P1.5-C  
**CLI:** `python -m linguaeval operating-point-offline <config.yaml>`  
**Code:** `src/linguaeval/confidence/operating_point.py`

## Principle (P11)

主验收 = **非 N2S toy binary**（`examples/toy_operating_point`，N=128，validation/test 各 64）。  
Toy 仅作 **functional validation**，不作模型质量结论。

自由生成无 score → `NOT_AVAILABLE`。  
**禁止** `optimize_on: test`（`TEST_LEAKAGE`，CLI exit 2）。

## Score

阈值使用 `P(positive_class)`（来自 `ConfidenceRecord.class_scores`），不是 top-label confidence。  
Multiclass v0 = **one-vs-rest**（指定 `positive_class`）。

## Modes

| mode | Meaning |
|------|---------|
| `best_f1` | argmax F1 |
| `best_fbeta` | argmax Fβ（`beta`） |
| `max_recall_at_precision` | max R s.t. P ≥ floor |
| `max_precision_at_recall` | max P s.t. R ≥ floor |

无可行点 → `NO_FEASIBLE_OPERATING_POINT`（不返回“最近”点）。

## Config

```yaml
operating_point:
  target: label
  positive_class: fraud
  optimize_on: validation   # validation | calibration — never test
  evaluate_on: test
  selection:
    mode: max_recall_at_precision
  constraint:
    precision:
      min: 0.90
```

Split via `sample.meta.split_role`.

## Artifacts

```text
operating_points.json
threshold_curve.json
confidence_records.jsonl
confidence_audit.json
manifest.json
report.md
```

## Acceptance

| ID | Check |
|----|-------|
| A | Toy 找到已知 threshold（0.48 @ P≥0.9, R=1.0） |
| B | Constraint 生效 |
| C | 无可行点 → `NO_FEASIBLE_OPERATING_POINT` |
| D | `optimize_on: test` → `TEST_LEAKAGE` |
| E | 无 confidence → `NOT_AVAILABLE` |
| F | 换 target 名仅改配置 |

## Out of P1.5-C

Temperature scaling / Risk-Coverage / selective prediction → **P1.5-D**。
