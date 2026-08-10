# Selective Prediction / Risk-Coverage — P1.5-D

**Status:** frozen for P1.5-D (closes P1.5)  
**CLI:** `python -m linguaeval selective-offline <config.yaml>`  
**Code:** `src/linguaeval/confidence/selective.py`

## Principle (P11)

主验收 = 非 N2S toy（复用 `toy_operating_point` test split，N=64）。  
无 confidence → `NOT_AVAILABLE`。不做 temperature scaling / 不伪造分数。

## Question answered

> 自动处理最有把握的 X% 请求，风险（错误率）是多少？其余 fallback。

## Metrics

| Metric | Meaning |
|--------|---------|
| Risk-Coverage curve | 按 confidence 降序接受 top-k；risk = 接受集错误率 |
| AURC | 曲线下面积（越低越好） |
| Risk@Coverage | 给定 coverage 的 risk |
| Coverage@Risk | 满足 risk≤目标的最大 coverage |

## Config

```yaml
selective:
  target: label
  evaluate_on: test   # test | validation | calibration | all
  coverage_targets: [0.5, 0.8, 0.9, 1.0]
  risk_targets: [0.05, 0.1, 0.2]
```

## Artifacts

```text
selective_metrics.json
risk_coverage_curve.json
confidence_records.jsonl
confidence_audit.json
manifest.json
report.md
```

## P1.5 complete

```text
A Confidence Contract
B Calibration / Discrimination
C Operating Point / Threshold
D Selective Prediction / Risk-Coverage
```

下一步路线：**P2 Robustness**（不要继续堆 calibration）。
