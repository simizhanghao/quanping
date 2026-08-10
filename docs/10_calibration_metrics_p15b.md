# Calibration Metrics — P1.5-B

**Status:** frozen for P1.5-B  
**CLI:** `python -m linguaeval confidence-offline <config.yaml>`  
**Code:** `src/linguaeval/confidence/metrics.py`  
**Depends on:** P1.5-A `ConfidenceRecord`

## Principle (P11)

主验收必须是 **非 N2S**（`examples/toy_calibration`）。  
自由生成无 score → pack `NOT_AVAILABLE`（正确）。禁止伪造概率。

## Inputs

仅使用 `status=AVAILABLE` 且具备 `gold` / `class_scores` / `confidence` 的记录。

## Metrics

| Key | Definition |
|-----|------------|
| `ece` | Top-label ECE，等宽 bins（默认 10） |
| `brier` | Multiclass Brier：`mean_i Σ_k (p_ik − y_ik)²` |
| `nll` | Mean NLL：`−mean log p(gold)`（ε clip） |
| `auroc_ovr_macro` | One-vs-rest ROC-AUC macro（纯 Python） |
| `accuracy` | Top-1 accuracy（诊断用） |

Binary = multiclass with K=2；无单独业务分支。

## Statuses

| Status | When |
|--------|------|
| AVAILABLE | 可算且 `n_usable >= min_samples`（pack / ECE / AUROC） |
| NOT_AVAILABLE | 无可用 confidence |
| NOT_APPLICABLE | AUROC 因单类 support 无定义 |
| INSUFFICIENT_SUPPORT | `n_usable < min_samples`（pack / ECE / AUROC）；Brier/NLL 仍可为 AVAILABLE |

## Config

```yaml
calibration:
  n_bins: 10
  min_samples: 10
```

## Artifacts

```text
calibration_metrics.json
confidence_records.jsonl
confidence_audit.json
report.md
```

## Out of P1.5-B

Threshold / operating point → **P1.5-C**（见 `docs/11_operating_point_p15c.md`）。  
Selective prediction / Risk-Coverage → P1.5-D。

**Note:** toy calibration N=16 is for **functional validation** only — not model-quality conclusions.
