# Confidence Contract — P1.5-A

**Status:** frozen for P1.5-A  
**CLI:** `python -m linguaeval confidence-offline <config.yaml>`  
**Code:** `src/linguaeval/confidence/`

## Principle (P11)

> **任何进入 Kernel 的新能力，都必须至少通过一个非 N2S 任务验证；新增业务只能通过配置、Adapter 或 Pack 接入，不能要求修改 Kernel。**

换 target 名 / 语言 / 业务 / adapter → Kernel 代码不修改。

Calibration Kernel 只认识：

```text
target · gold · prediction · confidence source
```

Never: `n2s` / banking / routing hardcoding inside extractors。

## Objects

```text
ConfidenceSpec  →  ConfidenceExtractor  →  ConfidenceRecord
```

Decoupled from Prediction protocol: free generation may have **no** confidence.

## ConfidenceSpec

```yaml
confidence:
  target: intent_class
  source:
    type: probabilities   # logits | logprob_margin | none
    path: scores.intent_class
  predicted_path: $.intent_class   # optional; default TaskSpec path
  labels: [refund, shipping, account]
```

## Binary / Multiclass

同一表示：`gold class` · `predicted class` · `class_scores` · scalar `confidence`。

- multiclass：默认 `confidence = P(predicted)`，否则 `max P`
- binary：就是 `K=2` 的 multiclass，不单独开业务分支

## Statuses

| Status | Meaning |
|--------|---------|
| AVAILABLE | usable class scores + scalar confidence |
| NOT_AVAILABLE | source missing / unreadable (e.g. free-gen JSON) |
| NOT_APPLICABLE | target not in TaskSpec / unsupported source type |
| INSUFFICIENT_SUPPORT | **reserved for P1.5-B+ metrics**（样本过少等）；Contract 层不伪造 0 |

**Never** invent scores, model-verbalized confidence, top-k floors, or silent forced-prefix probs.

## Acceptance (P1.5-A)

| ID | Check |
|----|-------|
| A | Toy multiclass with full probs → mostly AVAILABLE |
| B | N2S free-gen → NOT_AVAILABLE, no crash |
| C | Kernel has no n2s/banking branches |

## Out of P1.5-A

ECE / Brier / NLL / AUROC → **P1.5-B**（见 `docs/10_calibration_metrics_p15b.md`）。  
Threshold sweep / temperature scaling / Risk-Coverage → P1.5-C/D。
