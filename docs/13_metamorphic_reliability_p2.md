# P2 Metamorphic Reliability — Plan & Hard Rules

**Status:** P2-A in progress  
**Principle (P11):** Kernel 能力必须先过非 N2S 验收；业务只经 Adapter / Pack / 配置进入。

## Architecture

```text
Original Sample
      ↓
PerturbationSpec  (how to transform — Registry, not Kernel if/else)
      ↓
VariantRecord     (lineage: parent, seed, severity, validity)
      ↓
external inference (offline; ModelAdapter later)
      ↓
MetamorphicRelationSpec  (what must hold — invariance v0)
      ↓
RobustnessRecord
      ↓
Aggregate (+ reuse P1 bootstrap in P2-B+)
```

**Perturbation ≠ Robustness.** Robustness = perturbation + metamorphic relation.

## Slice plan

| Slice | Scope |
|-------|--------|
| **A** | Contract + Registry + invariance offline eval（本文件） |
| **B** | Deterministic surface: case / punctuation / whitespace + Flip/Violation/Δ + bootstrap |
| **C** | Realistic: typo/ASR, colloquial, code-switch, distractor + `semantic_validity` + severity |
| **D** | Robustness Regression（Base↔SFT，复用 P1 compare） |
| **E** | D6 Context + D8 Consistency → Reliability Trio |

## P2-A deliverables

```text
PerturbationSpec
VariantRecord
MetamorphicRelationSpec   # invariance implemented; directional reserved
RobustnessRecord
Perturbation Registry     # metadata; apply() deferred to P2-B
robustness-offline CLI
toy metamorphic intent (hand-authored variants; no LLM paraphrase)
```

**Out of P2-A:** case/punct/whitespace engines, typo/ASR, online inference, bootstrap CI, robustness regression.

## P2-B (done)

```text
case_lower / strip_punctuation / collapse_whitespace  apply()
perturb-offline → variants.jsonl + variant_fingerprint
robustness-offline bootstrap Flip/Violation CI (reuse P1 resample_indices)
```

**Still out:** typo/ASR/colloquial, online inference, robustness regression (P2-C/D).

## Hard pitfalls (locked)

1. **No perturbation-name branches in Kernel** — Registry only.  
2. **Do not assume all perturbations preserve semantics** — need relation + `semantic_validity`.  
3. **Never compare raw_output equality** — compare TaskSpec **targets**.  
4. **Not only ΔAccuracy/F1** — FlipRate + Metamorphic ViolationRate.  
5. **Reproducible randomness** — `seed` + `transform_version` in manifest.  
6. **Shared variant set for Base/SFT** — same `variant_fingerprint` (P2-D).  
7. **Invalid variants must not enter denominator** — report generated / valid / evaluated.  
8. **Deterministic first** — no LLM paraphrase/judge in P2-A/B.

## Offline CLI shape

```text
# P2-B+:
linguaeval perturb-offline   → variants.jsonl
# external model run
linguaeval robustness-offline → robustness_metrics.json / records / violations
```

P2-A ships **robustness-offline** only（variants 可手写）。

## Validity gate

Only `semantic_validity ∈ {VERIFIED, AUTO_VALIDATED}` enter formal robustness denominators.  
`UNVERIFIED` / `INVALID` → coverage audit, excluded from Flip/Violation.
