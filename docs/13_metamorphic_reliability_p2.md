# P2 Metamorphic Reliability — Plan & Hard Rules

**Principle (P11):** Kernel 能力必须先过非 N2S 验收；业务只经 Adapter / Pack / 配置进入。

## Architecture

```text
SampleRecord
      ↓
PerturbationSpec (Registry impl + YAML runtime params)
      ↓
VariantRecord (validity / NO-OP / applicability)
      ↓
external inference
      ↓
ScoreRecord (reuse D1 scorer — not a second correctness world)
      ↓
MetamorphicRelation
      ↓
RobustnessRecord (+ transitions)
      ↓
MetricSpec aggregates + Flip/Violation (+ P1 bootstrap)
```

**Perturbation ≠ Robustness.** Business correctness comes from **ScoreRecord**; P2 only compares clean↔perturbed behavior.

## Slice plan

| Slice | Scope | Status |
|-------|--------|--------|
| **A** | Contract + Registry + invariance offline | ✅ |
| **B** | Deterministic surface `perturb-offline` | ✅ |
| **C0** | Semantics hardening (ScoreRecord / MetricSpec / NO-OP / params / cluster) | ✅ |
| **C** | Realistic: typo, code-switch, context distractor | ✅ (first batch) |
| **D** | Base↔SFT Robustness Regression (`variant_fingerprint`) | ✅ |
| **E** | D6 Context + D8 Consistency | ✅ |

## Offline CLI

```text
linguaeval perturb-offline              → variants.jsonl + variant_manifest.json
# external model
linguaeval robustness-offline           → robustness_metrics.json / records / violations
linguaeval robustness-compare-offline   → Base↔Candidate Δ + shared fingerprint gate
linguaeval consistency-offline          → D8 replicate agreement (P2-E)
linguaeval context-offline              → D6 with/without context ablation (P2-E)
```

See also `docs/14_context_consistency_p2e.md`.

## P2-D Robustness Compare

```text
linguaeval robustness-compare-offline <config.yaml>
```

Hard gate: baseline and candidate must share the **same** `variants.jsonl`
(`variant_fingerprint`). Roles are `baseline` / `candidate` (display may say Base/SFT).

Artifacts:

```text
robustness_compare_metrics.json   # Δ rates + transitions
robustness_compare_records.jsonl
robustness_gain_cases.jsonl
robustness_regression_cases.jsonl
both_fragile_cases.jsonl
stable_robust_cases.jsonl
baseline_robustness_metrics.json
candidate_robustness_metrics.json
report.md
```

Model-level transitions (on `relation_satisfied`):

```text
stable_robust | robustness_gain | robustness_regression | both_fragile
```

Δ rates are **candidate − baseline** (negative `flip_rate` = candidate more invariant).

## P2-C first batch

```text
typo              — severity edit budget + protected_tokens
code_switch       — lexicon / lexicon_path plugin data (not Kernel if/else)
context_distractor — allowlisted distractor phrases only
```

Deferred: colloquial / ASR (harder semantic validity).

## P2-C0 checklist (locked)

1. Correctness via `build_score_records` / ScoreRecord — not ad-hoc string equality in P2  
2. Clean/perturbed metrics via `MetricSpec` + `score_targets` — not accuracy-only  
3. YAML `severity` / `params` / `applies_to` fully consumed  
4. Applicability + **NO-OP** → `semantic_validity=NOT_APPLICABLE` (out of denominator)  
5. Per-variant validation (changed ⇒ AUTO_VALIDATED for deterministic surface; no global blind trust)  
6. Split `variant_all_correct_rate` vs `end_to_end_robust_success_rate`  
7. Bootstrap `cluster_path` configurable (reuse P1 `resample_indices`)

## Hard rules (P2 PR gate)

1. Kernel: no Indonesian / N2S / banking / ASR-business branches  
2. New perturbation: non-N2S smoke first  
3. Declare `semantic_policy` / applicability / severity / validation  
4. NO-OP not in formal denominator  
5. Base/Candidate robustness compare requires shared `variant_fingerprint` (P2-D)  
6. Do not reimplement scorer/bootstrap — reuse D0/D1/P1  
