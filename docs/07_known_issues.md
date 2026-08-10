# Known Issues — P1 Freeze Notes

**Status:** P1-D semantics freeze  
**Date:** 2026-08-10

## Resolved by P1-D (guardrails)

| Issue | Guardrail |
|-------|-----------|
| Incomplete Base prediction paired with full SFT set | `comparison_protocol.allowed_pairs` + strict `sample_id` align → `NOT_COMPARABLE` / FAIL |
| Different backends mistaken for fair efficiency compare | `semantic_comparable` vs `efficiency_comparable` |
| Slice F1=0 on all-negative gold | Metric applicability → `NOT_APPLICABLE` (use accuracy / specificity / FPR) |
| Tiny-n CI gate false FAIL | Gate `requirements.min_samples` / `min_clusters` → `INSUFFICIENT_SUPPORT` |

## Open / deferred (not P1-D)

| Issue | Notes | Deferred to |
|-------|-------|-------------|
| One compare.target per run | Multi-target orchestration loops later | post-P1.5 |
| Joint success as first-class compare target | Derived target, not transition kernel change | later |
| Online inference | Still offline PredictionRecord in | P2+ / adapters |
| Calibration scores missing on free-gen N2S JSON | P1.5-A/B: extract + metrics `NOT_AVAILABLE` (expected) | P1.5-C/D curves |
| Public HTML dashboard | Explicit non-goal now | P4? |
| IndoMMLU / multilingual packs | Planned | P3 |
| Metamorphic robustness | Planned | P2 |

## Operational notes

- Golden N2S pair for release compare:
  - baseline: `content_Indonesian_multi_skill_qwen3_4b_base_en.json`
  - candidate: `qwen3_4b_test3.json`
  - **Do not** use `qwen3_4b_base_test3.json` as the golden baseline (partial run).
- Efficiency (latency) is **not** comparable for that pair (vLLM vs transformers); business metrics may still be `semantic_comparable=true` when prompt/context/scoring protocols match.
