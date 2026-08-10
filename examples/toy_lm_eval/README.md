# Toy lm-eval MC log_samples (P3-D / P3-F-C)

Offline fixture shaped like EleutherAI lm-evaluation-harness `--log_samples` dumps
for **multiple-choice** tasks only.

**No `lm_eval` package dependency** — LinguaEval only converts samples → PredictionRecord and re-scores.

Adapter names: `lm_eval_mc_samples` (preferred) or legacy `lm_eval_samples` (same MC loader).

Requires explicit:

```yaml
answer_encoding: letter | zero_based_index | one_based_index
```

```bash
PYTHONPATH=src python -m linguaeval score-offline \
  configs/examples/25_score_lm_eval_samples_toy.yaml
```

Expected accuracy after re-score: `0.75` (doc1 wrong).
