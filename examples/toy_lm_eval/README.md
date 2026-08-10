# Toy lm-eval log_samples (P3-D)

Offline fixture shaped like EleutherAI lm-evaluation-harness `--log_samples` dumps.
**No `lm_eval` package dependency** — LinguaEval only converts samples → PredictionRecord and re-scores.

```bash
PYTHONPATH=src python -m linguaeval score-offline \
  configs/examples/25_score_lm_eval_samples_toy.yaml
```

Expected accuracy after re-score: `0.75` (doc1 wrong).
