# Toy Consistency Intent (D8 / P2-E)

4 samples × 3 replicates. `c01–c03` fully agree; `c04` fully disagrees.

```text
all_agree_rate = 0.75
pairwise_agreement_rate = 0.75
majority_accuracy = 1.0   # c04 majority tie → first-seen refund
```

```bash
PYTHONPATH=src python -m linguaeval consistency-offline \
  configs/examples/19_consistency_toy_intent.yaml
```
