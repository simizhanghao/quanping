# Toy Metamorphic Intent (P2-A)

Hand-authored `case_lower` variants (transform engine lands in P2-B).

**Known rates (8 VERIFIED pairs; 1 INVALID excluded):**

```text
flip_rate = 0.25
metamorphic_violation_rate = 0.25
accuracy_clean = 1.0
accuracy_perturbed = 0.75
delta_accuracy = -0.25
robust_success_rate = 0.75
```

```bash
PYTHONPATH=src python -m linguaeval robustness-offline \
  configs/examples/15_robustness_toy_invariance.yaml
```
