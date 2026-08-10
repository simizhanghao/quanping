# Toy Context Intent (D6 / P2-E)

8 dialogue turns. Without context: `d01`/`d02` wrong (acc=0.75).  
With context: both fixed (acc=1.0).

```text
delta_accuracy = 0.25
context_gain_rate = 0.25
prediction_flip_rate = 0.25
```

```bash
PYTHONPATH=src python -m linguaeval context-offline \
  configs/examples/20_context_toy_intent.yaml
```
