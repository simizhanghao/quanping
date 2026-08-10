# Toy Operating Point (P1.5-C)

**Purpose:** functional validation of threshold selection (N=128; 64 val + 64 test).  
**Not** for claiming real-model calibration/operating-point quality.

Known optimum for `max_recall_at_precision` with `precision.min=0.90`:

```text
threshold = 0.48
precision >= 0.90
recall = 1.0
```

```bash
PYTHONPATH=src python -m linguaeval operating-point-offline \
  configs/examples/09_operating_point_toy_binary.yaml
```
