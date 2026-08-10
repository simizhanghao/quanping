# Toy Belebele-format parallel MC (P3-B)

Tiny fixtures only — **not** the full Belebele download.

Same `link` ids across `ind_Latn` / `arb_Arab`. Shared TaskSpec + `belebele_jsonl` adapter.

Known SFT deltas (accuracy):

```text
ind: base 6/8=0.75 → sft 8/8=1.0   Δ=+0.25
arb: base 5/8=0.625 → sft 4/8=0.5  Δ=-0.125
```

```bash
PYTHONPATH=src python -m linguaeval language-matrix-offline \
  configs/examples/22_language_matrix_belebele_toy.yaml
```
