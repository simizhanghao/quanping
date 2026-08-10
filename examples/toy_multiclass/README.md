# Example-02: Toy Multiclass Intent

Proves the Kernel is **not** N2S-bound: plain `text → intent_class` classification.

32 smoke samples. Two intentional mistakes (`t14`, `t25`) so accuracy < 1.0.

Target field is deliberately named `intent_class` (not `label` / `n2s`) to prove
primary metrics come from YAML `report.primary_target`, not hardcoded names.
