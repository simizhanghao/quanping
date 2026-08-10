# Fixed Slices + CI-aware Gates — P1-C

**Status:** frozen for P1-C  
**Depends on:** P1-A compare + P1-B statistics  
**Code:** `src/linguaeval/compare/slices.py`, `gates.py`

## Goal

1. Explain **where** baseline→candidate changed (fixed slices only).  
2. Make a **release decision** with declarative gates, including CI lower bounds.

No automatic slice discovery.

## Slices (config-driven)

```yaml
slices:
  enabled: true
  min_support: 20
  metrics: [f1, recall]   # optional; else statistics/MetricSpec/primary
  specs:
    - name: language
      source: meta.language
    - name: gold_label
      source: target.gold
    - name: input_length
      source: input.text.length
      buckets: [50, 150, 500, 100000]
      labels: [short, medium, long, xlong]
    - name: turn_bucket
      source: conversation.turn_id
      buckets: [3, 10, 30, 100000]
      labels: [early, mid, late, very_late]
    - name: format_both_ok
      source: format.both_ok
    - name: routing_skill   # example only — any gold.* path
      source: gold.routing_skill
```

Supported `source` prefixes: `meta.*`, `conversation.*`, `gold.*`, plus  
`target.gold`, `input.text.length`, `format.both_ok`.

Kernel never hardcodes skill/banking/n2s — only path resolution.

Artifact: `slice_comparison.json`.

## Gates

```yaml
gates:
  - id: candidate_primary_min
    path: candidate_business.primary.value
    op: ">="
    value: 0.75

  - id: delta_f1_min
    path: metric_deltas.metrics.f1.delta
    op: ">="
    value: 0

  - id: delta_f1_ci_lower
    path: statistics.metrics.f1.delta.ci_low
    op: ">="
    value: 0
```

Ops: `>= <= > < == !=`  
Overall: any ERROR → ERROR; else any FAIL → FAIL; else PASS.

Artifact: `gate.json`.

## Out of P1-C

IndoMMLU, robustness, calibration, HTML dashboards.
