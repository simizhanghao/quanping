# LinguaEval

**Business-Aware Evaluation Harness for Multilingual SFT Models**

Repo directory name: `eval_factory` · Public mirror: `simizhanghao/quanping`

> Architecture is generic from day one; implementation converges.  
> N2S (Indonesian banking Need-to-Search) is Example-01, not the kernel.

## What it does today

Offline evaluation from existing predictions:

1. **Score** one model (`score-offline`)
2. **Compare** baseline vs candidate (`compare-offline`) with Gain/Regression
3. **Cluster bootstrap** CIs (e.g. `dialogue_id`)
4. **Fixed slices** + **CI-aware gates**
5. **P1-D semantics**: golden pair protocol, comparability flags, metric applicability, gate support policy

## Quick start

```bash
cd eval_factory
PYTHONPATH=src python -m linguaeval score-offline configs/examples/01_score_toy_multiclass.yaml
PYTHONPATH=src python -m linguaeval compare-offline configs/examples/05_compare_base_sft_toy.yaml
PYTHONPATH=src python -m pytest -q
```

## Supported (P0–P1)

| Capability | Status |
|------------|--------|
| Contracts: Sample / Task / Output / Metric / Prediction / Score / Manifest | Supported |
| Offline scoring + parse `from_raw` / `from_parsed` | Supported |
| Dataset adapter registry (jsonl, n2s_dialogue_prediction) | Supported |
| Provenance / coverage / semantic·strict denominators | Supported |
| Paired compare (baseline/candidate), transition cases | Supported |
| Bootstrap CI + cluster bootstrap | Supported |
| Fixed slices + gates | Supported |
| Golden protocol / comparability / metric applicability / gate min-support | Supported |
| ConfidenceSpec extract (`confidence-offline`) | Supported (P1.5-A) |
| Calibration metrics ECE / Brier / NLL / AUROC | Supported (P1.5-B) |
| Operating point / threshold (`operating-point-offline`) | Supported (P1.5-C) |
| Selective prediction / Risk-Coverage (`selective-offline`) | Supported (P1.5-D) |

**Examples:** `toy_multiclass`, `toy_extraction`, `toy_compare_intent`, `toy_calibration` (probs), `toy_operating_point` (threshold/selective), `indonesian_n2s` (reference; confidence usually NOT_AVAILABLE).

## Planned (not supported yet)

| Capability | Phase |
|------------|-------|
| Metamorphic robustness / consistency | P2 |
| Metamorphic robustness / consistency | P2 |
| Multilingual language packs (IndoMMLU etc.) | P3 |
| Production profiling + rich release cards | P4 |
| Online model inference in-core | later adapters |
| HTML dashboard / auto slice discovery | non-goals for now |

## Docs

| Doc | Topic |
|-----|-------|
| `docs/00_executive_brief.md` | Stakeholder brief |
| `docs/01_project_structure.md` | Structure + roadmap |
| `docs/02_core_contracts.md` | Field contracts |
| `docs/03_metric_denominators.md` | Coverage / semantic / strict |
| `docs/04_paired_regression_p1a.md` | Paired compare |
| `docs/05_bootstrap_statistics_p1b.md` | Bootstrap |
| `docs/06_slices_and_gates_p1c.md` | Slices + gates |
| `docs/07_known_issues.md` | Known issues |
| `docs/08_evaluation_semantics_p1d.md` | P1-D semantics freeze |
| `docs/09_confidence_contract_p15a.md` | P1.5-A confidence contract |
| `docs/10_calibration_metrics_p15b.md` | P1.5-B calibration metrics |
| `docs/11_operating_point_p15c.md` | P1.5-C operating point / threshold |
| `docs/12_selective_prediction_p15d.md` | P1.5-D selective / Risk-Coverage |

## Design hard rules

- Kernel has **no** business field name branches (`n2s` / banking only in adapters & example YAML).
- **P11:** every new Kernel capability must pass a **non-N2S** acceptance first.
- Formal compare requires identical `sample_id` sets (default) and optional golden `allowed_pairs`.
- Do not invent confidence scores; missing source → `NOT_AVAILABLE`.
