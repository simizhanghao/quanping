# LinguaEval (eval_factory)

Business-aware evaluation harness for multilingual SFT models.

> Architecture is generic from day one; implementation converges by phase.  
> N2S is Example-01, not the kernel.

## Docs

| Doc | Purpose |
|-----|---------|
| [docs/00_executive_brief.md](docs/00_executive_brief.md) | Stakeholder / 立项口径 |
| [docs/01_project_structure.md](docs/01_project_structure.md) | Engineering structure |
| [docs/02_core_contracts.md](docs/02_core_contracts.md) | Field-level contracts |

## Supported vs Planned

### Supported (P0)

- Offline scoring kernel (`SampleRecord` → `PredictionRecord` → metrics)
- Binary / multiclass classification metrics
- Toy multiclass example (proves non-N2S binding)
- N2S dialogue prediction offline replay adapter (business F1 / format rate)

### Planned

- D0–D10 full packs (see structure doc)
- Online ModelAdapter (vLLM / HF / OpenAI-compatible)
- Robustness / calibration / language packs / release gate UI

## Quickstart (offline)

```bash
cd /data/hanchengcheng/hcc_1/eval_factory
pip install -e ".[dev]"

# Toy multiclass (P0 acceptance B)
python -m linguaeval score-offline configs/examples/01_score_toy_multiclass.yaml

# Metric swap without re-inference (P0 acceptance C)
python -m linguaeval score-offline configs/examples/02_score_toy_metric_swap.yaml

# N2S offline replay (P0 acceptance A) — set path to existing prediction JSON
python -m linguaeval score-offline configs/examples/03_score_n2s_offline_replay.yaml
```

## Design one-liner

**Not a HELM clone. Not an N2S scorer rewrite.**  
A reusable Base↔SFT regression evaluation protocol for business multilingual SFT.
