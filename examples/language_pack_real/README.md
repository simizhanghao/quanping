# LanguagePack real subset (P3-F-D)

Purpose: move from toy fixtures to a **statistically supported** Base↔SFT
language × capability report on real multilingual benchmarks.

## Matrix (3+1)

| Capability | Data | Role |
|------------|------|------|
| Reading | Belebele `ind_Latn` | Indonesian parallel reading |
| Knowledge | IndoMMLU | Native Indonesian knowledge |
| Culture | COPAL-ID | Native cultural reasoning |
| Control | Belebele `eng_Latn` | Detect Indonesian-business SFT damage |

No multilingual total score.

## Layout

```text
data/language_pack_real/
  belebele/{ind_Latn,eng_Latn}.jsonl
  indommlu/samples.jsonl
  copal_id/samples.jsonl
  pack_manifest.json
  predictions/
    base/{belebele_ind_Latn,belebele_eng_Latn,indommlu,copal_id}.jsonl
    sft/...
```

## Pipeline

1. Prepare subset (HF download, default n=64):

```bash
cd /data/hanchengcheng/hcc_1/eval_factory
python scripts/prepare_language_real_subset.py --from-hf --n 64 --seed 42
```

2. Predict Base then SFT (needs GPU + transformers):

```bash
BASE=/data/hanchengcheng/hcc_1/LlamaFactory/models/Qwen3-4B
SFT=/data/hanchengcheng/hcc_1/LlamaFactory/saves/qwen3_4b/full/n2s_0729
PRED=data/language_pack_real/predictions
```

See `scripts/run_language_real_p3fd.sh` for the full predict + matrix command.

3. Score / regress with P1 paired kernel:

```bash
PYTHONPATH=src python -m linguaeval language-matrix-offline \
  configs/examples/27_language_capability_real_base_sft.yaml
```

Artifacts: `results/27_language_capability_real_base_sft/`  
(`language_capability_report.json`, `language_regression.json`, `gate.json`, `report.md`)

## Encodings

| Adapter | `answer_encoding` |
|---------|-------------------|
| belebele_jsonl | (official one-indexed `correct_answer_num`) |
| indommlu_jsonl | `letter` |
| copal_jsonl | `zero_based_index` |
