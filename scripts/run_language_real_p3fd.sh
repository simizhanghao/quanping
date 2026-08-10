#!/usr/bin/env bash
# P3-F-D end-to-end: prepare (optional) → predict base/sft → language matrix.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
BASE="${BASE:-/data/hanchengcheng/hcc_1/LlamaFactory/models/Qwen3-4B}"
SFT="${SFT:-/data/hanchengcheng/hcc_1/LlamaFactory/saves/qwen3_4b/full/n2s_0729}"
N="${N:-64}"
DATA="${DATA:-data/language_pack_real}"
PRED="$DATA/predictions"
DEVICE="${DEVICE:-auto}"

# Prefer already-prepared Belebele; always refresh IndoMMLU/COPAL via CSV when PREPARE=1
if [[ "${PREPARE:-1}" == "1" ]]; then
  "$PYTHON" scripts/prepare_language_real_subset.py --out "$DATA" --from-hf --n "$N" --seed 42
fi

predict_cell () {
  local side="$1" model="$2" model_id="$3" adapter="$4" samples="$5" lang="$6" bid="$7" out="$8" enc="${9:-}"
  local enc_args=()
  if [[ -n "$enc" ]]; then
    enc_args=(--answer-encoding "$enc")
  fi
  "$PYTHON" scripts/run_mc_offline_predict.py \
    --model "$model" --model-id "$model_id" \
    --adapter "$adapter" --samples "$samples" \
    --language "$lang" --benchmark-id "$bid" \
    --out "$out" --device "$DEVICE" \
    "${enc_args[@]}"
}

for SIDE in base sft; do
  if [[ "$SIDE" == "base" ]]; then
    MODEL="$BASE"; MID="qwen3_4b_base"
  else
    MODEL="$SFT"; MID="qwen3_4b_n2s_sft"
  fi
  mkdir -p "$PRED/$SIDE"
  predict_cell "$SIDE" "$MODEL" "$MID" belebele_jsonl \
    "$DATA/belebele/ind_Latn.jsonl" ind belebele_ind_Latn \
    "$PRED/$SIDE/belebele_ind_Latn.jsonl"
  predict_cell "$SIDE" "$MODEL" "$MID" belebele_jsonl \
    "$DATA/belebele/eng_Latn.jsonl" eng belebele_eng_Latn \
    "$PRED/$SIDE/belebele_eng_Latn.jsonl"
  predict_cell "$SIDE" "$MODEL" "$MID" indommlu_jsonl \
    "$DATA/indommlu/samples.jsonl" ind indommlu \
    "$PRED/$SIDE/indommlu.jsonl" letter
  predict_cell "$SIDE" "$MODEL" "$MID" copal_jsonl \
    "$DATA/copal_id/samples.jsonl" ind copal_id \
    "$PRED/$SIDE/copal_id.jsonl" zero_based_index
done

PYTHONPATH=src "$PYTHON" -m linguaeval language-matrix-offline \
  configs/examples/27_language_capability_real_base_sft.yaml

echo "[run_language_real_p3fd] done → results/27_language_capability_real_base_sft"
