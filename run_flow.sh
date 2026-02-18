#!/usr/bin/env bash
set -euo pipefail

# Define common parameters
MODEL_NAME="llama3.2"
DS_PATH="./data_tqa/llama3.2_ans_avg_seed0_testsize0.5_layers_8_9_10_11_12_13"
LAYERS=(8 9 10 11 12 13)
SEED=0
METHOD="steer"
OPENGEN_EVAL="--opengen_eval"   # 留空则不会添加该参数
EVAL_METHOD="gpt"
ALPHA=3.5
EPOCHS=20

# k sweep
K=10

for LAYER in "${LAYERS[@]}"; do
  echo "Starting flow.py with k=${K}"
  python flow.py \
    --model_name "${MODEL_NAME}" \
    --ds_path "${DS_PATH}" \
    --layers "${LAYER}" \
    --seed "${SEED}" \
    --method "${METHOD}" \
    ${OPENGEN_EVAL} \
    --eval_method "${EVAL_METHOD}" \
    --train \
    --k "${K}" \
    --alpha "${ALPHA}" \
    --num_epochs "${EPOCHS}"

  echo "Finished flow.py with k=${K}"
  echo "------------------------------------"
done

echo "All scripts finished successfully."
