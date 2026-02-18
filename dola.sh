#!/bin/bash

# --------------------------------------------------------------------------
# 用法:
#   直接运行此脚本即可，无需任何参数。
#   ./run_experiments.sh
#
# 脚本会自动遍历下面 MODEL_NAMES 数组中定义的所有模型。
# --------------------------------------------------------------------------

# --- 配置 ---
# 在这里预先定义好要遍历的模型列表
MODEL_NAMES=("llama3" "llama3.2")

# 其他固定参数
LAYERS=12
SEED=0
METHOD="dola"
EVAL_METHOD="gpt"
K=20
ALPHA=1.5
NUM_EPOCHS=40


# --- 脚本主逻辑 ---
echo "开始执行预设模型的批量任务..."
echo

# 循环遍历预设在 MODEL_NAMES 数组中的每一个模型
for MODEL in "${MODEL_NAMES[@]}"
do
  # 使用 case 语句为每个模型设置特定的 DS_PATH
  case "$MODEL" in
    "gemma-2")
      DS_PATH="./data_tqa/gemma-2_ans_avg_seed0_testsize0.5_layers_18_20_22"
      ;;
    "llama3")
      DS_PATH="./data_tqa/llama3_ans_avg_seed0_testsize0.5_layers_12"
      ;;
    "llama3.2")
      DS_PATH="./data_tqa/llama3.2_ans_avg_seed0_testsize0.5_layers_8_9_10_11_12_13"
      ;;
    *)
      # 此情况理论上不会发生，因为列表是固定的，但作为保险措施保留
      echo "警告: 在预设列表中发现未知模型 '$MODEL'。将跳过。"
      continue
      ;;
  esac

  echo "========================================================================"
  echo "正在启动模型: $MODEL"
  echo "使用专属路径: $DS_PATH"
  echo "========================================================================"

  # 执行Python命令
  python flow.py \
    --model_name "$MODEL" \
    --ds_path "$DS_PATH" \
    --layers "$LAYERS" \
    --seed "$SEED" \
    --method "$METHOD" \
    --opengen_eval \
    --eval_method "$EVAL_METHOD" \
    --k "$K" \
    --alpha "$ALPHA" \
    --train \
    --num_epochs "$NUM_EPOCHS"

  # 检查命令是否执行成功
  if [ $? -ne 0 ]; then
    echo "错误: 模型 $MODEL 的任务执行失败。正在中止脚本。"
    exit 1
  fi

  echo "模型 $MODEL 的任务已成功完成。"
  echo
done

echo "所有预设模型的任务均已执行完毕。"