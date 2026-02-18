#!/bin/bash
set -euo pipefail

show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "说明: 依次执行 gemma-2 -> llama3 -> llama3.2 -> qwen2.5"
    echo ""
    echo "选项:"
    echo "  --layers LAYERS       仅对 gemma-2 生效的 layers (默认: 20)"
    echo "  --seed SEED           seed 参数 (默认: 0)"
    echo "  --num_epochs EPOCHS   训练轮数 (默认: 40)"
    echo "  -h, --help            显示帮助信息"
    echo ""
    echo "示例: $0 --seed 42 --num_epochs 50"
}

# 默认参数
LAYERS=20       # 仅用于 gemma-2
SEED=0
NUM_EPOCHS=20

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --layers)      LAYERS="$2"; shift 2;;
        --seed)        SEED="$2"; shift 2;;
        --num_epochs)  NUM_EPOCHS="$2"; shift 2;;
        -h|--help)     show_help; exit 0;;
        *) echo "未知选项: $1"; show_help; exit 1;;
    esac
done

#MODELS=( "gemma-2" "llama3" "llama3.2" "qwen2.5" )
MODELS=( "gemma-2")
resolve_ds_path() {
    local model_name="$1"
    case "$model_name" in
        gemma-2)
            echo "/hpc2hdd/home/hwang574/wzx/TruthAlpha_2/data_tqa/gemma-2_ans_avg_seed0_testsize0.5_layers_18_20_22"
            ;;
        qwen2.5)
            echo "/hpc2hdd/home/hwang574/wzx/TruthAlpha_2/data_tqa/qwen2.5_ans_avg_seed0_testsize0.5_layers_10_11_12_13_14_15_16_17_18_19_20"
            ;;
        llama3)
            echo "/hpc2hdd/home/hwang574/wzx/TruthAlpha_2/data_tqa/llama3_ans_avg_seed0_testsize0.5_layers_12"
            ;;
        llama3.2)
            echo "/hpc2hdd/home/hwang574/wzx/TruthAlpha_2/data_tqa/llama3.2_ans_avg_seed0_testsize0.5_layers_8_9_10_11_12_13"
            ;;
        *) return 1;;
    esac
}

# 按需求设置各模型的 layers
get_layers() {
    local model_name="$1"
    case "$model_name" in
        llama3.2) echo "11" ;;
        llama3)   echo "12" ;;
        qwen2.5)  echo "12" ;;
        gemma-2)  echo "$LAYERS" ;;   # 保持可配置
        *)        echo "$LAYERS" ;;
    esac
}
get_alpha() {
    local model_name="$1"
    case "$model_name" in
        gemma-2)  echo "1.5" ;;
        qwen2.5)  echo "2.5" ;;
        llama3)   echo "4.3" ;;
        llama3.2) echo "4.0" ;;
    esac
}
get_epochs() {
    local model_name="$1"
    case "$model_name" in
        gemma-2)  echo "40" ;;
        qwen2.5)  echo "30" ;;
        llama3)   echo "25" ;;
        llama3.2) echo "25" ;;
    esac
}
get_k() {
    local model_name="$1"
    case "$model_name" in
        gemma-2)  echo "20" ;;
        qwen2.5)  echo "20" ;;
        llama3)   echo "10" ;;
        llama3.2) echo "10" ;;
    esac
}
run_one() {
    local model_name="$1"
    local ds_path layers alpha epochs ks

    ds_path="$(resolve_ds_path "$model_name")" || {
        echo "错误: 不支持的模型名称: $model_name"
        return 1
    }

    layers="$(get_layers "$model_name")"
    alpha="$(get_alpha "$model_name")"
    epochs="$(get_epochs "$model_name")"
    ks="$(get_k "$model_name")"
    if [[ ! -d "$ds_path" ]]; then
        echo "警告: ds_path 目录不存在: $ds_path"
        echo "继续执行命令..."
    fi

    local cmd="python flow.py \
        --model_name $model_name \
        --ds_path $ds_path \
        --layers $layers \
        --seed $SEED \
        --method truthflow \
        --opengen_eval \
        --eval_method bleurt \
        --k $ks \
        --alpha $alpha \
        --train \
        --num_epochs $epochs"

    echo "======================================="
    echo "开始执行模型: $model_name"
    echo "layers: $layers"
    echo "执行命令: $cmd"
    echo "---------------------------------------"
    eval "$cmd"
    echo "完成模型: $model_name"
    echo "======================================="
    echo
}

fail_list=()
for m in "${MODELS[@]}"; do
    if ! run_one "$m"; then
        fail_list+=("$m")
        echo "模型 $m 执行失败，继续下一个。"
    fi
done

if (( ${#fail_list[@]} )); then
    echo "以下模型执行失败: ${fail_list[*]}"
    exit 2
else
    echo "全部模型执行完成。"
fi
