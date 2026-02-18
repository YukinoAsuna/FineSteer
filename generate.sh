#!/bin/bash

TRAIN_VAL_DIR=data/instructions/train_val
DEVICE=cuda:0

# 定义要处理的模型列表
models=( "llama3.1" "qwen2.5" "gemma-2")

# 遍历每个模型
for model in "${models[@]}"; do
    # 为每个模型设置相应的目录和昵称
    EMBEDDING_DIR="data/embeddings/$model"  # 输出嵌入的目录
    NICKNAME="$model"
    GENERATE_CONFIG_DIR="config/$model"
    
    # 创建嵌入目录（如果不存在）
    mkdir -p "$EMBEDDING_DIR"
    
    # 输出当前处理的模型信息
    echo "Generating response for $NICKNAME"
    
    # 检查配置目录是否存在
    if [ -d "$GENERATE_CONFIG_DIR" ]; then
        # 遍历配置文件并执行生成命令
        for file in "$GENERATE_CONFIG_DIR"/*.yaml; do
            # 检查是否有匹配的文件
            if [ -f "$file" ]; then
                filename=$(basename "$file" .yaml)
                echo "Generating response for $file"
                python new_eval.py --config_path "$file"
            fi
        done
    else
        echo "Warning: Configuration directory $GENERATE_CONFIG_DIR does not exist. Skipping."
    fi
    
    echo "---------------------------"
done