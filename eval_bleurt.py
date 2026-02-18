import torch
from bleurt_pytorch.bleurt.tokenization_bleurt import BleurtSPTokenizer
from bleurt_pytorch.bleurt.modeling_bleurt import BleurtForSequenceClassification
from typing import List, Union
import pandas as pd
import os
import argparse

def load_bleurt(device): 
    """BLEURT model and tokenizer""" 
    tokenizer = BleurtSPTokenizer.from_pretrained("lucadiliello/BLEURT-20")  # 或本地路径 
    model = BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20", use_safetensors=True).to(device) 
    model.eval() 
    
    return model, tokenizer

# BLEURT评估函数（基于您提供的代码）
def calculate_bleurt_score(model, tokenizer, ref, hyp): 
    model.eval() 
    input_data = tokenizer(ref, hyp, return_tensors='pt', max_length=511, truncation=True) 
    # to device
    input_data = {k: v.to(model.device) for k, v in input_data.items()} 
    with torch.no_grad(): 
        scores = model(**input_data).logits.flatten().squeeze() 
    return scores.item()  # 转换为Python标量

def bleurt_eval(bleurt, bleurt_tokenizer, gen_answer, correct_answers: Union[List[str], str], incorrect_answers: Union[List[str], str]): 
    if isinstance(correct_answers, str): 
        correct_answers = [correct_answers] 
    if isinstance(incorrect_answers, str): 
        incorrect_answers = [incorrect_answers] 
    
    c_scores = [] 
    for c_ans in correct_answers: 
        c_score = calculate_bleurt_score(bleurt, bleurt_tokenizer, c_ans, gen_answer) 
        c_scores.append(c_score) 
    
    inc_scores = [] 
    for inc_ans in incorrect_answers: 
        inc_score = calculate_bleurt_score(bleurt, bleurt_tokenizer, inc_ans, gen_answer) 
        inc_scores.append(inc_score) 
    
    c_score = max(c_scores) 
    inc_score = max(inc_scores) 
    
    if c_score > inc_score: 
        return 1 
    else: 
        return 0 

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description="BLEURT Evaluation Script")
    parser.add_argument("--input", "-i", required=True, help="Path to the input CSV file")
    args = parser.parse_args()
    
    # 输入文件路径
    input_path = args.input
    
    # 验证输入文件是否存在
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
    
    # 构造输出文件路径
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    output_filename = os.path.splitext(input_filename)[0] + "_bleurt.csv"
    output_path = os.path.join(input_dir, output_filename)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载BLEURT模型
    print("Loading BLEURT model...")
    bleurt, bleurt_tokenizer = load_bleurt(device)
    print("BLEURT model loaded successfully")
    
    # 读取CSV文件
    print(f"Reading input file: {input_path}")
    df = pd.read_csv(input_path)
    
    # 检查必要的列是否存在
    required_columns = ['model_answers', 'correct_answers', 'incorrect_answers']
    for col in required_columns:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in CSV file")
            return
    
    # 计算BLEURT分数
    print("Calculating BLEURT scores...")
    df['bleurt_result'] = df.apply(
        lambda row: bleurt_eval(
            bleurt, 
            bleurt_tokenizer, 
            row['model_answers'], 
            row['correct_answers'], 
            row['incorrect_answers']
        ),
        axis=1
    )
    
    # 保存结果
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    # 计算准确率
    accuracy = df['bleurt_result'].mean()
    print(f"BLEURT Evaluation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()