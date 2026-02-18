from transformers import AutoModelForCausalLM, AutoTokenizer
from bleurt_pytorch.bleurt.tokenization_bleurt import BleurtSPTokenizer
from bleurt_pytorch.bleurt.modeling_bleurt import BleurtForSequenceClassification
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import random
import string

SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
INSTRUCT = "Answer the question concisely. Q: {} A:"


MODEL_NAME = {
    "llama-2": "meta-llama/Llama-2-7b-chat-hf",
    "llama-2_13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-v0.2": "mistralai/Mistral-7B-Instruct-v0.2", 
    "mistral-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-2": "google/gemma-2-9b-it",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5_14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5_32b": "Qwen/Qwen2.5-32B-Instruct",
    "vicuna-v1.5": "lmsys/vicuna-7b-v1.5",
    "llama3.1":"meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.2":"meta-llama/Llama-3.2-3B-Instruct",
}


def get_model_name(model_name):
    return MODEL_NAME[model_name]

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def load_model_and_tokenizer(model_name, device, torch_dtype=torch.float16):
    """prepare LLM and tokenizer"""
    model_name = get_model_name(model_name)
    if torch.cuda.device_count() > 1:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa",torch_dtype=torch_dtype, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa",torch_dtype=torch_dtype).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    return model, tokenizer

def load_bleurt(device):
    """BLEURT model and tokenizer"""
    tokenizer = BleurtSPTokenizer.from_pretrained("lucadiliello/BLEURT-20")  # 或本地路径
    model = BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20",use_safetensors=True).to(device)
    model.eval()
    
    return model, tokenizer

def get_chat(model_name: str, question: str):
    """chat template for LLMs"""
    prompt = INSTRUCT.format(question)
    if "llama" in model_name or "qwen" in model_name:
        chat = [
            {"role": "user", "content": prompt},
        ]
    elif "mistral" in model_name or "gemma" in model_name:
        chat = [
            {"role": "user", "content": prompt},
        ]
        
    return chat
    
    
def write_to_csv(generated_sentence, label, file_path):
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([generated_sentence, label]) 
        
        
        
def preprocess_tqa(ds):
    """remove the null string in 'correct_answers' and 'incorrect_answers' """
    def remove_empty_answers(example):
        example["correct_answers"] = [answer for answer in example["correct_answers"] if answer.strip()]
        example["incorrect_answers"] = [answer for answer in example["incorrect_answers"] if answer.strip()]
        return example
    
    filtered_ds = ds.map(remove_empty_answers)
    
    return filtered_ds

