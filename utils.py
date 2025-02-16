from transformers import AutoModelForCausalLM, AutoTokenizer
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
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
    "llama-3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-v0.2": "mistralai/Mistral-7B-Instruct-v0.2", 
    "mistral-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma-2": "google/gemma-2-9b-it",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "vicuna-v1.5": "lmsys/vicuna-7b-v1.5",
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

    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer

def load_bleurt(device):
    """BLEURT model and tokenizer"""
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20').to(device)
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
    model.eval()
    
    return model, tokenizer

def get_chat(model_name: str, question: str):
    """chat template for LLMs"""
    prompt = INSTRUCT.format(question)
    if "llama" in model_name:
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
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

