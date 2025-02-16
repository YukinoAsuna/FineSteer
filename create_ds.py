from datasets import load_dataset
from tqdm import tqdm

import torch
import re
import numpy as np
import os

from utils import seed_everything, load_model_and_tokenizer, get_chat, preprocess_tqa

import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"



def extract_hq_avg_minus_tqa(ds, args):
    """extract query last token hidden states as y_lose, and (correct answer average hidden states - incorrect answer average hidden states) as y_win."""
    
    template_q = [] # chat template with only the question
    y_win_set, y_lose_set = [], [] # y_lose -- hq, y_win -- hc - hi
    
    for data in tqdm(ds):
        chat = get_chat(args.model_name, data["question"])
        
        formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        template_q.append(formatted_chat)
        tokenized_format_chat = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
        question_token_length = tokenized_format_chat["input_ids"].shape[1]
        
        # hq
        hq_list = []
        for layer in args.layers:
            with torch.no_grad():
                outputs = model(**tokenized_format_chat.to(device), output_hidden_states=True)
            # last token hidden states
            hq = outputs.hidden_states[layer][0, -1, :]
            hq_list.append(hq.cpu())
        
        hqs = torch.stack(hq_list)  #* (num_layers, hid_dim)
        y_lose_set.append(hqs.unsqueeze(0))
        
                  
        inc_chat = chat + [{"role": "assistant", "content": data["incorrect_answers"][0]}] if data["incorrect_answers"][0] != "" else chat
        c_chat = chat + [{"role": "assistant", "content": data["correct_answers"][0]}] if data["correct_answers"][0] != "" else chat
        
        inc_formatted_chat = tokenizer.apply_chat_template(inc_chat, tokenize=False, add_generation_prompt=False)
        c_formatted_chat = tokenizer.apply_chat_template(c_chat, tokenize=False, add_generation_prompt=False)

        tokenzied_inc_chat = tokenizer(inc_formatted_chat, return_tensors="pt", add_special_tokens=False)
        tokenized_c_chat = tokenizer(c_formatted_chat, return_tensors="pt", add_special_tokens=False)
        
        with torch.no_grad():
            c_outputs = model(**tokenized_c_chat.to(device), output_hidden_states=True)
            inc_outputs = model(**tokenzied_inc_chat.to(device), output_hidden_states=True)
        
        avg_hc_minus_hi = []
        for layer in args.layers:
            if args.token_pos == "qa_avg":
                # average over all Q and A tokens
                hc = c_outputs.hidden_states[layer][0, :, :].mean(dim=0)
                hi = inc_outputs.hidden_states[layer][0, :, :].mean(dim=0)

            elif args.token_pos == "ans_avg":
                # average over all answer tokens
                hc = c_outputs.hidden_states[layer][0, question_token_length:, :].mean(dim=0)
                hi = inc_outputs.hidden_states[layer][0, question_token_length:, :].mean(dim=0)
                
            elif args.token_pos == "last":
                # last token
                hc = c_outputs.hidden_states[layer][0, -1, :]
                hi = inc_outputs.hidden_states[layer][0, -1, :]

            else:
                raise ValueError("Invalid setting.")
                
            avg_hc_minus_hi.append((hc - hi).cpu())
            
        avgs = torch.stack(avg_hc_minus_hi) 
        y_win_set.append(avgs.unsqueeze(0))

    return y_win_set, y_lose_set, template_q

def construct_train_test_ds(args):
    train_win, train_lose, train_tmp = extract_hq_avg_minus_tqa(train_ds, args)
    test_win, test_lose, test_tmp = extract_hq_avg_minus_tqa(test_ds, args)
    train_data_dict = {
        'correct_answers': [x["correct_answers"] for x in train_ds],
        'incorrect_answers': [x["incorrect_answers"] for x in train_ds],
        'question': [x["question"] for x in train_ds],
        'template_q': train_tmp,
        'category': [x["category"] for x in train_ds],
    }

    test_data_dict = {
        'correct_answers': [x["correct_answers"] for x in test_ds],
        'incorrect_answers': [x["incorrect_answers"] for x in test_ds],
        'question': [x["question"] for x in test_ds],
        'template_q': test_tmp,
        'category': [x["category"] for x in test_ds],
    }
        
    for j, layer in enumerate(args.layers):
            train_data_dict[f"y_win_layer{layer}"] = [train_win[i][:, j, :] for i in range(len(train_win))]
            train_data_dict[f"y_lose_layer{layer}"] = [train_lose[k][:, j, :] for k in range(len(train_lose))]

    for j, layer in enumerate(args.layers):
        test_data_dict[f"y_win_layer{layer}"] = [test_win[i][:, j, :] for i in range(len(test_win))]
        test_data_dict[f"y_lose_layer{layer}"] = [test_lose[k][:, j, :] for k in range(len(test_lose))]
        
    from datasets import Dataset, DatasetDict
    train_dataset = Dataset.from_dict(train_data_dict)
    test_dataset = Dataset.from_dict(test_data_dict)
    dataset = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    return dataset
    
def construct_whole_ds(args):
    train_win, train_lose, train_tmp = extract_hq_avg_minus_tqa(train_ds, args)
    train_data_dict = {
        'correct_answers': [x["correct_answers"] for x in train_ds],
        'incorrect_answers': [x["incorrect_answers"] for x in train_ds],
        'question': [x["question"] for x in train_ds],
        'template_q': train_tmp,
        'category': [x["category"] for x in train_ds],
    }
        
    for j, layer in enumerate(args.layers):
            train_data_dict[f"y_win_layer{layer}"] = [train_win[i][:, j, :] for i in range(len(train_win))]
            train_data_dict[f"y_lose_layer{layer}"] = [train_lose[k][:, j, :] for k in range(len(train_lose))]
    from datasets import Dataset
    train_dataset = Dataset.from_dict(train_data_dict)

    return train_dataset
    
#! construct the dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create training data from truthfulqa.")

    parser.add_argument('--layers', type=int, default=None, nargs='+', help="The layers to extract hidden states.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    parser.add_argument('--model_name', type=str, default='llama-3', help="The base LLM.")
    parser.add_argument('--ds_name', type=str, default='tqa', help="The dataset name.")
    parser.add_argument('--test_size', type=float, default=0.5, help="The size of the test set.")
    parser.add_argument('--whole_dataset', action='store_true', help="Whether to use the whole dataset.")
    parser.add_argument('--few_shot', action='store_true', help="Whether to use few-shot prompt.")
    parser.add_argument('--token_pos', type=str, default='ans_avg', help="The average tokens to use. Can only be 'qa_avg', 'ans_avg', or 'last'.")
    parser.add_argument('--batch_size', type=int, default=1, help="The batch size.")
    parser.add_argument('--torch_dtype', type=str, default='fp16', help="The dtype of the model.")


    args = parser.parse_args()
    
    seed_everything(args.seed)


    print(f"=================={args.model_name}==================")
    if args.torch_dtype == "fp16":
        torch_dtype = torch.float16
    elif args.torch_dtype == "fp32":
        torch_dtype = torch.float32
    elif args.torch_dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError("Invalid dtype.")
    model, tokenizer = load_model_and_tokenizer(args.model_name, device, torch_dtype)
    model.eval()
    hid_dim = model.config.hidden_size

    if args.ds_name == "tqa":
        print("===================TruthfulQA==================")
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        ds = preprocess_tqa(ds)
    else:
        raise ValueError("Invalid dataset name.")
    

    #! split the dataset into train and test
    if not args.whole_dataset:
        train_test_split = ds.train_test_split(test_size=args.test_size, seed=args.seed)
        train_ds = train_test_split["train"]
        test_ds = train_test_split["test"]
        ds_name = f"{args.model_name}_{args.avg_tokens}_seed{args.seed}_testsize{args.test_size}_layers"
        print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")
        dataset = construct_train_test_ds(args)
    else:
        ds_name = f"{args.model_name}_whole_tqa_{args.avg_tokens}_seed{args.seed}_layers"
        dataset = construct_whole_ds(args)

    
    # save to local disk
    for layer in args.layers:
        ds_name += f"_{layer}"
        
    dir_name = f"data_{args.ds_name}"
    os.makedirs(dir_name, exist_ok=True)
    dataset.save_to_disk(os.path.join(dir_name, ds_name))
