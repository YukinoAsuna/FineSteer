import argparse
import math
import os
from typing import List

import torch
from datasets import Dataset, load_from_disk, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import time

from eval import OpenGenEvalPipeline, MCEvalPipeline
from model import LinearUNet
from rectified_flow import RectifiedFlow
from utils import load_model_and_tokenizer, seed_everything, get_chat, get_model_name
from wrapper import Wrapper

device = "cuda" if torch.cuda.is_available() else "cpu"



#! deal with dataset and dataloader
def transfer_data_loader(ds_name, layers, batch_size=136):
    """Whole TQA dataset as training set."""
    ds = load_from_disk(ds_name)
    ds.set_format(type='torch', columns=[f"y_win_layer{layer}" for layer in layers] + [f"y_lose_layer{layer}" for layer in layers])
    
    y_win_set = [[] for _ in range(len(layers))]
    y_lose_set = [[] for _ in range(len(layers))]
    for example in ds:
        for idx, layer in enumerate(layers):
            y_win = example[f"y_win_layer{layer}"]
            y_lose = example[f"y_lose_layer{layer}"]

            y_win_pair = y_win.repeat(1, y_lose.shape[0]).reshape(-1, y_win.shape[1])
            y_lose_pair = y_lose.tile((y_win.shape[0], 1))
            y_win_set[idx].append(y_win_pair)
            y_lose_set[idx].append(y_lose_pair)
        
    y_win_set = [torch.cat(y_win_per_layer) for y_win_per_layer in y_win_set]
    y_lose_set = [torch.cat(y_lose_per_layer) for y_lose_per_layer in y_lose_set]
        
    data_dict = {
        **{f"y_win_layer{layers[idx]}": y_win for idx, y_win in enumerate(y_win_set)},
        **{f"y_lose_layer{layers[idx]}": y_lose for idx, y_lose in enumerate(y_lose_set)}
    }
    dataset = Dataset.from_dict(data_dict)
    attr_list = [f"y_win_layer{layer}" for layer in layers] + [f"y_lose_layer{layer}" for layer in layers]
    dataset.set_format(type='torch', columns=attr_list)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return ds, data_loader

def prepare_tqa_train_test_ds(tokenizer, ds_name, layers:List[int]=[13]):
    ds = load_from_disk(ds_name)
    train_ds = ds["train"]
    test_ds = ds["test"] 

    def encode(example):
        return tokenizer(example["template_q"], return_tensors="pt", add_special_tokens=False)  
    
    attr_list = [f"y_win_layer{layer}" for layer in layers] + [f"y_lose_layer{layer}" for layer in layers]
    train_ds.set_format(type='torch', columns=attr_list)
    
    test_ds = test_ds.map(encode)
    test_ds.set_format(type='torch', columns=attr_list + ['question', 'template_q', 'input_ids', 'correct_answers', 'incorrect_answers'])
    
    return train_ds, test_ds

def prepare_pair_data_loader(ds, layers:List[int], ds_type:str="train", batch_size=136):
    y_win_set = [[] for _ in range(len(layers))]
    y_lose_set = [[] for _ in range(len(layers))]
    for example in ds:
        for idx, layer in enumerate(layers):
            y_win = example[f"y_win_layer{layer}"]
            y_lose = example[f"y_lose_layer{layer}"]

            y_win_pair = y_win.repeat(1, y_lose.shape[0]).reshape(-1, y_win.shape[1])
            y_lose_pair = y_lose.tile((y_win.shape[0], 1))
            y_win_set[idx].append(y_win_pair)
            y_lose_set[idx].append(y_lose_pair)
        
    y_win_set = [torch.cat(y_win_per_layer) for y_win_per_layer in y_win_set]
    y_lose_set = [torch.cat(y_lose_per_layer) for y_lose_per_layer in y_lose_set]
        
    data_dict = {
        **{f"y_win_layer{layers[idx]}": y_win for idx, y_win in enumerate(y_win_set)},
        **{f"y_lose_layer{layers[idx]}": y_lose for idx, y_lose in enumerate(y_lose_set)}
    }
    dataset = Dataset.from_dict(data_dict)
    attr_list = [f"y_win_layer{layer}" for layer in layers] + [f"y_lose_layer{layer}" for layer in layers]
    dataset.set_format(type='torch', columns=attr_list)
    
    if ds_type == "train":
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif ds_type == "validate":
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif ds_type == "test":
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError("Invalid dataset type.")
    
    return data_loader

def prepare_halueval_test_ds(model_name, tokenizer):
    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    
    def encode(example):
        chat = get_chat(model_name, example["question"])
        formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
    ds = ds.map(encode)
    ds.set_format(type='torch', columns=['input_ids', 'question', 'knowledge', 'right_answer', 'hallucinated_answer'])
    
    return ds

def prepare_nq_test_ds(model_name, tokenizer):
    ds = load_dataset("OamPatel/iti_nq_open_val", split="validation")
    
    def encode(example):
        chat = get_chat(model_name, example["question"])
        formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        return tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
    ds = ds.map(encode)
    ds.set_format(type='torch', columns=['input_ids', 'question', 'answer', 'false_answer'])
    
    return ds

def prepare_triviaqa_test_ds(model_name, tokenizer):
    ds = load_dataset("OamPatel/iti_trivia_qa_val", split="validation")
    input_ids = []
    for example in ds:
        chat = get_chat(model_name, example["question"])
        formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        input_ids.append(tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)["input_ids"])
    
    data_dict = {
        "question": [x["question"] for x in ds],
        "correct_answers": [x["answer"]["normalized_value"] for x in ds],
        "incorrect_answers": [x["false_answer"] for x in ds],
        "input_ids": input_ids
    }
    
    dataset = Dataset.from_dict(data_dict)
    dataset.set_format(type='torch', columns=['input_ids', 'question', 'correct_answers', 'incorrect_answers'])
    return dataset
    

#! train flow
def train_flow(model:RectifiedFlow, train_loader, val_loader, layer, num_epochs, device, wandb_proj:str='flow',save_path:str=None):
    if wandb_proj is not None:
        import wandb
        wandb.init(project=wandb_proj)
    print("Start training...")    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_warmup_steps = 100
    num_training_steps = len(train_loader) * num_epochs
    min_lr_scale = 0.7
    def cosine_schedule_with_warmup(current_step:int):
        if current_step < num_warmup_steps:
            # Linear warm-up
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay after warm-up
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_scale, cosine_decay)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_schedule_with_warmup)
    # calculate training time....
    start_time = time.time()
    train_losses = []
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Training")
        for example in train_bar:
            y_win = example[f"y_win_layer{layer}"]
            y_lose = example[f"y_lose_layer{layer}"]
            y_win, y_lose = y_win.to(device), y_lose.to(device)
                
            loss = model(y_win, y_lose, return_loss_breakdown = False)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_bar.set_postfix(train_loss=loss.item())
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        
        # validation phase
        if val_loader is None:
            continue
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Validation", leave=False)
        with torch.no_grad():
            for example in val_bar:
                y_win = example[f"y_win_layer{layer}"]
                y_lose = example[f"y_lose_layer{layer}"]
                print(y_win.shape, y_lose.shape)
                y_win, y_lose = y_win.to(device), y_lose.to(device)
                
                loss = model(y_win, y_lose, return_loss_breakdown = False)  
                val_loss += loss.item()
                val_bar.set_postfix(val_loss=loss.item())
        
        val_loss /= len(val_loader)
        
        # Log losses to W&B
        if wandb_proj is not None:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "layer": layer,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
    end_time = time.time()
    print(f"Training time: {end_time - start_time}")

    # save model
    torch.save(model.state_dict(), save_path)
    
    
#! Open generation -- TQA
def flow_llm(args, model, tokenizer, device, ds_name, wandb_proj=None, save_res_name=None, save_nn_name=None):
    hid_dim = model.config.hidden_size
    print("hidden dimension: ", hid_dim)
    
    res_dir = f"{args.model_name}_tqa_results"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    layers = args.layers
    if len(layers) == 1:
        flow_layer = layers[0]
    else:
        raise ValueError("Only support one layer for now.")
    
    train_ds, test_ds = prepare_tqa_train_test_ds(tokenizer, ds_name, layers)
    hs_mat = torch.cat([train_ds[i][f"y_win_layer{flow_layer}"] for i in range(len(train_ds))], dim=0)
    _, _, v = torch.svd(hs_mat)

    save_res_path = os.path.join(res_dir, save_res_name)
    
    flows = []
    open_gen_eval_pipeline = OpenGenEvalPipeline(model, tokenizer, device, layers, test_ds, eval_ds_name="tqa", k=args.k)
    wrapper = Wrapper
    for idx, layer in enumerate(layers):
        save_model_path = os.path.join(res_dir, save_nn_name)
        save_res_path += f"_{layer}"
        save_model_path += f"_{layer}.pth"
        
        # UNet for rectified flow
        unet = LinearUNet(
            hid_dim=hid_dim,
            depth=4,
            feature_scale=0.5,
            time_embedding_dim=128,
        ).to(device)

        rectified_flow = RectifiedFlow(unet, data_shape=(hid_dim,))
        if args.train:
            train_loader = prepare_pair_data_loader(train_ds, layers, ds_type="train")
            train_flow(rectified_flow, train_loader=train_loader, val_loader=None, layer=layer, num_epochs=args.num_epochs, device=device, wandb_proj=wandb_proj, save_path=save_model_path)
        else:
            rectified_flow.load_state_dict(torch.load(save_model_path))
        
        rectified_flow.eval()
        flows.append(rectified_flow)
        
    # evaluate
    open_gen_eval_pipeline.flow_eval_pipeline(flows, wrapper, v, args.alpha, args.beta, eval_method=args.eval_method, file_name=save_res_path)

def flow_llm_mc(args, model, tokenizer, device, ds_name, model_name, wandb_proj=None, save_nn_name=None):
    res_dir = f"{args.model_name}_tqa_results"
    if os.path.exists(res_dir) == False:
        os.makedirs(res_dir)
    layers = args.layers
    if len(layers) == 1:
        flow_layer = layers[0]
    else:
        raise ValueError("Only support one layer")
    
    train_ds, test_ds = prepare_tqa_train_test_ds(tokenizer, ds_name, layers)
    hs_mat = torch.cat([train_ds[i][f"y_win_layer{flow_layer}"] for i in range(len(train_ds))], dim=0)
    _, s, v = torch.svd(hs_mat)
    
    flows = []
    mc_eval_pipeline = MCEvalPipeline(model, tokenizer, device, layers, test_ds, model_name)
    wrapper = Wrapper
    for idx, layer in enumerate(layers):
        save_model_path = os.path.join(res_dir, save_nn_name)
        # save_res_path += f"_{layer}"
        save_model_path += f"_{layer}.pth"
        
        # UNet for rectified flow
        unet = LinearUNet(
            hid_dim=hid_dim,
            depth=4,
            feature_scale=0.5,
            time_embedding_dim=128,
        ).to(device)

        rectified_flow = RectifiedFlow(unet, data_shape=(hid_dim,))
        if args.train:
            train_loader = prepare_pair_data_loader(train_ds, layers, ds_type="train")
            train_flow(rectified_flow, train_loader=train_loader, val_loader=None, layer=layer, num_epochs=args.num_epochs, device=device, wandb_proj=wandb_proj, save_path=save_model_path)
        else:
            rectified_flow.load_state_dict(torch.load(save_model_path))
            
        rectified_flow.eval()
        flows.append(rectified_flow)
    
    mc_eval_pipeline.flow_mc_pipeline(flows, wrapper, v, args.alpha, args.beta) 
    
def base_llm(args, model, tokenizer, device, ds_name, save_res_name="base"):
    res_dir = f"{args.model_name}_tqa_results"
    if os.path.exists(res_dir) == False:
        os.makedirs(res_dir)
        
    layers = args.layers
    _, test_ds = prepare_tqa_train_test_ds(tokenizer, ds_name, layers)
    
    save_res_path = os.path.join(res_dir, f"{args.eval_method}_" + save_res_name)
    open_gen_eval_pipeline = OpenGenEvalPipeline(model, tokenizer, device, layers, test_ds)
    # evaluate
    open_gen_eval_pipeline.base_eval_pipeline(eval_method=args.eval_method, file_name=save_res_path)
    
def base_llm_mc(model, tokenizer, device, ds_name, model_name):
    layers = [20]
    _, test_ds = prepare_tqa_train_test_ds(tokenizer, ds_name, layers)
    
    mc_eval_pipeline = MCEvalPipeline(model, tokenizer, device, layers, test_ds, model_name)
    # evaluate
    mc_eval_pipeline.base_mc_pipeline()
    
#! Open generation -- transfer
def transfer_flow_gen(args, model, tokenizer, device, ds_name, wandb_proj=None, save_res_name=None, save_mlp_name=None, transfer_ds_name="halueval"):
    hid_dim = model.config.hidden_size
    print("hidden dimension: ", hid_dim)
    res_dir = f"{args.model_name}_{transfer_ds_name}_results"
    if os.path.exists(res_dir) == False:
        os.makedirs(res_dir)
    layers = args.layers
    if len(layers) == 1:
        flow_layer = layers[0]
    else:
        raise ValueError("Only support one layer for now.")
    
    train_ds, train_loader = transfer_data_loader(ds_name, layers)
    hs_mat = torch.cat([train_ds[i][f"y_win_layer{flow_layer}"] for i in range(len(train_ds))], dim=0)
    _, _, v = torch.svd(hs_mat)

    save_res_path = os.path.join(res_dir, save_res_name)
    
    flows = []
    if transfer_ds_name == "halueval":
        test_ds = prepare_halueval_test_ds(args.model_name, tokenizer)
    elif transfer_ds_name == "nq":
        test_ds = prepare_nq_test_ds(args.model_name, tokenizer)
    elif transfer_ds_name == "triviaqa":
        test_ds = prepare_triviaqa_test_ds(args.model_name, tokenizer)
    else:
        raise ValueError("Invalid dataset name.")
    
    open_gen_eval_pipeline = OpenGenEvalPipeline(model, tokenizer, device, layers, test_ds, eval_ds_name=transfer_ds_name, k=args.k)
    wrapper = Wrapper
    for idx, layer in enumerate(layers):
        save_model_path = os.path.join(res_dir, save_mlp_name)
        save_res_path += f"_{layer}"
        save_model_path += f"_{layer}.pth"
        
        # UNet for rectified flow
        unet = LinearUNet(
            hid_dim=hid_dim,
            depth=4,
            feature_scale=0.5,
            time_embedding_dim=128,
        ).to(device)

        rectified_flow = RectifiedFlow(unet, data_shape=(hid_dim,))
        if args.train:
            train_flow(rectified_flow, train_loader=train_loader, val_loader=None, layer=layer, num_epochs=args.num_epochs, device=device, wandb_proj=wandb_proj, save_path=save_model_path)
        else:
            rectified_flow.load_state_dict(torch.load(save_model_path))
        
        rectified_flow.eval()
        flows.append(rectified_flow)
        
    # evaluate
    open_gen_eval_pipeline.flow_eval_pipeline(flows, wrapper, v, args.alpha, args.beta, eval_method=args.eval_method, file_name=save_res_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthFlow: Truthful LLM Generation via Representation Flow Correction")
    
    parser.add_argument("--model_name", type=str, default="llama-3", help="The name of the model to be used.")
    parser.add_argument("--layers", type=int, default=None, help="Which layer to apply flow matching model")
    parser.add_argument("--train", action='store_true', help="Whether to train the flow model.")
    parser.add_argument("--num_epochs", type=int, default=25, help="The number of epochs for training rectified flow.")
    parser.add_argument('--ds_path', type=str, required=True, default=None, help="Local path to dataset to train flow.")
    parser.add_argument('--torch_dtype', type=str, default='fp16', help="The dtype of the model.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")
    
    parser.add_argument('--alpha', type=float, default=1.5, help="The weight for sv.")
    parser.add_argument('--k', type=int, default=20, help="The number of singular values to be used.")
    parser.add_argument('--truthflow', action='store_true', help="Whether to use TruthFlow method.")
    parser.add_argument('--base', action='store_true', help="Whether to use base LLM.")
    parser.add_argument('--mc_eval', action='store_true', help="Whether to use MC evaluation.")
    parser.add_argument('--opengen_eval', action='store_true', help="Whether to use OpenGen evaluation.")
    parser.add_argument('--eval_method', type=str, default="gpt", help="The evaluation method.")

    args = parser.parse_args()

    seed_everything(args.seed)
    
    # load llm and tokenizer
    model_name = get_model_name(args.model_name)
    ds_path = args.ds_path
    if args.torch_dtype == "fp16":
        torch_dtype = torch.float16
    elif args.torch_dtype == "fp32":
        torch_dtype = torch.float32
    elif args.torch_dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        raise ValueError("Invalid dtype.")
    
    print(f"Loading {model_name}...")
    model, tokenizer = load_model_and_tokenizer(args.model_name, device, torch_dtype)
    model.eval()
    hid_dim=model.config.hidden_size
    
    if args.truthflow:
        save_nn_name = f"TruthFlow_{args.model_name}_seed{args.seed}_epoch{args.num_epochs}" # save neural network for flow
        save_res_name = f"TruthFlow_{args.torch_dtype}_{args.model_name}_seed{args.seed}_k{args.k}_alpha{args.alpha}_epoch{args.num_epochs}" # save generation results for flow
            
        if args.opengen_eval:
            flow_llm(args, model, tokenizer, device, ds_path, wandb_proj=None, save_res_name=save_res_name, save_nn_name=save_nn_name)
        elif args.mc_eval:
            flow_llm_mc(args, model, tokenizer, device, ds_path, args.model_name, wandb_proj=None, save_nn_name=save_nn_name)
        else:
            raise ValueError("Invalid evaluation method.")

        
    elif args.base:
        if args.opengen_eval:
            base_llm(args, model, tokenizer, device, ds_path)
        elif args.mc_eval:
            base_llm_mc(model, tokenizer, device, ds_path, args.model_name)
            
    else:
        raise ValueError("Invalid method.")

    