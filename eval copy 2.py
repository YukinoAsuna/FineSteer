from copy import deepcopy
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm
import time
from eval_utils import bleurt_eval, tqa_gpt_eval_true, halueval_gpt_eval_true, nq_gpt_eval_true, triviaqa_gpt_eval_true, tqa_mini_eval_true
from utils.utils import load_bleurt, write_to_csv, get_chat
import datetime
device = "cuda" if torch.cuda.is_available() else "cpu"

def format_best(data):
    # add "." to the end of the best answer
    best_ans = data["correct_answers"][0]
    assert best_ans is not None, "No correct answer found!"
    if best_ans[-1] != ".":
        best_ans += "."
    return best_ans
    
def format_c_inc_ans(data):
    # add "." to the end of each answer
    for i in range(len(data['correct_answers'])):
        if data['correct_answers'][i][-1] != ".":
            data['correct_answers'][i] += "."
    for i in range(len(data['incorrect_answers'])):
        if data['incorrect_answers'][i][-1] != ".":
            data['incorrect_answers'][i] += "."
    return data


def MC_calcs(scores_true, scores_false, ref_true, ref_best):
    # compute MC1: 1vFalse -- best correct answer vs all false answers
    max_false = max(scores_false)
    if scores_true[ref_true.index(ref_best)] > max_false:
        mc1 = 1.0
    else:
        mc1 = 0.0

    # compute MC2: normalized probability mass for correct answers
    probs_true = np.exp(scores_true)
    probs_false = np.exp(scores_false)

    probs_true = probs_true / (sum(probs_true) + sum(probs_false))
    mc2 = sum(probs_true)
    return mc1, mc2


class OpenGenEvalPipeline:
    def __init__(self, model, tokenizer, device, layers, test_ds, eval_ds_name, k:int=20):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layers = layers
        self.test_ds = test_ds
        self.eval_ds_name = eval_ds_name
        print(f"{eval_ds_name} evaluating...")
        self.k = k
    def steer_eval_pipeline(self, steers, wrapper, v, alpha, beta, k,model_name,eval_method:str="gpt", file_name:str="result"):
        #评估 TruthFlow 模型（带 flow 修正）。
        """evaluate TruthFlow"""
        if eval_method == "gpt":
            print("Using GPT-4 to evaluate...")

        elif eval_method == "bleurt":
            print("Using BLEURT to evaluate...")
            bleurt, bleurt_tokenizer = load_bleurt(device)
            bleurt.eval()

        else:
            raise ValueError("Invalid evaluation method. Please choose between 'gpt' and 'bleurt'.")
        
        total_num = len(self.test_ds)
        true_score = 0
        truthful_labels = []
        for data in tqdm(self.test_ds):
            with torch.no_grad():
                outputs = self.model(input_ids=data["input_ids"].to(self.device), output_hidden_states=True)

            original_layers = []
            for idx, layer in enumerate(self.layers):
                hs = outputs.hidden_states[layer][:, -1, :]# hidden_states[layer] shape:[1,29,3584],然后取最后一个token的激活
                model= steers[idx][0].half() 
                _,_,hs_steer = model(hs,steers[idx][1].half().to(self.device),steers[idx][2].half().to(self.device)) # shape:[1,3584]，也就是输出，预测结果

                original_layers.append(deepcopy(self.model.model.layers[layer]))
                self.model.model.layers[layer] = wrapper(self.model.model.layers[layer], model_name,hs_steer, alpha=alpha)#r 是一个 包装器，用于在模型层中注入 修正操作,对于每一个sample，都要重新构建一个包装器。因为steering vector是会变的
            self.model.eval()

            with torch.no_grad():
                outputs = self.model.generate(input_ids=data["input_ids"].to(self.device), do_sample=False, top_k=0, top_p=1.0, temperature=0, return_dict_in_generate=True, max_new_tokens=256, pad_token_id=self.tokenizer.eos_token_id)

            for idx, layer in enumerate(self.layers):
                self.model.model.layers[layer] = original_layers[idx]#把模型复原，没有包装器

            # evaluate answers
            if eval_method == "gpt":
                if self.eval_ds_name == "tqa":
                    true = tqa_gpt_eval_true(data["question"], data["correct_answers"], data["incorrect_answers"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "halueval":
                    true = halueval_gpt_eval_true(data["question"], data['knowledge'], data["right_answer"], data["hallucinated_answer"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "nq":
                    true = nq_gpt_eval_true(data["question"], data["answer"], data["false_answer"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "triviaqa":
                    true = triviaqa_gpt_eval_true(data["question"], data["correct_answers"], data["incorrect_answers"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                else:
                    raise ValueError("Invalid evaluation dataset. Please choose between 'tqa', 'halueval', 'triviaqa', and 'nq'.")
            elif eval_method == "bleurt":
                true = bleurt_eval(bleurt, bleurt_tokenizer, self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True), data["correct_answers"], data["incorrect_answers"])
            else:
                raise ValueError("Invalid evaluation method. Please choose between 'gpt' and 'bleurt'.")
            
            true_score += true
            truthful_labels.append(true)

            # save qa+true_score to csv
            write_to_csv(self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True), true, file_name+".csv")#有结果和回答的csv
            # save answers to csv
            write_to_csv(self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True), true, file_name+"_answer.csv")#只有答案的csv

        print(f"{eval_method} true score: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}")
        log_file = "evaluation_log.txt"
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_message = f"{current_time} - {eval_method} true score: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}, K:{k}\n"

        with open(log_file, "a") as f:
            f.write(log_message)

        print(f"Log saved to {log_file}")
        write_to_csv(f"{eval_method} true score: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}", None, file_name+".csv")
        
    def alphasteer_eval_pipeline(self, steering_matrix: torch.Tensor, model_name,wrapper, alpha:float, eval_method:str="gpt", file_name:str="result"):
        #评估 TruthFlow 模型（带 flow 修正）。#steering_matrix:shape [layer,dim,dim]
        """evaluate TruthFlow"""
        if eval_method == "gpt":
            print("Using GPT-4 to evaluate...")

        elif eval_method == "bleurt":
            print("Using BLEURT to evaluate...")
            bleurt, bleurt_tokenizer = load_bleurt(device)
            bleurt.eval()

        else:
            raise ValueError("Invalid evaluation method. Please choose between 'gpt' and 'bleurt'.")
        
        total_num = len(self.test_ds)
        true_score = 0
        truthful_labels = []
        for data in tqdm(self.test_ds):
            with torch.no_grad():
                outputs = self.model(input_ids=data["input_ids"].to(self.device), output_hidden_states=True)

            original_layers = []
            if len(steering_matrix.shape) == 2:
                steering_matrix = steering_matrix.unsqueeze(0)
            time1=time.time()
            for idx, layer in enumerate(self.layers):
                hs = outputs.hidden_states[layer][:, -1, :]# hidden_states[layer] shape:[1,29,3584],然后取最后一个token的激活
                #hs_flow = flow[idx].sample(hidden_states=hs) # shape:[1,3584]，也就是输出，预测结果
                layer_steering_matrix=steering_matrix[idx]#取出对应层的 steering matrix,shape:[4096,4096]
                original_layers.append(deepcopy(self.model.model.layers[layer]))
                self.model.model.layers[layer] = wrapper(self.model.model.layers[layer], model_name,layer_steering_matrix,alpha=alpha)#r 是一个 包装器，用于在模型层中注入 修正操作,对于每一个sample，都要重新构建一个包装器。因为steering vector是会变的
            self.model.eval()
            time2=time.time()
            print(f'execute time:{time2-time1}')
            with torch.no_grad():
                outputs = self.model.generate(input_ids=data["input_ids"].to(self.device), do_sample=False, top_k=0, top_p=1.0, temperature=0, return_dict_in_generate=True, max_new_tokens=256)
            time3=time.time()
            print(f'generation time:{time3-time2}')
            for idx, layer in enumerate(self.layers):
                self.model.model.layers[layer] = original_layers[idx]#把模型复原，没有包装器

            # evaluate answers
            if eval_method == "gpt":
                if self.eval_ds_name == "tqa":
                    true = tqa_gpt_eval_true(data["question"], data["correct_answers"], data["incorrect_answers"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "halueval":
                    true = halueval_gpt_eval_true(data["question"], data['knowledge'], data["right_answer"], data["hallucinated_answer"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "nq":
                    true = nq_gpt_eval_true(data["question"], data["answer"], data["false_answer"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "triviaqa":
                    true = triviaqa_gpt_eval_true(data["question"], data["correct_answers"], data["incorrect_answers"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                else:
                    raise ValueError("Invalid evaluation dataset. Please choose between 'tqa', 'halueval', 'triviaqa', and 'nq'.")
            elif eval_method == "bleurt":
                true = bleurt_eval(bleurt, bleurt_tokenizer, self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True), data["correct_answers"], data["incorrect_answers"])
            else:
                raise ValueError("Invalid evaluation method. Please choose between 'gpt' and 'bleurt'.")
            
            true_score += true
            truthful_labels.append(true)

            # save qa+true_score to csv
            write_to_csv(self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True), true, file_name+".csv")#有结果和回答的csv
            # save answers to csv
            write_to_csv(self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True), true, file_name+"_answer.csv")#只有答案的csv
        log_file = "evaluation_log.txt"
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_message = f"{current_time} - {eval_method} true score: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}\n"

        with open(log_file, "a") as f:
            f.write(log_message)
        print(f"{eval_method} true score: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}")
        write_to_csv(f"{eval_method} true score: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}", None, file_name+".csv")
    
    def flow_eval_pipeline(self, flow, wrapper, v, alpha, beta, model_name,eval_method:str="gpt", file_name:str="result"):
        #评估 TruthFlow 模型（带 flow 修正）。
        """evaluate TruthFlow"""
        if eval_method == "gpt":
            print("Using GPT-4 to evaluate...")

        elif eval_method == "bleurt":
            print("Using BLEURT to evaluate...")
            bleurt, bleurt_tokenizer = load_bleurt(device)
            bleurt.eval()

        else:
            raise ValueError("Invalid evaluation method. Please choose between 'gpt' and 'bleurt'.")
        
        total_num = len(self.test_ds)
        true_score = 0
        truthful_labels = []
        for data in tqdm(self.test_ds):
            with torch.no_grad():
                outputs = self.model(input_ids=data["input_ids"].to(self.device), output_hidden_states=True)

            original_layers = []
            for idx, layer in enumerate(self.layers):
                hs = outputs.hidden_states[layer][:, -1, :]# hidden_states[layer] shape:[1,29,3584],然后取最后一个token的激活
                hs_flow = flow[idx].sample(hidden_states=hs) # shape:[1,3584]，也就是输出，预测结果

                original_layers.append(deepcopy(self.model.model.layers[layer]))
                self.model.model.layers[layer] = wrapper(self.model.model.layers[layer], model_name,hs_flow[0], v.to(self.device), k=self.k, alpha=alpha)#r 是一个 包装器，用于在模型层中注入 修正操作,对于每一个sample，都要重新构建一个包装器。因为steering vector是会变的
            self.model.eval()

            with torch.no_grad():
                outputs = self.model.generate(input_ids=data["input_ids"].to(self.device), do_sample=False, top_k=0, top_p=1.0, temperature=0, return_dict_in_generate=True, max_new_tokens=256, pad_token_id=self.tokenizer.eos_token_id)

            for idx, layer in enumerate(self.layers):
                self.model.model.layers[layer] = original_layers[idx]#把模型复原，没有包装器

            # evaluate answers
            if eval_method == "gpt":
                if self.eval_ds_name == "tqa":
                    true = tqa_gpt_eval_true(data["question"], data["correct_answers"], data["incorrect_answers"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "halueval":
                    true = halueval_gpt_eval_true(data["question"], data['knowledge'], data["right_answer"], data["hallucinated_answer"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "nq":
                    true = nq_gpt_eval_true(data["question"], data["answer"], data["false_answer"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "triviaqa":
                    true = triviaqa_gpt_eval_true(data["question"], data["correct_answers"], data["incorrect_answers"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                else:
                    raise ValueError("Invalid evaluation dataset. Please choose between 'tqa', 'halueval', 'triviaqa', and 'nq'.")
            elif eval_method == "bleurt":
                true = bleurt_eval(bleurt, bleurt_tokenizer, self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True), data["correct_answers"], data["incorrect_answers"])
            else:
                raise ValueError("Invalid evaluation method. Please choose between 'gpt' and 'bleurt'.")
            
            true_score += true
            truthful_labels.append(true)

            # save qa+true_score to csv
            write_to_csv(self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True), true, file_name+".csv")#有结果和回答的csv
            # save answers to csv
            write_to_csv(self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True), true, file_name+"_answer.csv")#只有答案的csv
        log_file = "evaluation_log.txt"
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_message = f"{current_time} - {eval_method} true score: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}\n"

        with open(log_file, "a") as f:
            f.write(log_message)
        print(f"{eval_method} true score: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}")
        write_to_csv(f"{eval_method} true score: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}", None, file_name+".csv")
    
    def dola_eval_pipeline(self, eval_method:str="gpt", file_name:str="dola_result"):
        """evaluate DOLA 评估 DOLA 方法（Decoding by Overthinking and LookAhead）。
↳

不经过 flow 修正，而是用 LLM 的 generate() 并开启特殊参数 dola_layers='high' 来影响生成策略"""
        if eval_method == "gpt":
            print("Using GPT-4 to evaluate...")

        elif eval_method == "bleurt":
            print("Using BLEURT to evaluate...")
            bleurt, bleurt_tokenizer = load_bleurt(device)
            bleurt.eval()

        else:
            raise ValueError("Invalid evaluation method. Please choose between 'gpt' and 'bleurt'.")
        
        total_num = len(self.test_ds)
        true_score = 0
        truthful_labels = []
        for data in tqdm(self.test_ds):
            with torch.no_grad():
                outputs = self.model.generate(input_ids=data["input_ids"].to(self.device), do_sample=False, dola_layers='high',repetition_penalty=1.2, max_new_tokens=256, pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True)

            # evaluate answers
            if eval_method == "gpt":
                if self.eval_ds_name == "tqa":
                    true = tqa_gpt_eval_true(data["question"], data["correct_answers"], data["incorrect_answers"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "halueval":
                    true = halueval_gpt_eval_true(data["question"], data['knowledge'], data["right_answer"], data["hallucinated_answer"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "nq":
                    true = nq_gpt_eval_true(data["question"], data["answer"], data["false_answer"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "triviaqa":
                    true = triviaqa_gpt_eval_true(data["question"], data["correct_answers"], data["incorrect_answers"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                else:
                    raise ValueError("Invalid evaluation dataset. Please choose between 'tqa', 'halueval', 'triviaqa', and 'nq'.")
            elif eval_method == "bleurt":
                true = bleurt_eval(bleurt, bleurt_tokenizer, self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True), data["correct_answers"], data["incorrect_answers"])
            else:
                raise ValueError("Invalid evaluation method. Please choose between 'gpt' and 'bleurt'.")
            true_score += true
            truthful_labels.append(true)

            # save to csv
            write_to_csv(self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True), true, file_name+".csv")
            # save answers to csv
            write_to_csv(self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True), true, file_name+"_answer.csv")
        log_file = "evaluation_log.txt"
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_message = f"{current_time} - {eval_method} true score: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}\n"

        with open(log_file, "a") as f:
            f.write(log_message)
        print(f"{eval_method} true score: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}")
        write_to_csv(f"{eval_method} true score: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}", None, file_name+".csv")
        
    def base_eval_pipeline(self, eval_method:str="gpt", file_name:str="result"):
        """evaluate base LLM
        评估原始 LLM（不修正）。

就是直接跑 generate()，然后用 GPT-4/BLEURT 对比答案和 ground truth。"""
        if eval_method == "gpt":
            print("Using GPT-4 to evaluate...")

        elif eval_method == "bleurt":
            print("Using BLEURT to evaluate...")
            bleurt, bleurt_tokenizer = load_bleurt(device)
            bleurt.eval()

        else:
            raise ValueError("Invalid evaluation method. Please choose between 'gpt' and 'bleurt'.")
        
        total_num = len(self.test_ds)
        true_score = 0
        truthful_labels = []
        for data in tqdm(self.test_ds):
            with torch.no_grad():#input_ids shape:[1,length]
                outputs = self.model.generate(input_ids=data["input_ids"].to(self.device), do_sample=False, top_k=0, top_p=1.0, temperature=0, max_new_tokens=256, pad_token_id=self.tokenizer.eos_token_id, return_dict_in_generate=True)
            # evaluate answers
            if eval_method == "gpt":
                if self.eval_ds_name == "tqa":
                    true = tqa_gpt_eval_true(data["question"], data["correct_answers"], data["incorrect_answers"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "halueval":
                    true = halueval_gpt_eval_true(data["question"], data['knowledge'], data["right_answer"], data["hallucinated_answer"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "nq":
                    true = nq_gpt_eval_true(data["question"], data["answer"], data["false_answer"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                elif self.eval_ds_name == "triviaqa":
                    true = triviaqa_gpt_eval_true(data["question"], data["correct_answers"], data["incorrect_answers"], self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True))
                else:
                    raise ValueError("Invalid evaluation dataset. Please choose between 'tqa', 'halueval', 'triviaqa', and 'nq'.")
            elif eval_method == "bleurt":
                true = bleurt_eval(bleurt, bleurt_tokenizer, self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True), data["correct_answers"], data["incorrect_answers"])
            else:
                raise ValueError("Invalid evaluation method. Please choose between 'gpt' and 'bleurt'.")
            true_score += true
            truthful_labels.append(true)

            # save to csv
            write_to_csv(self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True), true, file_name+".csv")
            # save answers to csv
            write_to_csv(self.tokenizer.decode(outputs.sequences[0][data["input_ids"].shape[1]:], skip_special_tokens=True), true, file_name+"_answer.csv")
        log_file = "evaluation_log.txt"
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_message = f"{current_time} - {eval_method} true score: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}\n"

        with open(log_file, "a") as f:
            f.write(log_message)
        print(f"Accuracy: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}")
        write_to_csv(f"Accuracy: {true_score/total_num}, Total number: {total_num}, Truthful number: {true_score}", None, file_name+".csv")


class MCEvalPipeline:
    def __init__(self, model, tokenizer, device, layers, test_ds, model_name):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layers = layers
        self.test_ds = test_ds
        self.model_name = model_name
    
    def base_mc_pipeline(self):
        sum_mc1 = 0
        sum_mc2 = 0
        
        for data in tqdm(self.test_ds):
            ref_best = format_best(data)
            format_data = format_c_inc_ans(data)
            ref_true = format_data['correct_answers']
            ref_false = format_data['incorrect_answers']

            scores_true = []
            scores_false = []
            
            query_len = self.tokenizer(data['template_q'], return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
            
            for c_ans in ref_true:
                chat = get_chat(self.model.config.model_type, data['question']) + [{"role": "assistant", "content": c_ans}]
                formatted_chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                tokenized_format_chat = self.tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
                prompt_ids = tokenized_format_chat['input_ids'].to(self.device)

                with torch.no_grad():
                    outputs = self.model(**tokenized_format_chat.to(self.device))[0].squeeze(0)
                    
                outputs = outputs.log_softmax(-1)  # logits to log probs

                outputs = outputs[query_len - 1: -1, :]
                prompt_ids = prompt_ids[0, query_len:]
                log_probs = outputs[range(outputs.shape[0]), prompt_ids.squeeze(0)]
                if self.model_name == "llama-3" or "mistral" in self.model_name:
                    log_probs = log_probs[:-1]
                elif "llama-2" in self.model_name or self.model_name == "gemma-2":
                    log_probs = log_probs[:-2]
                else:
                    log_probs = log_probs[:-1]
                    UserWarning("Please check which token to end for your LLM.")
                scores_true.append(log_probs.sum().item())
                
            for inc_ans in ref_false:
                chat = get_chat(self.model.config.model_type, data['question']) + [{"role": "assistant", "content": inc_ans}]
                formatted_chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                tokenized_format_chat = self.tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
                prompt_ids = tokenized_format_chat['input_ids'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**tokenized_format_chat.to(self.device))[0].squeeze(0)
                    

                outputs = outputs.log_softmax(-1)  # logits to log probs
                outputs = outputs[query_len - 1: -1, :]
                prompt_ids = prompt_ids[0, query_len:]
                log_probs = outputs[range(outputs.shape[0]), prompt_ids.squeeze(0)]
                if self.model_name == "llama-3" or "mistral" in self.model_name:
                    log_probs = log_probs[:-1]
                elif "llama-2" in self.model_name or self.model_name == "gemma-2":
                    log_probs = log_probs[:-2]
                else:
                    log_probs = log_probs[:-1]
                    UserWarning("Please check which token to end for your LLM.")
                scores_false.append(log_probs.sum().item())
                
            mc1, mc2 = MC_calcs(scores_true, scores_false, ref_true, ref_best)  
            sum_mc1 += mc1
            sum_mc2 += mc2
            
        metrics = {'mc_1': sum_mc1/len(self.test_ds), 'mc_2': sum_mc2/len(self.test_ds)}
        print(f"MC1: {metrics['mc_1']}, MC2: {metrics['mc_2']}")
        write_to_csv(f"MC1: {metrics['mc_1']}, MC2: {metrics['mc_2']} {self.model_name} Base", None, "mc_result.csv")
        
            
    def flow_mc_pipeline(self, flow, wrapper, v, alpha, beta):
        sum_mc1 = 0
        sum_mc2 = 0
        for data in tqdm(self.test_ds):
            #! set up for flow
            with torch.no_grad():
                outputs = self.model(input_ids=data["input_ids"].to(self.device), output_hidden_states=True)

            original_layers = []
            for idx, layer in enumerate(self.layers):
                hs = outputs.hidden_states[layer][:, -1, :]
                hs_flow = flow[idx].sample(hidden_states=hs)

                original_layers.append(deepcopy(self.model.model.layers[layer]))
                self.model.model.layers[layer] = wrapper(self.model.model.layers[layer], hs_flow[0], v.to(self.device), alpha=alpha, beta=beta)
            self.model.eval()
            
            #! mc eval
            ref_best = format_best(data)
            format_data = format_c_inc_ans(data)
            ref_true = format_data['correct_answers']
            ref_false = format_data['incorrect_answers']

            scores_true = []
            scores_false = []
            
            query_len = self.tokenizer(data['template_q'], return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
            
            for c_ans in ref_true:
                chat = get_chat(self.model.config.model_type, data['question']) + [{"role": "assistant", "content": c_ans}]
                formatted_chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                tokenized_format_chat = self.tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
                prompt_ids = tokenized_format_chat['input_ids'].to(self.device)

                with torch.no_grad():
                    outputs = self.model(**tokenized_format_chat.to(self.device))[0].squeeze(0)
                    
                outputs = outputs.log_softmax(-1)  # logits to log probs

                outputs = outputs[query_len - 1: -1, :]
                prompt_ids = prompt_ids[0, query_len:]
                log_probs = outputs[range(outputs.shape[0]), prompt_ids.squeeze(0)]
                if self.model_name == "llama-3" or "mistral" in self.model_name:
                    log_probs = log_probs[:-1]
                elif "llama-2" in self.model_name or self.model_name == "gemma-2":
                    log_probs = log_probs[:-2]
                else:
                    log_probs = log_probs[:-1]
                    UserWarning("Please check which token to end for your LLM.")
                scores_true.append(log_probs.sum().item())
                
            for inc_ans in ref_false:
                chat = get_chat(self.model.config.model_type, data['question']) + [{"role": "assistant", "content": inc_ans}]
                formatted_chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
                tokenized_format_chat = self.tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
                prompt_ids = tokenized_format_chat['input_ids'].to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**tokenized_format_chat.to(self.device))[0].squeeze(0)
                    

                outputs = outputs.log_softmax(-1)  # logits to log probs
                outputs = outputs[query_len - 1: -1, :]
                prompt_ids = prompt_ids[0, query_len:]
                log_probs = outputs[range(outputs.shape[0]), prompt_ids.squeeze(0)]
                if self.model_name == "llama-3" or "mistral" in self.model_name:
                    log_probs = log_probs[:-1]
                elif "llama-2" in self.model_name or self.model_name == "gemma-2":
                    log_probs = log_probs[:-2]
                else:
                    log_probs = log_probs[:-1]
                    UserWarning("Please check which token to end for your LLM.")
                scores_false.append(log_probs.sum().item())
                
            mc1, mc2 = MC_calcs(scores_true, scores_false, ref_true, ref_best)  
            sum_mc1 += mc1
            sum_mc2 += mc2
            
            #! reset model
            for idx, layer in enumerate(self.layers):
                self.model.model.layers[layer] = original_layers[idx]
                
        metrics = {'mc_1': sum_mc1/len(self.test_ds), 'mc_2': sum_mc2/len(self.test_ds)}
        print(f"MC1: {metrics['mc_1']}, MC2: {metrics['mc_2']}")
        write_to_csv(f"MC1: {metrics['mc_1']}, MC2: {metrics['mc_2']} {self.model_name} TruthFlow", None, "mc_result.csv")

        