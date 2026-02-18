import json
from datasets import Dataset
import torch
from tqdm import tqdm
from copy import deepcopy
import argparse
import os
import pickle
import yaml
import tempfile
from wrapper import Wrapper,Wrapper_Alphasteer,Wrapper_Steer
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
class UnifiedEvalPipeline:
    def __init__(self, 
                 args=None,
                 input_file=None, 
                 output_file="eval_results.json", 
                 model_name="llama3.1", 
                 device="cuda", 
                 layers=None, 
                 epochs=0,
                 k=20,
                 steer_model_dir="./steer_models",
                 method_name="steer",
                 train_ds=None):
        self.input_file = input_file
        self.output_file = output_file
        self.model_name = model_name
        self.device = device
        self.layers = layers
        if not isinstance(self.layers, list):
            self.layers = [self.layers] 
        self.k = k
        self.method_name=method_name
        self.epochs = epochs
        self.steer_model_dir = steer_model_dir
        self.train_ds = train_ds
        self.model = None
        self.tokenizer = None
        self.test_dataset = None
        self.steers = []

        self.prompt_column=args.prompt_column
        self.max_new_tokens=args.max_new_tokens
        # 运行期缓存：已存在的结果与已完成的 index
        self._existing_results = []
        self._done_indices = set()

    def load_model_and_tokenizer(self):
        from utils.utils import load_model_and_tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_name, self.device)
        return self.model, self.tokenizer

    def load_steers(self):
        from steering import SteeringModel
        from rectified_flow import RectifiedFlow
        from model import LinearUNet
        print(f"开始加载steer模型，共 {len(self.layers)} 层...")
        self.steers = []
        
        for layer in self.layers:
            if self.method_name=="steer":
                print(f"Loading steer model for layer {layer}...")
                save_full_model_path = os.path.join(
                    "model_weights",
                    f"{self.model_name}_results",
                    f"steer_epochs_{self.epochs}_k_{self.k}_layer_{layer}_full.pkl"
                )
                if os.path.exists(save_full_model_path):
                    print(f"Loading full model info for layer {layer} from {save_full_model_path}")
                    with open(save_full_model_path, 'rb') as f:
                        full_model_info = pickle.load(f)
                    hid_dim = full_model_info['hid_dim']
                    k_rest = full_model_info['model_config']['k_rest']
                    print(f"Model parameters - hid_dim: {hid_dim}, k_rest: {k_rest}")
                    steerModel = SteeringModel(input_dim=hid_dim, hidden_dim=hid_dim, k_rest=k_rest)
                    steerModel.load_state_dict(full_model_info['steerModel_state_dict'])
                    r = full_model_info['r']
                    U_rest = full_model_info['U_rest']
                    print(f"Loaded full model info - Layer: {full_model_info['layer']}, K: {full_model_info['k']}")
                else:
                    raise FileNotFoundError(f"Neither {save_full_model_path} exists")
                steerModel.to(self.device)
                steerModel.eval()
                self.steers.append([steerModel, r.to(self.device), U_rest.to(self.device)])
            elif self.method_name=="truthflow":
                hid_dim = self.model.config.hidden_size
                print(f"Loading TruthFlow model for layer {layer}...")
                unet = LinearUNet(
                    hid_dim=hid_dim,
                    depth=4,
                    feature_scale=0.5,
                    time_embedding_dim=128,
                ).to(self.device)
                rectified_flow = RectifiedFlow(unet, data_shape=(hid_dim,))
                save_full_model_path = os.path.join(
                    self.model_name+"_tqa_results",
                    f"TruthFlow_{self.model_name}_seed0_epoch{self.epochs}_{layer}.pth"
                )
                if os.path.exists(save_full_model_path):
                    print(f"Loading full model info for layer {layer} from {save_full_model_path}")
                    with open(save_full_model_path, 'rb') as f:
                        full_model_info = pickle.load(f)
                    alpha=full_model_info['alpha']
                    v=full_model_info['r']
                    k=full_model_info['k']
                    rectified_flow.load_state_dict(full_model_info['steerModel_state_dict'])
                    rectified_flow.eval()
                    self.steers.append([rectified_flow,v,alpha,k])
                    # with open(save_full_model_path, 'rb') as f:
                    #     full_model_info = pickle.load(f)
                    # hid_dim = full_model_info['hid_dim']
                    # k_rest = full_model_info['model_config']['k_rest']
                    # print(f"Model parameters - hid_dim: {hid_dim}, k_rest: {k_rest}")
                    # steerModel = SteeringModel(input_dim=hid_dim, hidden_dim=hid_dim, k_rest=k_rest)
                    # steerModel.load_state_dict(full_model_info['steerModel_state_dict'])
                    # r = full_model_info['r']
                    # U_rest = full_model_info['U_rest']
        print(f"成功加载 {len(self.steers)} 个steer模型")
        return self.steers

    def load_json_file(self, file_path=None):
        file_path = file_path or self.input_file
        if not file_path:
            print("错误：未提供输入文件路径")
            return None
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, list) and all(isinstance(item, dict) and self.prompt_column in item for item in data):
                    prompts_list = [item[self.prompt_column] for item in data]
                    print(f"成功加载 {len(prompts_list)} 个prompts。")
                    return prompts_list
                else:
                    print(f"错误：文件 '{file_path}' 的格式不符合预期。")
                    return None
        except FileNotFoundError:
            print(f"错误：找不到文件 '{file_path}'")
            return None
        except json.JSONDecodeError:
            print(f"错误：文件 '{file_path}' 不是有效的JSON格式")
            return None
        except Exception as e:
            print(f"加载JSON文件时发生错误：{e}")
            return None

    def create_test_dataset(self, prompts):
        from utils.utils import get_chat
        data_dict = {'chat': [], 'formatted_chat': [], 'input_ids': []}
        print(f"正在创建测试数据集，共处理 {len(prompts)} 个样本...")
        for prompt in tqdm(prompts):
            chat = get_chat(self.model_name, prompt)
            formatted_chat = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs['input_ids'].squeeze().tolist()
            data_dict['chat'].append(chat)
            data_dict['formatted_chat'].append(formatted_chat)
            data_dict['input_ids'].append(input_ids)
        self.test_dataset = Dataset.from_dict(data_dict)
        def encode(example):
            return self.tokenizer(example['formatted_chat'],return_tensors="pt", add_special_tokens=False)
        self.test_dataset = self.test_dataset.map(encode)
        self.test_dataset.set_format(type='torch', columns=["chat", "formatted_chat", "input_ids"])
        print(f"测试数据集创建完成，共包含 {len(self.test_dataset)} 个样本。")
        
        return self.test_dataset

    # ========== 新增：增量保存相关工具 ==========
    def _atomic_write_json(self, obj, path):
        """原子写：写到同目录临时文件，再替换，避免写一半损坏。"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dir_name = os.path.dirname(path) or "."
        with tempfile.NamedTemporaryFile('w', delete=False, dir=dir_name, encoding='utf-8') as tmp:
            json.dump(obj, tmp, ensure_ascii=False, indent=2)
            tmp_path = tmp.name
        os.replace(tmp_path, path)

    def _load_existing_results(self, output_file):
        """读取已有结果，填充 _existing_results 与 _done_indices（基于result['index']）。"""
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._existing_results = data
                    self._done_indices = {item.get("index") for item in data if isinstance(item, dict) and "index" in item}
                    print(f"检测到已有结果文件：{output_file}，已完成 {len(self._done_indices)} 条，将跳过这些样本。")
                else:
                    print(f"警告：{output_file} 内容不是列表，忽略旧内容。")
                    self._existing_results = []
                    self._done_indices = set()
            except Exception as e:
                print(f"警告：读取已有结果失败（{e}），将从空结果开始。")
                self._existing_results = []
                self._done_indices = set()
        else:
            self._existing_results = []
            self._done_indices = set()

    def save_result_item_incremental(self, item, output_file):
        """增量保存单条结果：读旧 → 追加/去重更新 → 原子写回。"""
        # 如果 index 已存在，则做“更新”（替换）；否则追加
        idx = item.get("index")
        replaced = False
        for i, old in enumerate(self._existing_results):
            if isinstance(old, dict) and old.get("index") == idx:
                self._existing_results[i] = item
                replaced = True
                break
        if not replaced:
            self._existing_results.append(item)
        # 更新已完成集合
        if idx is not None:
            self._done_indices.add(idx)
        # 原子写回
        self._atomic_write_json(self._existing_results, output_file)

    # ========== 评估主流程 ==========
    def flow_eval_pipeline(self,wrapper,alpha, eval_method:str="gpt",output_file=None):
        if self.test_dataset is None:
            print("错误：请先创建测试数据集")
            return []
        if not self.steers:
            print("错误：请先加载steer模型")
            return []

        results_dir = os.path.join("results", self.model_name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"创建目录: {results_dir}")

        # 动态输出文件名（沿用你的命名规则）
        output_file = os.path.splitext(os.path.basename(self.input_file))[0] + f"_TruthFlow.json"
        output_file = os.path.join(results_dir, output_file)

        # 加载已有结果（实现断点续跑）
        self._load_existing_results(output_file)

        print(f"开始评估模型（增量保存）...")
        total_num = len(self.test_dataset)

        # 返回值仍返回内存聚合的列表副本（读取已有 + 新增后的）
        for idx, data in enumerate(tqdm(self.test_dataset)):
            if idx in self._done_indices:
                # 跳过已完成的 index
                continue

            with torch.no_grad():
                outputs = self.model(input_ids=data["input_ids"].to(self.device), 
                                     output_hidden_states=True)

            original_layers = []
            for layer_idx, layer in enumerate(self.layers):
                hs = outputs.hidden_states[layer][:, -1, :]
                model = self.steers[layer_idx][0]
                hs_steer = model.sample(hidden_states=hs)
                original_layers.append(deepcopy(self.model.model.layers[layer]))
                self.model.model.layers[layer] = wrapper(
                    self.model.model.layers[layer],
                    self.model_name,
                    hs_steer[0],
                    self.steers[layer_idx][1].to(self.device),
                    k=self.steers[layer_idx][3],
                    alpha=self.steers[layer_idx][2]
                )

            with torch.no_grad():
                generate_outputs = self.model.generate(
                    input_ids=data["input_ids"].to(self.device),
                    do_sample=False,
                    top_k=0,
                    top_p=1.0,
                    temperature=0,
                    return_dict_in_generate=True,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_tokens = generate_outputs.sequences[0]
            new_tokens = generated_tokens[data["input_ids"].shape[1]:]
            output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            result_item = {
                "index": idx,
                "chat": data["chat"],
                "output": output_text
            }

            # —— 关键：每生成一条就增量保存 —— #
            try:
                self.save_result_item_incremental(result_item, output_file)
            except Exception as e:
                print(f"保存第 {idx} 条结果时发生错误：{e}")

            # 复原模型层
            for layer_idx, layer in enumerate(self.layers):
                self.model.model.layers[layer] = original_layers[layer_idx]

        print(f"评估完成，结果已增量保存到 {output_file}")
        # 返回内存里的聚合结果副本
        return list(self._existing_results)
    
    def steer_eval_pipeline(self, wrapper, alpha,  eval_method="gpt", output_file=None):
        if self.test_dataset is None:
            print("错误：请先创建测试数据集")
            return []
        if not self.steers:
            print("错误：请先加载steer模型")
            return []

        results_dir = os.path.join("results", self.model_name)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"创建目录: {results_dir}")

        # 动态输出文件名（沿用你的命名规则）
        output_file = os.path.splitext(os.path.basename(self.input_file))[0] + f"_steer_alpha_{alpha}_k_{self.k}.json"
        output_file = os.path.join(results_dir, output_file)

        # 加载已有结果（实现断点续跑）
        self._load_existing_results(output_file)

        print(f"开始评估模型（增量保存）...")
        total_num = len(self.test_dataset)

        # 返回值仍返回内存聚合的列表副本（读取已有 + 新增后的）
        for idx, data in enumerate(tqdm(self.test_dataset)):
            if idx in self._done_indices:
                # 跳过已完成的 index
                continue

            with torch.no_grad():
                outputs = self.model(input_ids=data["input_ids"].to(self.device), 
                                     output_hidden_states=True)

            original_layers = []
            for layer_idx, layer in enumerate(self.layers):
                hs = outputs.hidden_states[layer][:, -1, :]
                model = self.steers[layer_idx][0].half()
                _, _, hs_steer = model(
                    hs,
                    self.steers[layer_idx][1].half().to(self.device),
                    self.steers[layer_idx][2].half().to(self.device)
                )
                original_layers.append(deepcopy(self.model.model.layers[layer]))
                self.model.model.layers[layer] = wrapper(
                    self.model.model.layers[layer],
                    self.model_name,
                    hs_steer,
                    alpha=alpha
                )

            with torch.no_grad():
                generate_outputs = self.model.generate(
                    input_ids=data["input_ids"].to(self.device),
                    do_sample=False,
                    top_k=0,
                    top_p=1.0,
                    temperature=0,
                    return_dict_in_generate=True,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            generated_tokens = generate_outputs.sequences[0]
            new_tokens = generated_tokens[len(data["input_ids"]):]
            output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            result_item = {
                "index": idx,
                "chat": data["chat"],
                "output": output_text
            }

            # —— 关键：每生成一条就增量保存 —— #
            try:
                self.save_result_item_incremental(result_item, output_file)
            except Exception as e:
                print(f"保存第 {idx} 条结果时发生错误：{e}")

            # 复原模型层
            for layer_idx, layer in enumerate(self.layers):
                self.model.model.layers[layer] = original_layers[layer_idx]

        print(f"评估完成，结果已增量保存到 {output_file}")
        # 返回内存里的聚合结果副本
        return list(self._existing_results)

    # 旧的整批保存接口保留但不再使用
    def save_results_to_json(self, results, output_file):
        try:
            self._atomic_write_json(results, output_file)
            print(f"成功将 {len(results)} 条结果保存到 {output_file}")
        except Exception as e:
            print(f"保存结果时发生错误：{e}")

    def run_complete_pipeline(self, wrapper, alpha):
        self.load_model_and_tokenizer()
        self.load_steers()
        prompts = self.load_json_file()
        if prompts:
            self.create_test_dataset(prompts)
            if self.method_name=="steer":
                results = self.steer_eval_pipeline(wrapper, alpha)
            elif self.method_name=="truthflow":
                results=self.flow_eval_pipeline(wrapper,alpha)
            return results
        return []
def load_config(config_path):
    '''
    Expected arguments in config:
        device: device to use
        model_name: model name
        steering_matrix_path: path to steering matrix
        input_file: path to input file
        output_file: path to output file
        batch_size: batch size
        max_new_tokens: maximum number of tokens to generate
        prompt_column: name of prompt column
    '''
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
def main():
    parser = argparse.ArgumentParser(description="Run a unified model evaluation pipeline with configurable parameters.")
    parser.add_argument("--model_name", type=str, default="llama3.1", help="Name of the model to use for evaluation.")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    #parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file containing prompts.")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json", help="(已保留但当前按自动命名保存)")
    #parser.add_argument("--layers", type=int, nargs='+', default=[15], help="A list of model layers to evaluate.")
    parser.add_argument("--alpha", type=float, default=1.5, help="The alpha parameter for the evaluation process.")
    #parser.add_argument("--epochs", type=int, default=1, help="The beta parameter for the evaluation process.")
    parser.add_argument("--device", type=str, default="cuda", help="The device to run the model on (e.g., 'cuda', 'cpu').")
    parser.add_argument("--steer_model_dir", type=str, default="./steer_models", help="Directory containing the steer model files.")
    parser.add_argument("--k", type=int, default=20, help="The k parameter for steer models.")
    # parser.add_argument('--method', type=str, choices=['truthflow', 'alphasteer', 'base',"steer","dola"], default='base',
    #                     help="The method to use for LLM evaluation.")
    args = parser.parse_args()
    config = load_config(args.config_path)
    for key, value in config.items():
        setattr(args, key, value)
    print("starting")
    print(args)
    eval_pipeline = UnifiedEvalPipeline(
        args=args,
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name,
        device=args.device,
        epochs=args.epochs,
        layers=args.layers,
        steer_model_dir=args.steer_model_dir,
        method_name=args.method,
        k=args.k
    )
    if args.method=="truthflow":
        wrapper=Wrapper
    elif args.method=="steer":
        wrapper = Wrapper_Steer
    elif args.method=="base":
        wrapper=None
    _ = eval_pipeline.run_complete_pipeline(wrapper, alpha=args.alpha)

if __name__ == "__main__":
    main()