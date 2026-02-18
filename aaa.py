from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", device_map="cpu"
)
# Llama 模型在 HF 里通常是 model.model.layers
print("blocks =", len(model.model.layers))
