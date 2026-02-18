# import torch
# from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

# config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20')
# model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20',use_safetensors=True)
# tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
from bleurt_pytorch.bleurt.tokenization_bleurt import BleurtSPTokenizer
from bleurt_pytorch.bleurt.modeling_bleurt import BleurtForSequenceClassification
import torch
print("loading")
tokenizer = BleurtSPTokenizer.from_pretrained("lucadiliello/BLEURT-20")  # 或本地路径
model = BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20",use_safetensors=True).to("cuda")
references = ["a bird chirps by the window", "this is a random sentence"]
candidates = ["a bird chirps by the window", "this looks like a random sentence"]

model.eval()
print("evaluation")
with torch.no_grad():
    inputs = tokenizer(references, candidates, padding='longest', return_tensors='pt').to("cuda")
    res = model(**inputs).logits.flatten().tolist()
print(res)
# [0.9604414105415344, 0.8080050349235535]