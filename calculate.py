import torch
import os
from datasets import load_from_disk
from utils.steering_utils import *
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
# Load the dataset
ds = load_from_disk("data_tqa/qwen2.5_ans_avg_seed0_testsize0.5_layers_10_11_12_13_14_15_16_17_18_19_20")
train_ds = ds["train"]
test_ds = ds["test"]
print("starting...")
# Extract layer data
def extract_layer_data(train_ds, layer):
    hc = train_ds[f"hc_layer{layer}"]
    hi = train_ds[f"hi_layer{layer}"]
    hc_tensor = torch.stack([torch.tensor(x) for x in hc]).squeeze(1)
    hi_tensor = torch.stack([torch.tensor(x) for x in hi]).squeeze(1)
    return hc_tensor, hi_tensor
print("extracting")
hc18_tensor, hi18_tensor = extract_layer_data(train_ds, 12)
# hc20_tensor, hi20_tensor = extract_layer_data(train_ds, 20)
# hc22_tensor, hi22_tensor = extract_layer_data(train_ds, 22)

# Stack all layers together
def stack_layers(*tensors):
    return torch.stack(tensors, dim=0).permute(1, 0, 2)

# hc_all = stack_layers(hc18_tensor, hc20_tensor, hc22_tensor)
# hi_all = stack_layers(hi18_tensor, hi20_tensor, hi22_tensor)

hc_all = stack_layers(hc18_tensor)
hi_all = stack_layers(hi18_tensor)
# Extract and calculate the mean of y_win_layer
def calculate_layer_mean(train_ds, layer):
    y_win_tensor = torch.tensor(train_ds[f'y_win_layer{layer}']).squeeze(1)
    return torch.mean(y_win_tensor, dim=0)

print("calculating")
mean_18 = calculate_layer_mean(train_ds, 12)
# mean_20 = calculate_layer_mean(train_ds, 20)
# mean_22 = calculate_layer_mean(train_ds, 22)

# Stack all means
# mean_all = torch.stack([mean_18, mean_20, mean_22], dim=0)
mean_all = mean_18.unsqueeze(0)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize steering matrices
num_layer, d_model = mean_all.shape
P = torch.zeros(num_layer, d_model, d_model, device=device)  # [32, 4096, 4096]
tilde_delta = torch.zeros(num_layer, d_model, d_model, device=device)  # [32, 4096, 4096]
steering_matrix = torch.zeros(num_layer, d_model, d_model, device=device)  # [32, 4096, 4096]

# Layer and ratio configuration
# layers_ratio_list = [(0, 0.35), (1, 0.75), (2, 0.35)]  # Example layers with ratios
layers_ratio_list = [(0, 1)]  # Example layers with ratios
H_benign_train=hc_all.to(torch.float32)
H_harmful_train=hi_all.to(torch.float32)
refusal_vectors=mean_all.to(torch.float32)
# Calculate steering matrices for each layer
def calculate_steering_matrix(layer, ratio):
    print(f"layer: {layer}, ratio: {ratio}")

    # Step 1: Calculate Null Space Projection Matrix
    P_layer = null_space_projection_l(H_benign_train[:, layer, :], abs_nullspace_ratio=ratio)
    P[layer] = P_layer
    P_norm = torch.norm(P_layer)
    print(f"P_norm: {P_norm}")

    # Step 2: Calculate tilde delta matrix with regularization
    tilde_delta_layer = cal_tilde_delta_with_regularization_l(
        H_harmful_train[:, layer, :], P_layer, refusal_vectors[layer], lambda_reg=10.0, device=device)
    tilde_delta[layer] = tilde_delta_layer
    tilde_delta_norm = torch.norm(tilde_delta_layer)
    print(f"tilde_delta_norm: {tilde_delta_norm}")

    # Step 3: Calculate final steering matrix
    steering_matrix_layer = cal_steering_matrix_l(
        P_layer, tilde_delta_layer, device=device)
    steering_matrix[layer] = steering_matrix_layer

    steering_matrix_norm = torch.norm(steering_matrix_layer)
    print(f"steering matrix layer {layer} norm: {steering_matrix_norm}")

# Run the steering matrix calculation for each layer
for layer, ratio in layers_ratio_list:
    calculate_steering_matrix(layer, ratio)

# Save the steering matrix to a file
os.makedirs(os.path.dirname("./qwen2.5_steering_matrix_12.pt"), exist_ok=True)
torch.save(steering_matrix, "./qwen2.5_steering_matrix_12.pt")
