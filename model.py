import math

import torch
import torch.nn as nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_timescale=10000):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_timescale = max_timescale

    def forward(self, t):
        position = t.unsqueeze(1)  # Shape: (batch_size, 1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2) * (-math.log(self.max_timescale) / self.embedding_dim)).to(t.device)
        
        pos_embedding = torch.zeros(t.shape[0], self.embedding_dim, device=t.device)
        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_embedding  # Shape: (batch_size, embedding_dim)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x += residual
        return self.relu(x)


class LinearUNet(nn.Module):
    def __init__(self, hid_dim=4096, depth=4, feature_scale=0.5, time_embedding_dim=128):
        super(LinearUNet, self).__init__()
        self.depth = depth
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding
        self.time_embedding = SinusoidalPositionalEmbedding(embedding_dim=time_embedding_dim)

        # Downsampling path
        self.down_layers = nn.ModuleList()
        for i in range(depth):
            in_dim = int(hid_dim * (feature_scale ** i)) + time_embedding_dim
            out_dim = int(hid_dim * (feature_scale ** (i + 1)))
            self.down_layers.append(ResidualBlock(in_dim, out_dim))

        # Bottleneck layer
        self.bottleneck = ResidualBlock(int(hid_dim * (feature_scale ** depth)) + time_embedding_dim, int(hid_dim * (feature_scale ** depth)))
        
        # Upsampling path
        self.up_layers = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            in_dim = int(hid_dim * (feature_scale ** (i + 1))) * 2 + time_embedding_dim
            out_dim = int(hid_dim * (feature_scale ** i))
            self.up_layers.append(ResidualBlock(in_dim, out_dim))

        # Final output layer
        self.final_layer = nn.Linear(hid_dim, hid_dim)

    def forward(self, x, times):
        # Compute time embedding
        time_emb = self.time_embedding(times)

        down_features = []

        # Downsampling
        for down in self.down_layers:
            x = torch.cat([x, time_emb], dim=-1)
            x = down(x)
            down_features.append(x)
        
        # Bottleneck
        x = torch.cat([x, time_emb], dim=-1)
        x = self.bottleneck(x)

        # Upsampling
        for up in self.up_layers:
            down_feat = down_features.pop()
            x = torch.cat([x, down_feat, time_emb], dim=-1)
            x = up(x)
        
        # Final layer
        x = self.final_layer(x)
        
        return x




# calculate the size of the model
def calculate_params_and_size(model):
    total_params = 0
    total_size = 0  # in bytes

    for param in model.parameters():
        param_size = param.numel() * param.element_size()  # number of elements * element size in bytes
        total_params += param.numel()
        total_size += param_size

    print(f"Total Parameters: {total_params}")
    print(f"Total Size (in bytes): {total_size}")
    print(f"Total Size (in MB): {total_size / (1024**2):.2f}")

