import torch
import torch.nn as nn

def proj_svd(v, y, k, ref):
    v = v.to(device=ref.device, dtype=ref.dtype)
    y = y.to(device=ref.device, dtype=ref.dtype)#shape:[3584]
    vk = v[:, :k]#：取出 v 的前 k 列，即保留 前 k 个最重要的奇异向量.shape:[3584,k]
    return vk @ (vk.transpose(0, 1) @ y)#返回的shape [3584]
    #括号内的部分是将 y 投影到 由前 k 个奇异向量组成的子空间上，括号外的部分是将投影结果重新映射回原始空间。
class Wrapper(nn.Module):
    def __init__(self, block, model_name,vec, v, k=20, alpha=2.0):
        super().__init__()
        self.block = block
        self.k = k
        self.alpha = alpha
        self.register_buffer('vec', vec)  # 不训练就用 buffer,就是flow-matching的输出结果，shape:[3584]
        self.register_buffer('v', v)#就是之前算得到的特征向量构成的矩阵,shape:[3584,408]
        self.model_name=model_name
    # 关键：把未知属性（如 attention_type）透传给原 block
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)

    def forward(self, *args, **kwargs):
        outputs = self.block(*args, **kwargs)      # Gemma2DecoderLayer 通常返回 tuple.args就是一个，tensor，shape:[1,29,3584]
        hidden_states = outputs[0]                 # (B, T, H),output是一个tuple.所以要提出来，shape:[1,29,3584]

        hc_proj = proj_svd(self.v, self.vec.view(-1), self.k, hidden_states)  # (H,)#proj_svd：这是一个投影函数，核心功能是将 hidden states 投影到一个由奇异向量组成的低维空间，从而进行流量修正。
        steer = (self.alpha * hc_proj).to(hidden_states).view(1, 1, -1)       # (1,1,H)
        hidden_states = hidden_states + steer
        if "llama" in self.model_name or "qwen" in self.model_name:
            return hidden_states
        else:
            return (hidden_states, *outputs[1:])       # 保持返回为 tuple

class Wrapper_Alphasteer(nn.Module):
    def __init__(self, block, model_name,steering_matrix, alpha=2.0):
        super().__init__()
        self.block = block
        self.alpha = alpha
        # self.register_buffer('vec', vec)  # 不训练就用 buffer,就是flow-matching的输出结果，shape:[3584]
        # self.register_buffer('v', v)#就是之前算得到的特征向量构成的矩阵,shape:[3584,408]
        self.register_buffer("s",steering_matrix)
        self.model_name=model_name
    # 关键：把未知属性（如 attention_type）透传给原 block
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)

    def forward(self, *args, **kwargs):
        outputs = self.block(*args, **kwargs)      # Gemma2DecoderLayer 通常返回 tuple.args就是一个，tensor，shape:[1,29,3584]
        if "llama" in self.model_name or "qwen" in self.model_name:
            hidden_states=outputs
        else:
            hidden_states = outputs[0]                 # (B, T, H),output是一个tuple.所以要提出来，shape:[1,29,3584]
        steering_vector = hidden_states[:, -1, :] @ self.s * self.alpha
        steering_vector=steering_vector.unsqueeze(1)
        hidden_states = hidden_states + steering_vector

        if "llama" in self.model_name or "qwen" in self.model_name:
            return hidden_states
        else:
            return (hidden_states, *outputs[1:])       # 保持返回为 tuple
    
    
class Wrapper_Steer(nn.Module):
    def __init__(self, block, model_name,vec, alpha=2.0):
        super().__init__()
        self.block = block
        self.alpha = alpha
        self.register_buffer('vec', vec)  # 不训练就用 buffer,就是flow-matching的输出结果，shape:[3584
        self.model_name=model_name
    # 关键：把未知属性（如 attention_type）透传给原 block
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)

    def forward(self, *args, **kwargs):
        outputs = self.block(*args, **kwargs)      # Gemma2DecoderLayer 通常返回 tuple.args就是一个，tensor，shape:[1,29,3584]
        if "llama" in self.model_name or "qwen" in self.model_name:
            hidden_states=outputs
        else:
            hidden_states = outputs[0]                 # (B, T, H),output是一个tuple.所以要提出来，shape:[1,29,3584]
        #hidden_states = outputs[0]                 # (B, T, H),output是一个tuple.所以要提出来，shape:[1,29,3584]

       
        # steer = (self.alpha * hc_proj).to(hidden_states).view(1, 1, -1)       # (1,1,H)
        hidden_states = hidden_states + self.alpha*self.vec
        
        if "llama" in self.model_name or "qwen" in self.model_name:
            return hidden_states
        else:
            return (hidden_states, *outputs[1:])       # 保持返回为 tuple