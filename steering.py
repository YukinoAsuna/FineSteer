import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional,Tuple
import torch.nn.functional as F
from sklearn.decomposition import PCA
class SteeringModel(nn.Module):
    """
    s(h) = α(h)*r0 + U_rest @ β(h)
    推理：h' = h + s(h) （无门控，所有样本都施加）
    """
    def __init__(self, input_dim: int, hidden_dim: int, k_rest: int, use_tanh: bool = False):
        super().__init__()
        self.use_tanh = use_tanh
        self.scale = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.mix = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, k_rest)
        )
        self.gamma = nn.Parameter(torch.tensor(1.0))  # 全局强度

    def forward(self, h: torch.Tensor, r0: torch.Tensor, U_rest: torch.Tensor):
        """
        h:      (B,d)
        r0:     (d,)   单位向量
        U_rest: (d,k_rest) 列正交；k_rest=0 时允许传空 (d,0)
        """
        B, d = h.shape
        k_rest = U_rest.shape[1] if U_rest.ndim == 2 else 0

        alpha = self.scale(h)              # (B,1)
        beta  = self.mix(h) if k_rest > 0 else torch.zeros(B, 0, device=h.device, dtype=h.dtype)

        if self.use_tanh:
            alpha = torch.tanh(alpha)      # 稳定幅度（可选）
            beta  = torch.tanh(beta)

        s_r0 = alpha * r0.unsqueeze(0)     # (B,d)
        s_U  = beta @ U_rest.T if k_rest > 0 else torch.zeros_like(s_r0)
        s = self.gamma * (s_r0 + s_U)      # (B,d)
        return alpha, beta, s
    
# class SteeringModel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, k):
#         super(SteeringModel, self).__init__()
        
#         # 门控头：决定是否施加 steering 向量
#         self.gate = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()
#         )
        
#         # 尺度头：控制沿全局向量 r_0 的强度
#         self.scale = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
        
#         # 混合头：在目标子空间 U 内做个性化修正
#         self.mix = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, k)  # k 是目标子空间的维度
#         )

#     def forward(self, h, r_0, U):
#         # 计算门控值
#         g = self.gate(h).squeeze(-1)  # gate output
#         g = torch.clamp(g, 0.0, 1.0)  # 确保 gate 在 [0, 1] 之间
        
#         # 计算尺度和混合头
#         alpha = self.scale(h).squeeze(-1)  # alpha 是沿 r_0 的强度
#         beta = self.mix(h)  # beta 是在 U 子空间的个性化修正
        
#         # 扩展 alpha 为与 r_0 相同的维度，逐元素乘法
#         alpha_expanded = alpha.unsqueeze(-1) * r_0  # alpha 对应每个样本与 r_0 相乘
        
#         # 计算最终的 steering 向量
#         s_raw = alpha_expanded + torch.matmul(beta, U.T)  # r_0 是全局方向，beta 在 U 中
#         return g, alpha, beta, s_raw
def normalize_vec(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)

def orthonormalize_columns(U: torch.Tensor) -> torch.Tensor:
    # 列正交 + 单位化
    Q, _ = torch.linalg.qr(U, mode='reduced')
    return Q

def project_out(v: torch.Tensor, u_unit: torch.Tensor) -> torch.Tensor:
    # 去掉 v 在单位向量 u_unit 上的分量
    coeff = (v @ u_unit)  # (...,)
    return v - coeff.unsqueeze(-1) * u_unit

def make_projection_mats(U_full: torch.Tensor):
    # U_full: (d,k)，列正交单位
    P = U_full @ U_full.T
    I = torch.eye(U_full.size(0), device=U_full.device, dtype=U_full.dtype)
    return P, I - P

def build_truth_subspace(
    Hc: torch.Tensor,               # (M, d) 正确回复隐藏态
    Hi: torch.Tensor,               # (M, d) 或 (M, K, d) 错误回复隐藏态
    k: int = 32,                    # 子空间维数（含 r0）
    center_residual: bool = True,   # 是否对残差做去均值
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    返回:
      r0: (d,) 单位向量，全局真相方向
      U:  (d, k) 列为正交基，第一列是 r0
    说明:
      - 若 Hi 是 (M, K, d)，会先对 K 个负例取均值。
      - 在残差上做 PCA，取前 k-1 个方向，最后与 r0 拼接并正交化。
    """
    assert Hc.ndim == 2, "Hc must be (M, d)"
    M, d = Hc.shape

    if Hi.ndim == 3:
        Hi_mean = Hi.mean(dim=1)  # (M, d)
    elif Hi.ndim == 2:
        Hi_mean = Hi
    else:
        raise ValueError("Hi must be (M, d) or (M, K, d)")

    if device is None:
        device = Hc.device
    if dtype is None:
        dtype = Hc.dtype

    # 1) 差分与全局真相方向 r0
    Hc=Hc.to(device)
    Hi_mean=Hi_mean.to(device)
    Delta = Hc - Hi_mean                 # (M, d)
    delta_mean = Delta.mean(dim=0)       # (d,)
    r0 = delta_mean / (delta_mean.norm() + 1e-12)  # (d,)

    # 2) 去 r0 分量后的残差
    R = project_out(Delta, r0)           # (M, d)
    if center_residual:
        R = R - R.mean(dim=0, keepdim=True)

    # 3) 在残差上做 PCA，取 k-1 个方向（如果 k=1 则只有 r0）
    k = max(1, min(k, d))
    k_rest = max(0, k - 1)

    if k_rest > 0:
        R_np = R.detach().cpu().numpy()
        pca = PCA(n_components=k_rest, svd_solver="auto")
        pca.fit(R_np)
        U_rest = torch.from_numpy(pca.components_.T).to(device=device, dtype=dtype)  # (d, k-1)

        # 保险：确保 U_rest 与 r0 正交，且列正交单位
        # 先把 r0 分量去掉再 QR
        U_rest = U_rest - r0[:, None] * (r0 @ U_rest)
        U_rest = orthonormalize_columns(U_rest)
        U = torch.cat([r0[:, None], U_rest], dim=1)  # (d, k)
    else:
        U = r0[:, None]

    # 再做一次轻微的正交化（保持 r0 不变）
    # 用 Gram-Schmidt 对其余列做正交，r0 作为第一列固定
    if U.shape[1] > 1:
        U2 = U.clone()
        U2[:, 0] = r0
        for j in range(1, U2.shape[1]):
            v = U2[:, j]
            v = project_out(v, r0)
            # 与之前列正交
            for i in range(1, j):
                vi = U2[:, i]
                v = v - vi * (vi @ v)
            v = v / (v.norm() + 1e-12)
            U2[:, j] = v
        U = U2

    return r0, U  # r0:(d,), U:(d,k)

def get_vector_and_space(train_ds,layer,k):
    hc_list = []
    hi_list = []
    for item in train_ds:
        # 提取 hc_layer20 并添加到 hc_list
        hc_list.append(item[f'hc_layer{layer}'])
        # 提取 hi_layer20 并添加到 hi_list
        hi_list.append(item[f'hi_layer{layer}'])
    hc_tensor = torch.stack(hc_list).squeeze(1)
    hi_tensor = torch.stack(hi_list).squeeze(1)
    r,U=build_truth_subspace(hc_tensor,hi_tensor,k)
    return r,U
