from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MoSEConfig:
    """Configuration for one of the two supported MoSE implementations."""

    name: str
    normalize_deltas: bool
    residual_space: str  # all_pca | prototype_residual
    value_projection: bool
    selection_rule: str = "calinski"  # calinski | zip_median
    residual_dim: int = 10
    k_min: int = 4
    k_max: int = 10
    pca_components: int = 256
    random_state: int = 42

    @staticmethod
    def preset(name: str = "MoSE", residual_dim: int = 10) -> "MoSEConfig":
        presets = {
            # Equation (11)-(14): normalized deltas, CH-selected fixed centroids,
            # identity value bank, PCA steering basis.
            "MoSE": (True, "all_pca", False, "calinski"),
            # Orthogonal Residual MoSE: remove the prototype span before residual PCA.
            "orthogonal_residual": (False, "prototype_residual", True, "zip_median"),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset {name!r}; choose from {sorted(presets)}")
        normalize, residual_space, value_projection, selection_rule = presets[name]
        return MoSEConfig(
            name=name,
            normalize_deltas=normalize,
            residual_space=residual_space,
            value_projection=value_projection,
            selection_rule=selection_rule,
            residual_dim=residual_dim,
        )


def _safe_components(requested: int, samples: int, features: int) -> int:
    return max(1, min(requested, samples - 1, features))


def _choose_k(x: np.ndarray, cfg: MoSEConfig) -> tuple[int, dict[str, Any]]:
    max_k = min(cfg.k_max, len(x) - 1)
    min_k = min(cfg.k_min, max_k)
    rows: list[dict[str, float | int]] = []
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=10, max_iter=300)
        labels = km.fit_predict(x)
        rows.append(
            {
                "k": k,
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(x, labels)),
                "calinski": float(calinski_harabasz_score(x, labels)),
            }
        )
    if cfg.selection_rule == "calinski":
        chosen = int(max(rows, key=lambda row: row["calinski"])["k"])
    else:
        elbow = rows[0]["k"]
        if len(rows) > 2:
            inertia = np.asarray([row["inertia"] for row in rows], dtype=np.float64)
            denom = max(float(inertia.max() - inertia.min()), 1e-12)
            elbow = rows[int(np.argmax(np.abs(np.diff((inertia - inertia.min()) / denom, n=2)))) + 1]["k"]
        silhouette = max(rows, key=lambda row: row["silhouette"])["k"]
        calinski = max(rows, key=lambda row: row["calinski"])["k"]
        chosen = int(np.median([elbow, silhouette, calinski]))
    return chosen, {"chosen_k": chosen, "scores": rows}


def _prototype_residual(delta: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    # Orthonormal basis of the row span of C, then Delta - Proj_span(C)(Delta).
    q, _ = torch.linalg.qr(prototypes.T.float(), mode="reduced")
    return delta - (delta @ q) @ q.T


def build_mose_components(
    deltas: torch.Tensor, cfg: MoSEConfig
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Construct the fixed prototype bank C and residual basis U_res."""

    raw = deltas.detach().float().cpu()
    cluster_work = F.normalize(raw, dim=-1) if cfg.normalize_deltas else raw
    work_np = cluster_work.numpy()
    n = _safe_components(cfg.pca_components, *work_np.shape)
    clustering_x = PCA(n_components=n, random_state=cfg.random_state).fit_transform(work_np)

    chosen_k, selection = _choose_k(clustering_x, cfg)
    km = KMeans(n_clusters=chosen_k, random_state=cfg.random_state, n_init=20, max_iter=500)
    labels = km.fit_predict(clustering_x)

    # Compute centroids in the original model space. This also avoids losing the
    # PCA mean when mapping projected centers back to hidden space.
    prototypes = torch.stack([cluster_work[torch.from_numpy(labels == i)].mean(0) for i in range(chosen_k)])

    if cfg.residual_space == "prototype_residual":
        residual_source = _prototype_residual(raw, prototypes)
    elif cfg.residual_space == "all_pca":
        # The paper normalizes D_delta only for expert clustering. Equation (13)
        # defines U_res as PCA over the original difference vectors.
        residual_source = raw
    else:
        raise ValueError(f"Unsupported residual space: {cfg.residual_space}")

    residual_source = residual_source - residual_source.mean(0, keepdim=True)
    n_res = _safe_components(cfg.residual_dim, *residual_source.shape)
    pca_res = PCA(n_components=n_res, random_state=cfg.random_state)
    pca_res.fit(residual_source.numpy())
    basis = torch.from_numpy(pca_res.components_.T).float()
    metadata = {
        "implementation_version": 2,
        "config": asdict(cfg),
        "selection": selection,
        "cluster_sizes": [int((labels == i).sum()) for i in range(chosen_k)],
        "residual_explained_variance": float(pca_res.explained_variance_ratio_.sum()),
    }
    return prototypes, basis, metadata


class MoSE(nn.Module):
    """Mixture-of-Steering-Experts from equations (11)-(13)."""

    def __init__(
        self,
        prototypes: torch.Tensor,
        residual_basis: torch.Tensor,
        *,
        value_projection: bool,
        attention_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden = int(prototypes.shape[1])
        attention_dim = attention_dim or hidden
        self.register_buffer("prototypes", prototypes.float())
        self.register_buffer("residual_basis", residual_basis.float())
        self.query = nn.Linear(hidden, attention_dim)
        self.key = nn.Linear(hidden, attention_dim)
        self.value = nn.Linear(hidden, hidden) if value_projection else nn.Identity()
        self.beta = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, residual_basis.shape[1]))
        self.attention_dim = attention_dim

    def forward(self, query_activation: torch.Tensor, return_components: bool = False):
        original_dtype = query_activation.dtype
        h = query_activation.float()
        c = self.prototypes
        scores = self.query(h) @ self.key(c).T / math.sqrt(self.attention_dim)
        alpha = scores.softmax(dim=-1)
        prototype_part = alpha @ self.value(c)
        beta = self.beta(h)
        residual_part = beta @ self.residual_basis.T
        steering = prototype_part + residual_part
        steering = steering.to(original_dtype)
        if return_components:
            return steering, alpha, prototype_part, residual_part
        return steering


def train_mose(
    model: MoSE,
    query_activations: torch.Tensor,
    deltas: torch.Tensor,
    *,
    epochs: int = 100,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-5,
    patience: int = 8,
    seed: int = 0,
) -> dict[str, Any]:
    """Train AGN and residual coefficients against observed representation shifts."""

    torch.manual_seed(seed)
    device = next(model.parameters()).device
    x = query_activations.float().to(device)
    y = deltas.float().to(device)
    order = torch.randperm(len(x), device=device)
    n_val = max(1, int(round(0.1 * len(x))))
    val_idx, train_idx = order[:n_val], order[n_val:]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x[train_idx]), y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = F.mse_loss(model(x[val_idx]), y[val_idx]).item()
        history.append({"epoch": epoch + 1, "train_mse": float(loss.item()), "val_mse": float(val_loss)})
        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_mse": best_loss, "epochs_ran": len(history), "history": history}
