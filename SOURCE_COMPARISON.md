# FineSteer MoSE source comparison

## Sources inspected

| Source | Revision/state | MoSE correspondence |
|---|---|---|
| `FineSteer.zip` | HEAD `5200e3095289f45fc8b6089a86964586d2c86659`, heavily modified working tree | Closest source. Adds prototype clustering, attention routing, residual PCA, and multiple clustering branches. |
| `D:\download\FineSteer(3)\FineSteer` | Same HEAD and practically the same experimental working tree as the ZIP | Same MoSE family as the ZIP; not an independent third algorithm. |
| `github.com/YukinoAsuna/FineSteer` | `09ac89e17e65320a424ecd761014092df088816a` | Older single-direction steering model. It lacks the paper's K-Means prototype bank and dense attentive expert mixture. |

## Paper requirements used for the comparison

Ignoring SCS, the paper's MoSE requires:

1. Build response-shift vectors `delta = h_preferred - h_undesired`.
2. Normalize the difference vectors for K-Means and select K automatically using the Calinski-Harabasz index.
3. Keep the resulting cluster centroids as a fixed prototype bank `C`.
4. Learn only `W_Q`, `W_K`, and the lightweight residual coefficient MLP `beta`; the prototype value vectors themselves are not projected or trained.
5. Use dense scaled dot-product attention over all experts, not Top-K token routing.
6. Build `U_res` from PCA of the original `D_delta` and predict a continuous residual `U_res beta(h_q)`.
7. Minimize MSE to the observed shift plus L2 regularization.

## Important deviations in the ZIP/local working tree

- The active `SteeringModel` applies a learned value projection to prototype vectors, while equation (11) uses the fixed centroids directly.
- The active K selection takes the median of elbow, silhouette, and Calinski-Harabasz suggestions; the paper specifies Calinski-Harabasz.
- The historical `base`, `delta_pca`, `joint`, and Scheme 1 branches construct the prototype/residual spaces differently; only one can match the paper. The retained Scheme 1 implementation is now published as `orthogonal_residual` (Orthogonal Residual MoSE).
- A commented implementation adds Top-K routing, which contradicts the paper's explicit dense representation-level routing.
- Several branches apply PCA or residualization in ways not described by Algorithm 2.
- The repository training code contains experiment-specific copies and inconsistent CLI defaults, so commit identity alone is not enough to reproduce the paper.

The `MoSE` preset in this directory implements the seven requirements above. The retained `orthogonal_residual` preset provides the meaningful alternative for controlled comparison.
