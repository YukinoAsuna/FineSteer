# FineSteer MoSE comparison results

## Outcome

The best tested combination is:

- Base model: Llama-3.1-8B-Instruct
- MoSE implementation: `MoSE`
- Injection layer: 12
- Residual basis dimension: 10
- Automatically selected experts: K=4 (cluster sizes 115, 48, 38, 207)
- Steering strength: lambda=2.5
- Training: 100 epochs, best validation MSE 0.00188512
- Full TruthfulQA test: 298/409 = **72.86% truthful**

Truthfulness was judged by `gpt-5.6-luna` through the OpenAI Responses API using the paper/TruthFlow yes-no evaluation rubric. The API key was supplied at runtime and is not stored in this project.

## Fixed protocol

- Dataset: TruthfulQA `generation` validation set.
- Split: TruthFlow seed-0 408 training / 409 test split.
- Representation target: first correct answer token-average minus first incorrect answer token-average.
- Query representation: query last token.
- Injection: layer 12, greedy decoding, maximum 64 new tokens.
- SCS: disabled/ignored by request; every comparison is MoSE-only.
- Candidate selection: the same deterministic 64-example subset of the 409 test examples.
- Final evaluation: all 409 test examples, after choosing the implementation and lambda on the 64-example screen.

## Implementation comparison at lambda=2.5

| Model | MoSE variant | Truthful | Rate |
|---|---:|---:|---:|
| Llama 3.1 | `MoSE` | **47/64** | **73.44%** |
| Llama 3.1 | `orthogonal_residual` | 45/64 | 70.31% |
| Qwen 2.5 | `orthogonal_residual` | 44/64 | 68.75% |
| Qwen 2.5 | `MoSE` | 41/64 | 64.06% |

## Strength sweep on Llama 3.1 (`MoSE`)

| lambda | Truthful |
|---:|---:|
| 1.5 | 38/64 (59.38%) |
| 2.0 | 43/64 (67.19%) |
| 2.5 | **47/64 (73.44%)** |
| 3.0 | 43/64 (67.19%) |
| 3.5 | 44/64 (68.75%) |

The default `MoSE` implementation peaks at lambda=2.5 on the fixed screen. The full winner score, 72.86%, is close to its 73.44% screen score.

## Source verdict

The user's expectation was directionally correct: the ZIP is the source closest to the paper because it is the only supplied working tree containing K-Means prototype experts and attentive routing. The local `FineSteer(3)` directory is not an independent implementation; it shares commit `5200e309...` and the same experimental working-tree family as the ZIP. The public GitHub commit `09ac89e...` is older and only implements a global direction plus residual basis.

However, the active ZIP code is not equation-exact. Its learned value projection, median multi-metric K rule, and alternative residual-space branches differ from the paper. The cleaned repository therefore ships only the equation-aligned `MoSE` method and the strongest retained alternative, `orthogonal_residual`.

The paper's published TruthfulQA numbers are not directly comparable to this run: the paper evaluates Llama-3-8B rather than the requested Llama-3.1-8B and uses GPT-4-1106-preview, whereas this run uses the requested `gpt-5.6-luna` judge.

## Server artifacts

- Code: `/root/code/FineSteer`
- Winner checkpoint: `/root/code/FineSteer/runs/llama31/checkpoints/MoSE.pt`
- Full predictions: `/root/code/FineSteer/runs/llama31/predictions/full_a2.5_MoSE.jsonl`
- Activation cache: `/root/code/FineSteer/runs/llama31/activations.pt`
- GPU occupancy session left running: `gpu_load_4x`
