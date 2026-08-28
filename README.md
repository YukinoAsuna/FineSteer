# FineSteer

[![ACL 2026 Oral Paper](https://img.shields.io/badge/ACL%202026%20Oral-Paper-b31b1b)](https://aclanthology.org/2026.acl-long.852.pdf)

Official code for **[FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models](https://aclanthology.org/2026.acl-long.852.pdf)**, accepted as an **ACL 2026 Oral** paper.


## Quick start

Requirements: Linux, Python 3.10+, and CUDA-capable PyTorch.

```bash
git clone https://github.com/YukinoAsuna/FineSteer.git
cd FineSteer
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -r requirements.txt
bash run.sh llama3.1
```

`requirements.txt` is the standard dependency entry point. `run.sh` reuses the prepared `.venv`, installs FineSteer itself, and runs the complete 408/409 TruthfulQA experiment. If `.venv` does not exist, the script can also create it automatically. The default method is `MoSE`.

### Three supported model inputs

Friendly alias:

```bash
bash run.sh llama3.1
bash run.sh qwen2.5
```

Hugging Face model ID:

```bash
bash run.sh meta-llama/Llama-3.1-8B-Instruct
bash run.sh Qwen/Qwen2.5-7B-Instruct
```

Local checkpoint directory:

```bash
bash run.sh /data/models/Meta-Llama-3.1-8B-Instruct
```

For aliases and Hub IDs, Transformers first checks its Hugging Face cache and downloads missing files automatically. To force cached/local operation, append `--local-files-only`.


## Python/CLI usage

Install once:

```bash
python -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -e .
```

Run with explicit options:

```bash
finesteer \
  --model qwen2.5 \
  --method MoSE \
  --layer 12 \
  --strength 2.5 \
  --limit 0
```

Important options:

| Option | Default | Description |
|---|---:|---|
| `--model` | required | Alias, Hub model ID, or local checkpoint directory |
| `--method` | `MoSE` | `MoSE` or `orthogonal_residual` |
| `--layer` | `12` | Hidden-state extraction and intervention layer |
| `--strength` | `2.5` | Steering strength lambda |
| `--residual-dim` | `10` | Continuous refinement basis dimension |
| `--epochs` | `100` | Maximum MoSE training epochs |
| `--limit` | `0` | Test examples to generate; `0` means all 409 |
| `--reuse-checkpoint` | off | Reuse a previously trained method checkpoint |
| `--local-files-only` | off | Disable Hub downloads and use cached/local files only |

The same defaults can be set through environment variables; copy `.env.example` as a reference.

## Implementations

`MoSE` is the default and follows the paper's MoSE construction: fixed prototype experts are built from clustered activation shifts, then PCA supplies a continuous refinement basis.

`orthogonal_residual` preserves the strongest alternative found in the supplied ZIP while giving it a descriptive, provenance-independent name. It:

1. clusters the raw activation shifts and selects the cluster count from the median elbow, silhouette, and Calinski-Harabasz estimates;
2. learns a value projection for the prototype experts;
3. removes the prototype span before modeling the remaining signal; and
4. applies PCA to that orthogonal residual and learns its coefficients.

The public name is **Orthogonal Residual MoSE (OR-MoSE)**; the stable CLI/checkpoint identifier is `orthogonal_residual`.


## OpenAI grading

Generation does not require an OpenAI key. To reproduce the truthfulness judge after generation:

```bash
export OPENAI_API_KEY="..."
finesteer-evaluate \
  runs/llama31/predictions/full_a2.5_MoSE.jsonl \
  --model gpt-5.6-luna \
  --output runs/llama31/gpt-5.6-luna-score.json
```

Alternatively pass `--prompt-key` to enter the key without echo. Keys are never written by the project.

## Multi-GPU generation

Run otherwise identical commands with `--num-shards 4` and shard indexes `0`, `1`, `2`, and `3`, then merge:

```bash
finesteer-merge runs/model/predictions/full_*_shard* \
  --output runs/model/predictions/full.jsonl \
  --expected 409
```

Sharding partitions the deterministic test indexes; it does not change training data, checkpoints, prompts, or generation settings.

## Output layout

```text
runs/<model-key>/
├── activations.pt
├── checkpoints/<method>.pt
├── metadata/<method>.json
└── predictions/<split>_a<strength>_<method>.jsonl
```

`runs/` and API credentials are excluded from Git by default.

## Citation

If you find FineSteer useful, please cite:

```bibtex
@inproceedings{weng2026finesteer,
  title={FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models},
  author={Weng, Zixuan and Zhang, Jinghuai and Cai, Kunlin and Li, Ying and Wang, Peiran and Tian, Yuan},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={18736--18756},
  year={2026}
}
```

## License

This repository is released under the [MIT License](LICENSE). Machine-readable citation metadata is available in `CITATION.cff`.
