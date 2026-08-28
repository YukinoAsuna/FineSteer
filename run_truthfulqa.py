from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from finesteer_moe import MoSE, MoSEConfig, build_mose_components, resolve_model, train_mose


SYSTEM_PROMPT = "You are a helpful, honest and concise assistant."
INSTRUCT = "Answer the question concisely. Q: {} A:"


def prompt_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCT.format(question)},
    ]


def clean_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": row["question"],
        "correct_answers": [x for x in row["correct_answers"] if x.strip()],
        "incorrect_answers": [x for x in row["incorrect_answers"] if x.strip()],
        "category": row["category"],
    }


def load_model(model_source: str, device: str, *, local_files_only: bool = False, cache_dir: str | None = None):
    kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_source, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            **kwargs,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to load {model_source!r}. Check the model ID/network/cache. "
            "For gated models such as Llama, accept the model license and run `hf auth login`. "
            "Use --local-files-only only when the checkpoint is already cached."
        ) from exc
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


@torch.inference_mode()
def extract_record(model, tokenizer, row: dict[str, Any], layer: int, device: str):
    messages = prompt_messages(row["question"])
    query_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    query_tokens = tokenizer(query_text, return_tensors="pt", add_special_tokens=False).to(device)
    query_len = query_tokens["input_ids"].shape[1]
    query_out = model(**query_tokens, output_hidden_states=True, use_cache=False)
    hq = query_out.hidden_states[layer][0, -1].float().cpu()

    correct = messages + [{"role": "assistant", "content": row["correct_answers"][0]}]
    incorrect = messages + [{"role": "assistant", "content": row["incorrect_answers"][0]}]
    correct_text = tokenizer.apply_chat_template(correct, tokenize=False, add_generation_prompt=False)
    incorrect_text = tokenizer.apply_chat_template(incorrect, tokenize=False, add_generation_prompt=False)
    correct_tokens = tokenizer(correct_text, return_tensors="pt", add_special_tokens=False).to(device)
    incorrect_tokens = tokenizer(incorrect_text, return_tensors="pt", add_special_tokens=False).to(device)
    correct_out = model(**correct_tokens, output_hidden_states=True, use_cache=False)
    incorrect_out = model(**incorrect_tokens, output_hidden_states=True, use_cache=False)
    hc = correct_out.hidden_states[layer][0, query_len:].float().mean(0).cpu()
    hi = incorrect_out.hidden_states[layer][0, query_len:].float().mean(0).cpu()
    return hq, hc - hi, query_text


@torch.inference_mode()
def extract_query(model, tokenizer, row: dict[str, Any], layer: int, device: str):
    messages = prompt_messages(row["question"])
    query_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    query_tokens = tokenizer(query_text, return_tensors="pt", add_special_tokens=False).to(device)
    query_out = model(**query_tokens, output_hidden_states=True, use_cache=False)
    return query_out.hidden_states[layer][0, -1].float().cpu(), query_text


def build_activation_cache(args, model, tokenizer, cache_path: Path) -> dict[str, Any]:
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    split = dataset.train_test_split(test_size=0.5, seed=0)
    train_records = [clean_record(row) for row in split["train"]]
    test_records = [clean_record(row) for row in split["test"]]
    train_hq, train_delta = [], []
    test_hq, test_prompts = [], []
    for index, row in enumerate(train_records):
        hq, delta, _ = extract_record(model, tokenizer, row, args.layer, args.device)
        train_hq.append(hq)
        train_delta.append(delta)
        if (index + 1) % 25 == 0:
            print(f"activation train {index + 1}/{len(train_records)}", flush=True)
    for index, row in enumerate(test_records):
        hq, prompt = extract_query(model, tokenizer, row, args.layer, args.device)
        test_hq.append(hq)
        test_prompts.append(prompt)
        if (index + 1) % 25 == 0:
            print(f"activation test {index + 1}/{len(test_records)}", flush=True)
    cache = {
        "train_hq": torch.stack(train_hq),
        "train_delta": torch.stack(train_delta),
        "test_hq": torch.stack(test_hq),
        "train_records": train_records,
        "test_records": test_records,
        "test_prompts": test_prompts,
        "layer": args.layer,
        "split_seed": 0,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


def train_variant(args, variant: str, cache: dict[str, Any], checkpoint_path: Path) -> tuple[MoSE, dict[str, Any]]:
    cfg = MoSEConfig.preset(variant, residual_dim=args.residual_dim)
    prototypes, basis, component_meta = build_mose_components(cache["train_delta"], cfg)
    model = MoSE(
        prototypes,
        basis,
        value_projection=cfg.value_projection,
        attention_dim=None if args.attention_dim == 0 else args.attention_dim,
    ).to(args.device)
    train_meta = train_mose(
        model,
        cache["train_hq"],
        cache["train_delta"],
        epochs=args.epochs,
        patience=args.patience,
        seed=0,
    )
    metadata = {"components": component_meta, "training": train_meta}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, checkpoint_path)
    return model, metadata


def load_variant(args, variant: str, checkpoint_path: Path) -> tuple[MoSE, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    cfg = MoSEConfig.preset(variant, residual_dim=args.residual_dim)
    model = MoSE(
        state["prototypes"],
        state["residual_basis"],
        value_projection=cfg.value_projection,
        attention_dim=None if args.attention_dim == 0 else args.attention_dim,
    )
    model.load_state_dict(state)
    return model.to(args.device).eval(), payload["metadata"]


def add_fixed_steering(output, vector: torch.Tensor):
    if isinstance(output, tuple):
        return (output[0] + vector.view(1, 1, -1).to(output[0]), *output[1:])
    if isinstance(output, list):
        return [output[0] + vector.view(1, 1, -1).to(output[0]), *output[1:]]
    return output + vector.view(1, 1, -1).to(output)


@torch.inference_mode()
def generate_variant(args, base_model, tokenizer, mose: MoSE | None, cache: dict[str, Any], variant: str, out_path: Path):
    indexes = list(range(len(cache["test_records"])))
    if args.limit:
        rng = random.Random(args.screen_seed)
        indexes = sorted(rng.sample(indexes, min(args.limit, len(indexes))))
    if args.num_shards > 1:
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard-index must be in [0, num-shards)")
        indexes = indexes[args.shard_index :: args.num_shards]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layer_module = base_model.model.layers[args.layer]
    with out_path.open("w", encoding="utf-8") as stream:
        for position, index in enumerate(indexes):
            hq = cache["test_hq"][index].unsqueeze(0).to(args.device)
            vector = None if mose is None else args.strength * mose(hq)[0]
            handle = None
            if vector is not None:
                handle = layer_module.register_forward_hook(lambda _m, _i, out, v=vector: add_fixed_steering(out, v))
            try:
                tokens = tokenizer(cache["test_prompts"][index], return_tensors="pt", add_special_tokens=False).to(args.device)
                generated = base_model.generate(
                    **tokens,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            finally:
                if handle is not None:
                    handle.remove()
            answer = tokenizer.decode(generated[0, tokens["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            row = dict(cache["test_records"][index])
            row.update({"index": index, "variant": variant, "answer": answer})
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{variant} generation {position + 1}/{len(indexes)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate FineSteer MoSE on TruthfulQA.")
    parser.add_argument("--model", default=os.environ.get("FINESTEER_MODEL") or os.environ.get("FINESTEER_MODEL_PATH"))
    parser.add_argument("--model-path", dest="legacy_model_path", help=argparse.SUPPRESS)
    parser.add_argument("--model-key", default=os.environ.get("FINESTEER_MODEL_KEY"))
    parser.add_argument(
        "--method",
        choices=("MoSE", "orthogonal_residual"),
        default=os.environ.get("FINESTEER_METHOD", "MoSE"),
    )
    parser.add_argument("--device", default=os.environ.get("FINESTEER_DEVICE", "cuda:0"))
    parser.add_argument("--layer", type=int, default=int(os.environ["FINESTEER_LAYER"]) if os.environ.get("FINESTEER_LAYER") else None)
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--residual-dim", type=int, default=10)
    parser.add_argument("--attention-dim", type=int, default=0, help="0 means the full hidden dimension, matching the zip.")
    parser.add_argument("--strength", type=float, default=float(os.environ["FINESTEER_STRENGTH"]) if os.environ.get("FINESTEER_STRENGTH") else None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=int(os.environ.get("FINESTEER_LIMIT", "0")), help="0 runs the complete 409-example test split.")
    parser.add_argument("--screen-seed", type=int, default=260415488)
    parser.add_argument("--reuse-checkpoint", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--cache-dir", default=os.environ.get("FINESTEER_CACHE_DIR"))
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=os.environ.get("FINESTEER_LOCAL_FILES_ONLY", "").lower() in {"1", "true", "yes"},
    )
    args = parser.parse_args()

    requested_model = args.model or args.legacy_model_path
    if not requested_model:
        parser.error("--model is required (or set FINESTEER_MODEL)")
    try:
        spec = resolve_model(requested_model)
    except ValueError as exc:
        parser.error(str(exc))
    args.model_source = spec.source
    args.model_key = args.model_key or spec.model_key
    args.layer = spec.layer if args.layer is None else args.layer
    args.strength = spec.strength if args.strength is None else args.strength
    print(f"model: {requested_model} -> {args.model_source}", flush=True)

    model_dir = args.run_dir / args.model_key
    cache_path = model_dir / "activations.pt"
    base_model, tokenizer = load_model(
        args.model_source,
        args.device,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
    )
    cache = torch.load(cache_path, map_location="cpu", weights_only=False) if cache_path.exists() else build_activation_cache(args, base_model, tokenizer, cache_path)

    method = args.method
    checkpoint = model_dir / "checkpoints" / f"{method}.pt"
    if checkpoint.exists() and args.reuse_checkpoint:
        mose, metadata = load_variant(args, method, checkpoint)
    else:
        mose, metadata = train_variant(args, method, cache, checkpoint)
    meta_path = model_dir / "metadata" / f"{method}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    tag = ("full" if args.limit == 0 else f"screen{args.limit}") + f"_a{args.strength:g}"
    shard = f"_shard{args.shard_index}of{args.num_shards}" if args.num_shards > 1 else ""
    output = model_dir / "predictions" / f"{tag}_{method}{shard}.jsonl"
    generate_variant(args, base_model, tokenizer, mose, cache, method, output)
    print(f"predictions: {output}", flush=True)


if __name__ == "__main__":
    main()
