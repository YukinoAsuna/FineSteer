from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    source: str
    model_key: str
    layer: int = 12
    strength: float = 2.5


MODEL_ALIASES: dict[str, ModelSpec] = {
    "llama3.1": ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "llama31"),
    "llama31": ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "llama31"),
    "qwen2.5": ModelSpec("Qwen/Qwen2.5-7B-Instruct", "qwen25"),
    "qwen25": ModelSpec("Qwen/Qwen2.5-7B-Instruct", "qwen25"),
}


def resolve_model(model: str) -> ModelSpec:
    """Resolve a friendly alias, Hub model ID, or local checkpoint directory."""

    value = model.strip()
    if not value:
        raise ValueError("Model cannot be empty")

    alias = MODEL_ALIASES.get(value.lower())
    if alias is not None:
        return alias

    path = Path(value).expanduser()
    if path.exists():
        if not path.is_dir() or not (path / "config.json").is_file():
            raise ValueError(f"Local model directory must contain config.json: {path}")
        resolved = str(path.resolve())
        return ModelSpec(resolved, _model_key(path.name))

    # Hugging Face repository IDs have the canonical namespace/repository form.
    if value.count("/") == 1 and all(part.strip() for part in value.split("/")):
        return ModelSpec(value, _model_key(value.rsplit("/", 1)[-1]))

    aliases = ", ".join(sorted({"llama3.1", "qwen2.5"}))
    raise ValueError(
        f"Unknown model {model!r}. Use an alias ({aliases}), a Hugging Face ID "
        "such as meta-llama/Llama-3.1-8B-Instruct, or a local checkpoint directory."
    )


def _model_key(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in cleaned.split("-") if part) or "model"
