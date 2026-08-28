#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL="${FINESTEER_MODEL:-${FINESTEER_MODEL_PATH:-}}"
if [[ -z "$MODEL" && $# -gt 0 ]]; then
  MODEL="$1"
  shift
fi
if [[ -z "$MODEL" ]]; then
  echo "Usage: bash run.sh MODEL [MoSE|orthogonal_residual] [extra finesteer arguments]" >&2
  echo "MODEL may be llama3.1, qwen2.5, a Hugging Face ID, or a local directory." >&2
  echo "Alternatively set FINESTEER_MODEL." >&2
  exit 2
fi
METHOD="${FINESTEER_METHOD:-${1:-MoSE}}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$METHOD" != "MoSE" && "$METHOD" != "orthogonal_residual" ]]; then
  echo "Method must be MoSE or orthogonal_residual, got: $METHOD" >&2
  exit 2
fi

if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install -q -e .

exec .venv/bin/finesteer --model "$MODEL" --method "$METHOD" "$@"
