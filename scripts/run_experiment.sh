#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Error: provide an experiment configuration." >&2
  echo "Usage: bash $0 <path_to_yaml_config>" >&2
  echo "Example: bash $0 configs/swinv2_base_384_minimal_pseudo.yaml" >&2
  exit 1
fi

CONFIG_PATH="$1"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Error: configuration file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

mkdir -p weights outputs docs

echo "[1/2] Training with ${CONFIG_PATH}"
uv run python src/train.py --config "${CONFIG_PATH}"

echo "[2/2] Running inference with ${CONFIG_PATH}"
uv run python src/predict.py --config "${CONFIG_PATH}"

echo "Experiment completed."
