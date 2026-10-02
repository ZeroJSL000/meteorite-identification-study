#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/swinv2_base_384_minimal_pseudo.yaml}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Error: config file not found: ${CONFIG_PATH}" >&2
  exit 1
fi

uv run python src/predict.py --config "${CONFIG_PATH}"
