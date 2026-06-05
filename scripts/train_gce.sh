#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

. .venv/bin/activate
mkdir -p saved logs

timestamp="$(date +%Y%m%d-%H%M%S)"
log_path="logs/train-${timestamp}.log"

PYTHONPATH=.:./slimevolleygym python src/evolve.py 2>&1 | tee "$log_path"
