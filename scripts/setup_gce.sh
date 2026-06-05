#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  ffmpeg \
  git \
  python3-dev \
  python3-venv

python3 -m venv .venv
. .venv/bin/activate

python -m pip install --upgrade "pip==23.3.2" "setuptools==65.5.0" "wheel==0.37.1"
python -m pip install --no-build-isolation -r requirements-gce.txt
python -m pip install --no-deps -e ./slimevolleygym

mkdir -p saved logs

echo "GCE setup complete."
echo "Run training with: bash scripts/train_gce.sh"
