#!/usr/bin/env bash
# G4 (-6 shape) setup: OPEN driver (Blackwell requires it; also enables HMM),
# data hyperdisk, FreeToken fork, models. Rerun after the driver reboot.
set -euo pipefail

if ! command -v nvidia-smi >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y nvidia-driver-590-server-open nvidia-utils-590-server python3.12-dev build-essential
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt-get update -qq && sudo apt-get install -yqq cuda-toolkit-13-0
  echo ">>> reboot then re-run"; sudo reboot
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

DATA=/dev/disk/by-id/google-ftdata
if ! sudo blkid "$DATA" >/dev/null 2>&1; then sudo mkfs.ext4 -qF "$DATA"; fi
sudo mkdir -p /mnt/data && mountpoint -q /mnt/data || sudo mount "$DATA" /mnt/data
sudo chown "$USER" /mnt/data

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-13.0

cd /mnt/data
[ -d FreeToken ] || git clone --branch feat/moe-disk-tier https://github.com/jomcgi/FreeToken.git
cd FreeToken && git pull -q
[ -x .venv/bin/ft ] || { uv venv -q --python 3.12 && uv pip install -q -e ".[accel]" pytest ninja; }

mkdir -p /mnt/data/models
M=/mnt/data/models/flash-next-nvfp4
[ -f "$M/config.json" ] || { uv tool install -q "huggingface_hub[cli]" 2>/dev/null || true; hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir "$M"; }
[ -d /mnt/data/models/ple-quant/ples_nvfp4 ] || hf download primitive-ai/Qwen3.8-Flash-Next-PLE-quant --include "ples_nvfp4/*" --local-dir /mnt/data/models/ple-quant
echo "g4 setup done"
