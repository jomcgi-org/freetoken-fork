#!/usr/bin/env bash
# Run ON the bench VM after first boot. Driver + NVMe mount + FreeToken + models.
set -euo pipefail

# --- NVIDIA driver (CUDA 13-capable; 590.48 is the known-good line upstream) ---
if ! command -v nvidia-smi >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y nvidia-driver-590-server-open nvidia-utils-590-server
  echo ">>> reboot, then re-run this script"; sudo reboot
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# --- local NVMe at /mnt/nvme ---
if ! mountpoint -q /mnt/nvme; then
  sudo mkfs.ext4 -F /dev/nvme0n1
  sudo mkdir -p /mnt/nvme && sudo mount /dev/nvme0n1 /mnt/nvme
  sudo chown "$USER" /mnt/nvme
fi

# --- FreeToken (disk-tier branch) ---
curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"
cd /mnt/nvme
[ -d FreeToken ] || git clone --branch feat/moe-disk-tier https://github.com/jomcgi/FreeToken.git
cd FreeToken
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[accel]"
uv tool install "huggingface_hub[cli]"

# --- models (small tier first; Flash-Next is ~115GB, start it in tmux) ---
mkdir -p /mnt/nvme/models
hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --local-dir /mnt/nvme/models/qwen3.6-35b-a3b-nvfp4
# hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --local-dir /mnt/nvme/models/qwen3.8-flash-next-nvfp4

# --- FTW conversion (required by the DISK tier) ---
ft checkpoint --model /mnt/nvme/models/qwen3.6-35b-a3b-nvfp4 --out /mnt/nvme/models/qwen3.6-35b-a3b-nvfp4.ftw
echo "setup done"
