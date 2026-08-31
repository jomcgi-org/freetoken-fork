#!/usr/bin/env bash
# Idempotent post-boot recovery for the spot bench VM.
# Persistent state on /mnt/data (pd, survives preemption); bench serving reads
# the FTW copy on /mnt/nvme (local SSD, wiped by every preemption).
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$HOME/.local/bin:$PATH"

DATA_DEV=/dev/disk/by-id/google-ftdata
SSD_DEV=/dev/disk/by-id/google-local-nvme-ssd-0

# persistent data disk: format only if blank
if ! sudo blkid "$DATA_DEV" >/dev/null 2>&1; then sudo mkfs.ext4 -qF "$DATA_DEV"; fi
sudo mkdir -p /mnt/data && mountpoint -q /mnt/data || sudo mount "$DATA_DEV" /mnt/data
sudo chown "$USER" /mnt/data

# local SSD: always reformat after a preemption
sudo mkdir -p /mnt/nvme
if ! mountpoint -q /mnt/nvme; then sudo mkfs.ext4 -qF "$SSD_DEV" && sudo mount "$SSD_DEV" /mnt/nvme; fi
sudo chown "$USER" /mnt/nvme

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }

cd /mnt/data
[ -d FreeToken ] || git clone --branch feat/moe-disk-tier https://github.com/jomcgi/FreeToken.git
cd FreeToken && git pull -q
[ -x .venv/bin/ft ] || { uv venv -q --python 3.12 && uv pip install -q -e ".[accel]" pytest; }

mkdir -p /mnt/data/models
M=/mnt/data/models/qwen3.6-35b-a3b-nvfp4
[ -f "$M/config.json" ] || { uv tool install -q "huggingface_hub[cli]" 2>/dev/null || true; hf download nvidia/Qwen3.6-35B-A3B-NVFP4 --local-dir "$M"; }
[ -f "$M.ftw/freetoken_weight.json" ] || .venv/bin/ft checkpoint --model "$M" --out "$M.ftw"

# bench copy on local SSD
rsync -a "$M.ftw/" /mnt/nvme/model.ftw/

# Flash-Next FTW: restore the stored copy from pd (fast) or convert once and store it
F=/mnt/data/models/flash-next-nvfp4
if [ -d /mnt/data/flash.ftw ]; then
  rsync -a /mnt/data/flash.ftw/ /mnt/nvme/flash.ftw/
else
  .venv/bin/ft checkpoint --model "$F" --out /mnt/nvme/flash.ftw
  rsync -a /mnt/nvme/flash.ftw/ /mnt/data/flash.ftw/
fi
ln -sf "$F"/*.safetensors /mnt/nvme/flash.ftw/
echo "recover done"
