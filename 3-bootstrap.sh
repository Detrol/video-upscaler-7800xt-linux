#!/usr/bin/env bash
set -euo pipefail
# 3-bootstrap.sh -- install ComfyUI + SeedVR2 node + models into this repo dir.
# Run AFTER ./1-install-rocm.sh and a PASSING ./2-smoke-test.py.

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

if [ ! -d .venv ]; then
  echo "[!] .venv missing -- run ./1-install-rocm.sh first."
  exit 1
fi
# shellcheck disable=SC1091
. .venv/bin/activate
# shellcheck disable=SC1091
source ./env.sh

# Guard: ROCm torch must be present before we install anything that depends on torch.
python3 - <<'PY'
import sys
import torch
if not torch.version.hip:
    print("[!] torch is not a ROCm build -- re-run ./1-install-rocm.sh")
    sys.exit(1)
print(f"[*] torch {torch.__version__} hip {torch.version.hip}")
PY

# --- ComfyUI ---
if [ ! -d ComfyUI ]; then
  git clone https://github.com/comfyanonymous/ComfyUI.git
fi
cd ComfyUI

# Install ComfyUI deps WITHOUT clobbering the ROCm torch: strip the torch* lines first.
grep -vE '^(torch|torchvision|torchaudio)([=<>!~ ]|$)' requirements.txt > requirements.norocm.txt || [ "$?" -eq 1 ]
pip install -r requirements.norocm.txt

# --- SeedVR2 node ---
NODE="custom_nodes/seedvr2_videoupscaler"
if [ ! -d "$NODE" ]; then
  git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git "$NODE"
fi
if [ -f "$NODE/requirements.txt" ]; then
  grep -vE '^(torch|torchvision|torchaudio)([=<>!~ ]|$)' "$NODE/requirements.txt" > "$NODE/requirements.norocm.txt" || [ "$?" -eq 1 ]
  pip install -r "$NODE/requirements.norocm.txt"
fi

# Verify the ROCm torch survived the node's dependency install.
python3 - <<'PY'
import sys
import torch
if not torch.version.hip:
    print("[!] ROCm torch got clobbered by a node dependency (a CUDA/CPU wheel was pulled).")
    print("    Re-run ./1-install-rocm.sh to reinstall the ROCm wheel, then re-run ./3-bootstrap.sh")
    sys.exit(1)
print(f"[*] torch still ROCm: {torch.__version__} hip {torch.version.hip}")
PY

# --- Models (HF numz/SeedVR2_comfyUI) ---
pip install -U "huggingface_hub[cli]"
mkdir -p models/SEEDVR2
echo "[*] Downloading 3B fp16 + VAE (~7 GB) ..."
hf download numz/SeedVR2_comfyUI \
  seedvr2_ema_3b_fp16.safetensors ema_vae_fp16.safetensors \
  --local-dir models/SEEDVR2

echo
echo "== Bootstrap done. Upscale a clip: =="
echo "   ./4-upscale.sh /path/to/clip.mp4"
echo "   (Optional higher-quality 7B fp8: hf download numz/SeedVR2_comfyUI \\"
echo "      seedvr2_ema_7b_fp8_e4m3fn.safetensors --local-dir ComfyUI/models/SEEDVR2 ; see TROUBLESHOOTING.md)"
