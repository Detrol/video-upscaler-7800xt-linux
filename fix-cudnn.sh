#!/usr/bin/env bash
set -euo pipefail
# fix-cudnn.sh -- re-enable cuDNN/MIOpen for RDNA3 (gfx1101) in ComfyUI.
#
# ComfyUI (PR #10302, v0.3.65+) sets `torch.backends.cudnn.enabled = False` for ALL AMD GPUs
# to work around an RDNA4-only bug. On the RX 7800 XT (RDNA3) this cripples VAE Conv3d ~10x
# (2.09s -> 0.19s once re-enabled) and ~10x VRAM (ComfyUI#10460). This patch flips it back.
# Idempotent. Auto-run by 3-bootstrap.sh; run standalone after a ComfyUI update.

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

MM="ComfyUI/comfy/model_management.py"
if [ ! -f "$MM" ]; then
  echo "[!] $MM not found -- run ./3-bootstrap.sh first."
  exit 1
fi

OLD='torch.backends.cudnn.enabled = False  # Seems to improve things a lot on AMD'
NEW='torch.backends.cudnn.enabled = True  # PATCHED for RDNA3/gfx1101: MIOpen ON is ~10x faster (ComfyUI#10460)'

if grep -qF "PATCHED for RDNA3/gfx1101" "$MM"; then
  echo "[*] cuDNN already patched (RDNA3)."
elif grep -qF "$OLD" "$MM"; then
  sed -i "s|${OLD}|${NEW}|" "$MM"
  echo "[*] Patched: re-enabled cuDNN/MIOpen for RDNA3 in $MM"
else
  echo "[!] Expected cuDNN-disable line not found in $MM."
  echo "    ComfyUI may have changed it. Look for 'cudnn.enabled = False' and set it True."
  exit 1
fi
