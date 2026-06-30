#!/usr/bin/env bash
set -euo pipefail
# 4-upscale.sh -- restore/upscale clip(s) with SeedVR2 (3B fp16, 1080p) on the RX 7800 XT.
# Usage:  ./4-upscale.sh clip.mp4 [more.mp4 ...]
# Override defaults with env vars, e.g.:  RES=1440 BATCH=9 ./4-upscale.sh clip.mp4

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

if [ "$#" -lt 1 ]; then
  echo "Usage: ./4-upscale.sh <video> [video2 ...]"
  exit 1
fi
if [ ! -d .venv ] || [ ! -d ComfyUI ]; then
  echo "[!] Run ./1-install-rocm.sh and ./3-bootstrap.sh first."
  exit 1
fi
# shellcheck disable=SC1091
. .venv/bin/activate
# shellcheck disable=SC1091
source ./env.sh

# SeedVR2 runs the WHOLE pipeline at the target resolution (it downscales the source to
# RES first, then restores upward). RES therefore drives VRAM, not the source size.
RES="${RES:-1080}"      # 1080 safe on 16GB; 1440 testable. OOM -> see TROUBLESHOOTING.md.
BATCH="${BATCH:-5}"     # MUST be 4n+1 (1,5,9,13,17,...). OOM -> drop to 1.
MODEL="${MODEL:-seedvr2_ema_3b_fp16.safetensors}"

# Resolve inputs to absolute paths NOW, while cwd is still the repo root.
# (After 'cd ComfyUI' a relative arg would resolve against ComfyUI/ and be missed.)
abs_files=()
for f in "$@"; do
  if [ ! -f "$f" ]; then
    echo "[!] Not a file, skipping: $f"
    continue
  fi
  abs_files+=("$(cd "$(dirname "$f")" && pwd)/$(basename "$f")")
done
if [ "${#abs_files[@]}" -eq 0 ]; then
  echo "[!] No valid input files; nothing to do." >&2
  exit 1
fi

cd ComfyUI
mkdir -p ../output
NODE="custom_nodes/seedvr2_videoupscaler"

fail=0
for abs in "${abs_files[@]}"; do
  base="$(basename "${abs%.*}")"
  echo "=== Upscaling $base  ($MODEL, ${RES}p, batch $BATCH) ==="
  if ! python3 "$NODE/inference_cli.py" "$abs" \
      --dit_model "$MODEL" \
      --resolution "$RES" \
      --batch_size "$BATCH" \
      --attention_mode sdpa \
      --output "../output/${base}_${RES}p.mp4"; then
    echo "[!] FAILED: $base (continuing to next file)"
    fail=$((fail + 1))
    continue
  fi
done

if [ "$fail" -gt 0 ]; then
  echo "Done with $fail failure(s) -> ./output/"
  exit 1
fi
echo "Done -> ./output/"
