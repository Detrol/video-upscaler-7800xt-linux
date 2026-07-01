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

# FAST=1 enables the AOTriton flash/mem-efficient attention kernels. On Linux these ARE built
# for gfx1101 (unlike Windows, where the same flag crashes) and are FAR faster than the default
# math attention. Experimental -> VALIDATE FIRST: `FAST=1 python3 2-smoke-test.py` must say
# ALL PASS, then eyeball a short clip's output. Default off = slow but known-correct.
if [ "${FAST:-0}" = 1 ]; then
  export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
  echo "[*] FAST attention ON (AOTriton experimental) -- verify output looks correct."
fi

# SeedVR2 runs the WHOLE pipeline at the target resolution (it downscales the source to
# RES first, then restores upward). RES therefore drives VRAM, not the source size.
RES="${RES:-1080}"      # 1080 safe on 16GB VRAM; 1440 testable. VRAM OOM -> see TROUBLESHOOTING.md.
BATCH="${BATCH:-5}"     # MUST be 4n+1 (1,5,9,13,17,...). VRAM OOM -> drop to 1.
MODEL="${MODEL:-seedvr2_ema_3b_fp16.safetensors}"
# Streaming chunk = frames decoded into SYSTEM RAM at once. The CLI default (0) loads the
# WHOLE clip as one float32 tensor and OOM-kills on a 16 GB-RAM box. Frames are float32 at
# SOURCE resolution, so a 4K source is ~100 MB/frame -> keep this small. 25 is safe (~2.5 GB);
# raise to 50/100 only with RAM headroom. README rule of thumb: chunk ~= 3-4x batch_size.
CHUNK="${CHUNK:-25}"
# Blend frames between streamed chunks to avoid visible seams (README: 2-4).
OVERLAP="${OVERLAP:-2}"
# Quick test: cap TOTAL frames processed (0 = whole clip). e.g. LOADCAP=150 -> ~first 5 s.
# (RAM stays bounded by CHUNK regardless; LOADCAP just shortens the job.)
LOADCAP="${LOADCAP:-0}"
# VAE tiling: splits VAE encode/decode into smaller tiles so each GPU kernel is short enough
# to avoid the Windows/WSL2 driver timeout (TDR) that resets the GPU on long (>2 s) kernels.
# 512 is safe; drop to 256 if a "driver timeout" still occurs. 0 disables tiling.
VAETILE="${VAETILE:-512}"
TILEARGS=()
if [ "$VAETILE" -gt 0 ]; then
  TILEARGS=(--vae_encode_tiled --vae_encode_tile_size "$VAETILE" --vae_decode_tiled --vae_decode_tile_size "$VAETILE")
fi

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
  echo "=== Upscaling $base  ($MODEL, ${RES}p, batch $BATCH, chunk $CHUNK, tile $VAETILE) ==="
  if ! python3 "$NODE/inference_cli.py" "$abs" \
      --dit_model "$MODEL" \
      --resolution "$RES" \
      --batch_size "$BATCH" \
      --chunk_size "$CHUNK" \
      --temporal_overlap "$OVERLAP" \
      --load_cap "$LOADCAP" \
      "${TILEARGS[@]}" \
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
