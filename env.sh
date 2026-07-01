#!/usr/bin/env bash
# env.sh -- verified ROCm environment for RX 7800 XT (gfx1101) on WSL2 / Linux.
# Source before the smoke test and before running ComfyUI/SeedVR2:
#     source ./env.sh
# Verified 2026-06-30 against AMD ROCm 7.2.1 (Radeon/Ryzen) docs + PyTorch issue trackers.

# --- MANDATORY on WSL2 (librocdxg DXG GPU-detection path). Harmless on native Linux. ---
# Still needed on current ROCm (7.2.x) for WSL2 GPU detection -- required until AMD's
# ROCDXG/librocdxg WSL2 migration lands (tracking: ROCm#6296). No version cutoff is published.
export HSA_ENABLE_DXG_DETECTION=1

# --- DO NOT override the GFX version ---
# gfx1101 (RX 7800 XT) is a NATIVE, officially-supported ROCm target. The 11.0.0
# override is for UNSUPPORTED cards only and would mis-target kernels here. Keep it unset.
unset HSA_OVERRIDE_GFX_VERSION 2>/dev/null || true

# --- GPU memory allocator: FORCE the native caching allocator (CRITICAL on WSL2) ---
# The ROCm-on-WSL PyTorch wheel defaults to the ASYNC allocator (hipMallocAsync), which issues a
# real stream-ordered driver malloc/free for EVERY op. On WSL2 each call crosses the dxg boundary
# into the Windows video-memory manager and serializes -- so the VAE's per-layer alloc/free churn
# dominates wall time (py-spy shows HipMallocAsync::mallocAsync/freeAsync as the hot path, GPU idle).
# The native caching allocator pools and REUSES freed blocks -> ~zero driver calls after warmup.
# This is THE performance fix on this setup. (PYTORCH_ALLOC_CONF is the unified var since torch 2.8;
# the HIP/CUDA aliases are set too for safety. Do NOT add expandable_segments on WSL -- it needs VMM
# driver APIs that are not implemented over dxg. Verified via PyTorch 2.9 source + AMD HIP docs.)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-backend:native}"
export PYTORCH_HIP_ALLOC_CONF="${PYTORCH_HIP_ALLOC_CONF:-backend:native}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:native}"

# --- MIOpen / cuDNN ---
# Belt-and-suspenders with fix-cudnn.sh: ComfyUI disables cuDNN for ALL AMD (an RDNA4 workaround),
# which cripples VAE Conv3d ~10x on RDNA3 (ComfyUI#10460). Keep MIOpen on.
export COMFYUI_ENABLE_MIOPEN="${COMFYUI_ENABLE_MIOPEN:-1}"
# MIOpen FAST conv-kernel find = quicker startup (default find mode causes big initial slowness).
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"

# --- TunableOp: OFF for SeedVR2 (varying-shape workload) ---
# TunableOp re-tunes GEMMs per new tensor shape. SeedVR2's VAE runs varying temporal dims, so the
# tuning never amortizes and compounds the per-shape kernel churn (ROCm Conv3d recompiles per shape,
# ComfyUI#12672). Leave off; override to 1 only if you pin a single fixed resolution+batch.
export PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-0}"

# --- Opt-in (NOT set here) ---
# TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 -> AOTriton attention for the DiT (not the VAE, so it
#   won't fix VAE-encode time). Toggle via FAST=1 in 4-upscale.sh; validate output looks correct.

echo "[env.sh] DXG on. alloc=$PYTORCH_ALLOC_CONF, MIOpen find=$MIOPEN_FIND_MODE (enable=$COMFYUI_ENABLE_MIOPEN), TunableOp=$PYTORCH_TUNABLEOP_ENABLED. HSA_OVERRIDE_GFX_VERSION unset (correct for gfx1101)."
