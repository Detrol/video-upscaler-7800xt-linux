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

# --- Performance / stability (optional; safe) ---
# TunableOp benchmarks GEMM kernel choices for your tensor shapes (RDNA3 heuristics are
# immature). First run is slower while it tunes, then faster.
export PYTORCH_TUNABLEOP_ENABLED=1
# MIOpen FAST conv-kernel find = quicker ComfyUI startup. Unset for a final batch run if you
# want MIOpen to search for peak conv performance.
export MIOPEN_FIND_MODE=2

# --- Intentionally NOT enabled (documented so you know why) ---
# FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE   -> only if you BUILD the ROCm flash-attn Triton fork.
#                                             SeedVR2 uses PyTorch SDPA; leave off.
# TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1-> community flag, only ever reported working on
#                                             gfx1100, NEVER confirmed on gfx1101. Untested; leave off.

echo "[env.sh] HSA_ENABLE_DXG_DETECTION=1, TunableOp on, MIOpen FAST. HSA_OVERRIDE_GFX_VERSION unset (correct for gfx1101)."
