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
# TunableOp benchmarks GEMM kernel choices for your tensor shapes. It pays off for REPEATED
# identical workloads, but for a one-off upscale (many distinct tile shapes) the first-run
# tuning overhead may not amortize -- try PYTORCH_TUNABLEOP_ENABLED=0 if a run feels slow.
# Overridable: set the var before calling and this default won't clobber it.
export PYTORCH_TUNABLEOP_ENABLED="${PYTORCH_TUNABLEOP_ENABLED:-1}"
# MIOpen FAST conv-kernel find = quicker startup. Overridable.
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"

# --- Intentionally NOT enabled (documented so you know why) ---
# FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE   -> only if you BUILD the ROCm flash-attn Triton fork.
#                                             SeedVR2 uses PyTorch SDPA; leave off.
# TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1-> community flag, only ever reported working on
#                                             gfx1100, NEVER confirmed on gfx1101. Untested; leave off.

echo "[env.sh] DXG on. TunableOp=$PYTORCH_TUNABLEOP_ENABLED, MIOpen find=$MIOPEN_FIND_MODE. HSA_OVERRIDE_GFX_VERSION unset (correct for gfx1101)."
