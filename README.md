# video-upscaler-7800xt-linux

AI video **restoration / upscaling** with [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) on an **AMD Radeon RX 7800 XT** (gfx1101, RDNA3, 16 GB), running on **Linux ROCm** (WSL2 primary, native Ubuntu fallback). Free / open-source, live-action friendly.

> **Why Linux and not Windows?** On Windows-ROCm the diffusion-transformer's attention/3D-conv kernels (AOTriton) are **not built for gfx1101**, which produced hard GPU hangs. On Linux those kernels exist (AOTriton 0.11b, PR #117), so this is the max-quality path. The repo is scripts only — model weights and ComfyUI are downloaded by the scripts, never committed.

## Staged flow (the smoke test is a gate)

```
git clone <this repo> && cd video-upscaler-7800xt-linux

./1-install-rocm.sh                                   # ROCm 7.2.1 + venv + ROCm PyTorch
source .venv/bin/activate && source ./env.sh
python3 2-smoke-test.py                               # <-- GATE: must say ALL PASS

# only if the smoke test passes:
./3-bootstrap.sh                                      # ComfyUI + SeedVR2 + models (~7 GB)
./4-upscale.sh /path/to/clip.mp4                      # result in ./output/
```

The smoke test runs the two ops that crashed on Windows (`scaled_dot_product_attention` + `Conv3d`) plus a torch-import check. If it fails you find out in ~15 minutes, **before** downloading 7 GB of models and building the pipeline. See `TROUBLESHOOTING.md` for what each failure means.

## Requirements

- **Windows host (for WSL2):** AMD Software: Adrenalin Edition **26.2.2 or later** — without it the GPU is invisible inside WSL.
- **WSL2 guest:** Ubuntu **24.04** (ships Python 3.12) recommended, or 22.04 with Python 3.12 installed.
- **Clone location:** clone into your Linux home (e.g. `~/video-upscaler-7800xt-linux`), **not** under `/mnt/c/...` — the Windows-mounted filesystem is far slower over WSL2 and can break the venv/pip steps and the ~7 GB model download.
- **Disk:** ~30 GB free (ROCm + venv + ComfyUI + models).

## Versions (pinned, verified 2026-06-30)

| Piece | Version | Note |
|---|---|---|
| ROCm | **7.2.1** | The *Radeon/Ryzen* consumer track that governs the RX 7800 XT. The core/Instinct track shows 7.2.4 — **ignore that number for this GPU.** |
| Adrenalin (host) | 26.2.2+ | Windows-side GPU driver for WSL. |
| PyTorch | torch 2.9.1 + rocm7.2.1 | From `repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/`. |
| Python | 3.12 | ROCm wheels are cp312-only. |
| SeedVR2 model | `seedvr2_ema_3b_fp16` + `ema_vae_fp16` | Fits 16 GB. 7B fp8 is an optional quality bump. |

## Honest status

- **No public report confirms SeedVR2 running end-to-end on gfx1101 on Linux yet** (proven on gfx1100 / 7900 XTX). The smoke test exists precisely to settle this on your box before you invest time.
- AMD's *latest* WSL install docs are mid-migration to "ROCDXG" and acknowledged incomplete (AMD docs bug ROCm#6296). `1-install-rocm.sh` uses the legacy single-command path that still works, and points to the ROCDXG fallback if `rocminfo` can't see the card.
- A bare WSL2 torch install can fail at *import* (ROCm#6053, missing `libroctx64.so.4`). If the smoke test fails at step 1, use the **ROCm Docker** fallback in `TROUBLESHOOTING.md`.

## Files

| File | Role |
|---|---|
| `1-install-rocm.sh` | ROCm 7.2.1 + Python venv + ROCm PyTorch (WSL2). |
| `2-smoke-test.py` | **Gate.** torch import + gfx1101 + SDPA attention + Conv3d on GPU. |
| `3-bootstrap.sh` | ComfyUI + SeedVR2 node + model download (torch-clobber guarded); auto-runs `fix-cudnn.sh`. |
| `fix-cudnn.sh` | Re-enables cuDNN/MIOpen for RDNA3 (ComfyUI disables it for all AMD → ~10x slower VAE). |
| `4-upscale.sh` | Run wrapper. `RES`/`BATCH`/`MODEL`/`CHUNK`/`VAETILE`/`FAST` overridable via env. |
| `env.sh` | Verified ROCm env vars; sourced by the scripts. |
| `TROUBLESHOOTING.md` | Smoke-test failures, Docker fallback, OOM knobs, native Ubuntu, known bugs. |
