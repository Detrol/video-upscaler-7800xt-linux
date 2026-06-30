# Troubleshooting

The `2-smoke-test.py` line that failed tells you which section applies.

## `1-install-rocm.sh` says "Ubuntu '<name>' is not supported"

ROCm 7.2.1 only supports Ubuntu **24.04 (noble)** and **22.04 (jammy)**. `wsl --install` now defaults to the newest LTS (e.g. 26.04 "resolute"), which is too new for the gfx1101-verified ROCm version.

Best fix — add a 24.04 distro (WSL runs several side by side, the Adrenalin host driver is shared):

```powershell
wsl --install -d Ubuntu-24.04
```

then clone + run the repo inside Ubuntu-24.04. Quick unsupported attempt on your current distro: `FORCE_UBUNTU=noble ./1-install-rocm.sh` (may fail on glibc/dependency mismatches against a newer Ubuntu).

## Smoke test FAIL: `import torch -> ...` (exit 2)

The ROCm wheel won't import — usually a missing runtime lib on bare WSL2 (AMD ROCm issue **#6053**, `libroctx64.so.4`).

**Fallback: run inside AMD's ROCm container** (the confirmed workaround). On WSL2 with Docker Desktop (or `docker` in the distro):

```bash
docker run -it --rm \
  --device=/dev/dxg \
  -v /usr/lib/wsl:/usr/lib/wsl \
  -e HSA_ENABLE_DXG_DETECTION=1 \
  -v "$PWD:/work" -w /work \
  rocm/pytorch:latest \
  python3 2-smoke-test.py
```

If it passes in the container, run `3-bootstrap.sh`'s steps inside the same container instead of the host venv. (Native Linux uses `--device=/dev/kfd --device=/dev/dri` instead of `/dev/dxg` + the wsl mount.)

## Smoke test FAIL: `torch.cuda.is_available() == False` (exit 3)

GPU not visible to ROCm.

1. `source ./env.sh` first (sets `HSA_ENABLE_DXG_DETECTION=1`).
2. WSL: confirm **Adrenalin 26.2.2+** is installed on the Windows host.
3. `rocminfo | grep gfx1101` — if empty, the install didn't land. See next section.

## `rocminfo` does not list gfx1101

`1-install-rocm.sh` installs ROCm (`--usecase=rocm --no-dkms`) **and** `librocdxg` (the ROCm 7.1+ WSL→Windows-GPU bridge that replaced the old `wsl` usecase; AMD docs for this are incomplete, bug **ROCm#6296**). If `rocminfo` still shows no gfx1101 after that:

- **Most common:** group membership isn't active yet. In Windows PowerShell run `wsl --shutdown`, reopen the distro, and re-run `./1-install-rocm.sh`. (`render`/`video` group needs a fresh login.)
- Confirm **Adrenalin 26.2.2+** on the Windows host.
- Check the librocdxg `dpkg -i` didn't error (scroll up in the install output). Manual install: grab `rocdxg-roct_<ver>_amd64.deb` from <https://github.com/ROCm/librocdxg/releases> and `sudo dpkg -i` it.

## Smoke test FAIL: `torch is NOT a ROCm build`

A CPU/CUDA wheel got installed. Reinstall the ROCm wheel:

```bash
. .venv/bin/activate
pip uninstall -y torch torchvision torchaudio
pip install torch==2.9.1 torchvision==0.24.0 torchaudio==2.9.0 \
  -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/
```

## Smoke test FAIL: `scaled_dot_product_attention` or `Conv3d`

This is the Windows failure reproducing on Linux. Before giving up:
- Try the **ROCm Docker** path above (different runtime libs).
- Cross-check on **native Ubuntu** (see below) — its gfx1101 support is more clearly documented than WSL's.
- Note: SeedVR2 uses `--attention_mode sdpa`, which falls back to math attention on gfx1101 (flash/mem-efficient kernels are not auto-enabled here — PyTorch #159226). Correctness is fine; it's just not the fastest path.

## Upscale prints "Terminated" / "Killed" mid-run (system RAM OOM)

This is **system RAM**, not VRAM (a VRAM OOM raises a Python `torch.OutOfMemoryError` traceback instead). The SeedVR2 CLI loads the **whole video into RAM** when `--chunk_size 0` (its default), which OOM-kills on a 16 GB-RAM box. `4-upscale.sh` now defaults to `CHUNK=100` (streaming, memory-bounded) to prevent this.

- Still killed? Lower the chunk: `CHUNK=50 ./4-upscale.sh clip.mp4` (or 25).
- Quick end-to-end test on a few seconds first: `LOADCAP=150 ./4-upscale.sh clip.mp4`.
- Give WSL2 a swap cushion (and don't disable it). In `C:\Users\<you>\.wslconfig`:
  ```ini
  [wsl2]
  memory=14GB
  swap=32GB
  ```
  then `wsl --shutdown` in PowerShell and reopen. (Confirm the ceiling with `free -h` — `total` is WSL2's RAM cap, `Swap` should not be `0B`.)

## Upscale OOMs (out of VRAM)

`4-upscale.sh` honors env overrides. Lower in this order:

```bash
RES=1080 BATCH=1 ./4-upscale.sh clip.mp4
```

Still OOM? Add VAE tiling / model offload by editing the `inference_cli.py` call in `4-upscale.sh`:

```
--vae_encode_tiled --vae_encode_tile_size 512 \
--vae_decode_tiled --vae_decode_tile_size 512 \
--dit_offload_device cpu --blocks_to_swap 16 --vae_offload_device cpu
```

(`--blocks_to_swap` is DiT-only and does not help VAE-encode OOM; tiling does. Raise `blocks_to_swap` 16→24→32 if the DiT load itself OOMs.)

## ComfyUI / SeedVR2 hangs with no output

Known on some ROCm setups (SeedVR2 issue **#511**, seen on a gfx1150 iGPU). Cross-check on **native Ubuntu** or the **ROCm Docker** container. If it hangs everywhere on gfx1101, the node may not yet be viable on this GPU — report back the exact stall point.

## 7B fp8 model (higher quality)

```bash
. .venv/bin/activate
hf download numz/SeedVR2_comfyUI seedvr2_ema_7b_fp8_e4m3fn.safetensors \
  --local-dir ComfyUI/models/SEEDVR2
MODEL=seedvr2_ema_7b_fp8_e4m3fn.safetensors ./4-upscale.sh clip.mp4
```

16 GB headroom for 7B fp8 is not officially quantified — expect to need the VAE-tiling / BlockSwap flags above. (The SeedVR2 README also lists a `..._mixed_block35_...` name that does **not** exist in the HF repo — ignore it.)

## Native Ubuntu (fallback to WSL2)

Native gfx1101 support is more clearly documented than WSL. Differences from `1-install-rocm.sh`:

- Exact point release required: **Ubuntu 24.04.4** or **22.04.5** (`lsb_release -d`).
- Build the in-kernel driver: `--usecase=graphics,rocm` (no `--no-dkms`), plus `sudo apt install amdgpu-dkms linux-headers-$(uname -r) linux-modules-extra-$(uname -r)`.
- Grant device access: `sudo usermod -a -G render,video $LOGNAME`, then **reboot**.
- No `HSA_ENABLE_DXG_DETECTION` needed (that's a WSL-only DXG path); leave the rest of `env.sh` as-is.
- Verify: `rocminfo | grep gfx1101` and `clinfo`.

The rest (`2-smoke-test.py` → `3-bootstrap.sh` → `4-upscale.sh`) is identical.

## Version note

For the RX 7800 XT, follow ROCm **7.2.1** (the Radeon/Ryzen consumer docs track). The core/Instinct ROCm docs show 7.2.4 — that's a separate track and not the number to match for this GPU. `HSA_OVERRIDE_GFX_VERSION` is **not** needed (gfx1101 is natively supported); do not set it.
