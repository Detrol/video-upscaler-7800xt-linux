#!/usr/bin/env python3
"""2-smoke-test.py -- GPU smoke test for RX 7800 XT (gfx1101) on ROCm.

Run this BEFORE bootstrapping ComfyUI/SeedVR2. It validates the exact ops that
crashed on Windows-ROCm (SDPA attention + 3D conv) plus that the ROCm torch wheel
even imports -- AMD ROCm issue #6053 makes a bare WSL2 install fail at import time.

Usage:
    source .venv/bin/activate && source ./env.sh && python3 2-smoke-test.py

Exit 0  -> all checks PASS, safe to run ./3-bootstrap.sh
Exit !=0 -> a check FAILED; see the FAIL line and TROUBLESHOOTING.md
"""
import sys


def ok(msg):
    print(f"  PASS  {msg}")


def bad(msg):
    print(f"  FAIL  {msg}")


fails = 0

# 1. import torch -- ROCm #6053: bare WSL2 wheels can fail here (missing libroctx64.so.4)
try:
    import torch
    ok(f"import torch {torch.__version__}")
except Exception as e:
    bad(f"import torch -> {e}")
    print("\nHINT: the ROCm torch wheel failed to import (see AMD ROCm issue #6053).")
    print("      Use the ROCm Docker container fallback in TROUBLESHOOTING.md.")
    sys.exit(2)

# 2. Is this actually a ROCm/HIP build (not a stray CPU/CUDA wheel)?
if torch.version.hip:
    ok(f"ROCm/HIP build: hip {torch.version.hip}")
else:
    bad("torch is NOT a ROCm build (torch.version.hip is None) -- wrong wheel installed")
    fails += 1

# 3. Is the GPU visible?
if torch.cuda.is_available():
    ok("torch.cuda.is_available() == True")
else:
    bad("torch.cuda.is_available() == False -- GPU not visible")
    if not torch.version.hip:
        print("\nHINT: this torch is not a ROCm build (see the FAIL above), so the GPU")
        print("      can never be visible. Reinstall the ROCm wheel into the venv:")
        print("      pip install torch==2.9.1 torchvision==0.24.0 torchaudio==2.9.0 \\")
        print("        -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/")
        print("      See TROUBLESHOOTING.md.")
    else:
        print("\nHINT: did you 'source ./env.sh' (HSA_ENABLE_DXG_DETECTION=1)?")
        print("      On WSL confirm Adrenalin 26.2.2+ on the Windows host and that")
        print("      'rocminfo | grep gfx1101' reports the card. See TROUBLESHOOTING.md.")
    sys.exit(3)

# 4. Is it the right GPU?
name = torch.cuda.get_device_name(0)
arch = torch.cuda.get_device_properties(0).gcnArchName
print(f"  INFO  device: {name} | gcnArch: {arch}")
if "gfx1101" in arch:
    ok("gfx1101 (RX 7800 XT) confirmed")
else:
    bad(f"expected gfx1101, got '{arch}' -- wrong GPU targeted")
    fails += 1

dev = torch.device("cuda")

# 5. matmul (rocBLAS / hipBLASLt)
try:
    a = torch.randn(1024, 1024, device=dev)
    b = torch.randn(1024, 1024, device=dev)
    c = a @ b
    torch.cuda.synchronize()
    assert c.shape == (1024, 1024)
    assert torch.isfinite(c).all(), "matmul produced NaN/Inf (broken rocBLAS/hipBLASLt kernel)"
    ok("matmul 1024x1024 on GPU")
except Exception as e:
    bad(f"matmul -> {e}")
    fails += 1

# 6. scaled_dot_product_attention -- THE op that segfaulted on Windows-ROCm (AOTriton kernels)
try:
    import torch.nn.functional as F
    q = torch.randn(1, 8, 512, 64, device=dev, dtype=torch.float16)
    k = torch.randn(1, 8, 512, 64, device=dev, dtype=torch.float16)
    v = torch.randn(1, 8, 512, 64, device=dev, dtype=torch.float16)
    out = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    assert out.shape == q.shape
    assert torch.isfinite(out).all(), "SDPA produced NaN/Inf (broken AOTriton kernel)"
    ok("scaled_dot_product_attention (fp16) on GPU  <-- the Windows crash op")
except Exception as e:
    bad(f"scaled_dot_product_attention -> {e}")
    fails += 1

# 7. Conv3d -- VAE / temporal-conv-like op
try:
    import torch.nn as nn
    # fp16 to match the production VAE (ema_vae_fp16) -- MIOpen picks different conv kernels per precision.
    conv = nn.Conv3d(4, 8, kernel_size=3, padding=1).to(dev).half()
    x = torch.randn(1, 4, 5, 64, 64, device=dev, dtype=torch.float16)
    y = conv(x)
    torch.cuda.synchronize()
    assert y.shape[1] == 8
    assert torch.isfinite(y).all(), "Conv3d produced NaN/Inf (broken MIOpen kernel)"
    ok("Conv3d forward (fp16) on GPU (VAE-like)")
except Exception as e:
    bad(f"Conv3d -> {e}")
    fails += 1

print()
if fails == 0:
    print("ALL PASS -- gfx1101 runs SDPA + 3D conv on ROCm. Safe to run ./3-bootstrap.sh")
    sys.exit(0)
print(f"{fails} CHECK(S) FAILED -- do NOT run bootstrap yet; see TROUBLESHOOTING.md")
sys.exit(1)
