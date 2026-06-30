#!/usr/bin/env bash
set -euo pipefail
# 1-install-rocm.sh -- install AMD ROCm 7.2.1 + ROCm PyTorch for RX 7800 XT (gfx1101).
# Target: WSL2 Ubuntu 24.04 (primary) or 22.04. Native Ubuntu: see TROUBLESHOOTING.md.
# Verified 2026-06-30 vs AMD ROCm 7.2.1 (Radeon/Ryzen) docs. README explains the version split.

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

echo "== video-upscaler-7800xt-linux : ROCm 7.2.1 install =="

# --- WSL vs native ---
if grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then
  IS_WSL=1
  echo "[*] WSL2 detected."
  echo "    PREREQ (Windows host): AMD Software: Adrenalin Edition 26.2.2 or later must be installed,"
  echo "    otherwise the GPU is invisible inside WSL. Press Ctrl-C now if it is not installed."
else
  IS_WSL=0
  echo "[*] Native Linux detected (not WSL)."
  echo "    This script's defaults target WSL. For native Ubuntu the dkms/usermod/reboot"
  echo "    differences are in TROUBLESHOOTING.md -- read them before continuing."
fi

# --- Ubuntu codename ---
# shellcheck disable=SC1091
. /etc/os-release
UB=""
case "${FORCE_UBUNTU:-${VERSION_CODENAME:-}}" in
  noble) UB=noble ;;   # 24.04 LTS
  jammy) UB=jammy ;;   # 22.04 LTS
esac
if [ -z "$UB" ]; then
  cat <<EOF
[!] Ubuntu '${VERSION_CODENAME:-?}' (${VERSION_ID:-?}) is not supported by ROCm 7.2.1.
    AMD validates only Ubuntu 24.04 (noble) and 22.04 (jammy) for the RX 7800 XT.
    'wsl --install' defaults to the newest LTS, which can be too new.

    BEST FIX -- install a 24.04 WSL distro alongside this one (they coexist):
        (Windows PowerShell)  wsl --install -d Ubuntu-24.04
    then clone + run this repo inside Ubuntu-24.04.

    Advanced/unsupported -- try the 24.04 packages on THIS distro:
        FORCE_UBUNTU=noble ./1-install-rocm.sh
    (may hit glibc/dependency issues on a newer Ubuntu).
EOF
  exit 1
fi
echo "[*] Ubuntu ${VERSION_ID:-?} -> using '$UB' ROCm packages."

# --- Python 3.12 (ROCm wheels are cp312-only) ---
if ! python3 --version 2>&1 | grep -q "3\.12"; then
  echo "[!] Python 3.12 required (ROCm torch wheels are cp312-only). Found: $(python3 --version 2>&1)."
  echo "    Ubuntu 24.04 ships 3.12 by default. On 22.04 install python3.12 (deadsnakes) and re-run."
  exit 1
fi
if ! python3 -c 'import ensurepip' 2>/dev/null; then
  echo "[*] Installing python3-venv ..."
  sudo apt update && sudo apt install -y python3-venv python3-pip
fi

# --- DXG detection env (WSL) ---
export HSA_ENABLE_DXG_DETECTION=1
if [ "$IS_WSL" = 1 ] && ! grep -q HSA_ENABLE_DXG_DETECTION "$HOME/.bashrc" 2>/dev/null; then
  echo 'export HSA_ENABLE_DXG_DETECTION=1' >> "$HOME/.bashrc"
  echo "[*] Added HSA_ENABLE_DXG_DETECTION=1 to ~/.bashrc"
fi

# --- amdgpu-install (ROCm 7.2.1) ---
sudo apt update && sudo apt -y upgrade
DEB="amdgpu-install_7.2.1.70201-1_all.deb"
if ! dpkg -l amdgpu-install >/dev/null 2>&1; then
  wget -nc "https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/$UB/$DEB"
  sudo apt install -y "./$DEB"
  sudo apt update
fi

# --- ROCm runtime ---
if [ "$IS_WSL" = 1 ]; then
  # WSL: full ROCm userspace WITHOUT the kernel module (the Windows host driver owns the
  # GPU via /dev/dxg). 'rocm' is the valid usecase -- the old 'wsl' usecase was removed
  # (confirmed via 'amdgpu-install --list-usecase').
  sudo amdgpu-install -y --usecase=rocm --no-dkms
  sudo usermod -a -G render,video "$LOGNAME" || true

  # WSL REQUIRES librocdxg: the ROCm 7.1+ bridge that lets the ROCm runtime reach the
  # Windows GPU via /dev/dxg. It REPLACED the old 'wsl' usecase. Without it rocminfo
  # won't list the card and torch.cuda.is_available() is False. (HSA_ENABLE_DXG_DETECTION=1,
  # exported above, switches it on -- needed until ROCm 7.13.) Fetch the newest release .deb;
  # fall back to a pinned version if the GitHub release API is unreachable.
  echo "[*] Installing librocdxg (WSL GPU bridge) ..."
  DXG_URL="$(wget -qO- https://api.github.com/repos/ROCm/librocdxg/releases 2>/dev/null \
    | grep -oE 'https://[^"]*/rocdxg-roct_[^"]*_amd64\.deb' | head -1 || true)"
  if [ -z "$DXG_URL" ]; then
    DXG_URL="https://github.com/ROCm/librocdxg/releases/download/v1.2.1/rocdxg-roct_1.2.1_amd64.deb"
    echo "[*] (release API unavailable; using pinned $DXG_URL)"
  fi
  wget -nc "$DXG_URL"
  sudo dpkg -i "$(basename "$DXG_URL")" || sudo apt-get install -f -y
else
  sudo amdgpu-install -y --usecase=graphics,rocm
  sudo usermod -a -G render,video "$LOGNAME" || true
  echo "[!] Native: log out and back in (or reboot) so the render/video groups + dkms module load."
fi

# --- System-level GPU sanity ---
if command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -qi gfx1101; then
  echo "[*] rocminfo sees gfx1101."
else
  cat <<'EOF'
[!] rocminfo did NOT report gfx1101 yet. Most likely:
    - Group membership not active -> run 'wsl --shutdown' in Windows PowerShell, reopen
      this distro, and re-run this script (render/video group needs a fresh login).
    - Adrenalin 26.2.2+ not installed on the Windows host.
    - librocdxg failed to install above (scroll up for dpkg errors). See TROUBLESHOOTING.md.
    (You can still continue to the PyTorch step below; the smoke test is the real judge.)
EOF
fi

# --- Python venv + ROCm PyTorch ---
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -U pip wheel
echo "[*] Installing ROCm PyTorch (torch 2.9.1 + rocm7.2.1) ..."
pip install torch==2.9.1 torchvision==0.24.0 torchaudio==2.9.0 \
  -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/

echo
echo "== DONE. Next: run the smoke test (the GATE) =="
echo "   source .venv/bin/activate && source ./env.sh && python3 2-smoke-test.py"
echo "   ALL PASS  ->  ./3-bootstrap.sh"
