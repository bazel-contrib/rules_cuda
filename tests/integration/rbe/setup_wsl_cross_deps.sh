#!/usr/bin/env bash
# Install packages needed for REQUIRED-A Linux exec inside WSL.
# - aarch64 cross-gcc: linux-sbsa target objects
# - qemu-user: only required when exec tools are aarch (optional case 4)
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not found; skipping package install (non-Debian WSL?)" >&2
  uname -a
  exit 0
fi

sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  g++-aarch64-linux-gnu \
  gcc-aarch64-linux-gnu \
  binutils-aarch64-linux-gnu \
  qemu-user-static \
  binfmt-support \
  ca-certificates \
  curl \
  file \
  binutils \
  psmisc

# Guest aarch64 loader so binfmt can run sbsa tools when needed.
if [[ -e /usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1 ]]; then
  sudo ln -sfn /usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1 /lib/ld-linux-aarch64.so.1 || true
  if [[ -d /usr/aarch64-linux-gnu/lib ]]; then
    if [[ -d /lib/aarch64-linux-gnu && ! -L /lib/aarch64-linux-gnu ]]; then
      sudo rm -rf /lib/aarch64-linux-gnu
    fi
    sudo ln -sfn /usr/aarch64-linux-gnu/lib /lib/aarch64-linux-gnu || true
  fi
fi

echo "WSL host arch: $(uname -m)"
command -v aarch64-linux-gnu-g++
command -v qemu-aarch64-static || command -v qemu-aarch64 || true
update-binfmts --display qemu-aarch64 2>/dev/null || true
