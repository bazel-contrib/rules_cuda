#!/usr/bin/env bash
# Install packages needed for Linux remote execution inside WSL.
# - aarch64 cross-gcc builds linux-sbsa target objects
# - qemu-user runs linux-sbsa execution tools on the x86_64 worker
#
# Idempotent: skips apt when tools are already present (CI may install them
# via Vampire/setup-wsl additional-packages under a working network).
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

need_install=false
for cmd in aarch64-linux-gnu-g++ curl gcc; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    need_install=true
    break
  fi
done
# psmisc provides fuser; optional but useful for cleanup.
if ! command -v fuser >/dev/null 2>&1; then
  need_install=true
fi

if [[ "$need_install" == true ]]; then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get not found and required tools missing" >&2
    uname -a
    exit 1
  fi

  # If DNS is broken (e.g. after a bad WSL networkingMode), pin public resolvers.
  if ! getent hosts archive.ubuntu.com >/dev/null 2>&1 && \
     ! getent hosts github.com >/dev/null 2>&1; then
    echo "WSL DNS looks broken; writing temporary resolv.conf" >&2
    if [[ -f /etc/wsl.conf ]] && grep -q 'generateResolvConf\s*=\s*false' /etc/wsl.conf 2>/dev/null; then
      :
    else
      # Best-effort: override resolv.conf for this session's package install.
      sudo cp -a /etc/resolv.conf /etc/resolv.conf.bak.rules_cuda 2>/dev/null || true
    fi
    printf 'nameserver 8.8.8.8\nnameserver 1.1.1.1\n' | sudo tee /etc/resolv.conf >/dev/null
  fi

  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends \
    g++-aarch64-linux-gnu \
    gcc-aarch64-linux-gnu \
    g++ \
    gcc \
    binutils-aarch64-linux-gnu \
    qemu-user-static \
    binfmt-support \
    ca-certificates \
    curl \
    file \
    binutils \
    psmisc
else
  echo "Cross-compile tools already present; skipping apt install"
fi

# Guest aarch64 loader so binfmt can run sbsa tools when needed.
if [[ -e /usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1 ]]; then
  sudo ln -sfn /usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1 /lib/ld-linux-aarch64.so.1 || true
  if [[ -d /usr/aarch64-linux-gnu/lib ]]; then
    if [[ -e /lib/aarch64-linux-gnu && ! -L /lib/aarch64-linux-gnu ]]; then
      sudo cp -asn /usr/aarch64-linux-gnu/lib/. /lib/aarch64-linux-gnu/ || true
    else
      sudo ln -sfn /usr/aarch64-linux-gnu/lib /lib/aarch64-linux-gnu || true
    fi
  fi
fi

echo "WSL host arch: $(uname -m)"
command -v aarch64-linux-gnu-g++
command -v curl
command -v gcc
command -v qemu-aarch64-static || command -v qemu-aarch64 || true
update-binfmts --display qemu-aarch64 2>/dev/null || true
