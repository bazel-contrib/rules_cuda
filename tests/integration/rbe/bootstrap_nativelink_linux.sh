#!/usr/bin/env bash
# Download NativeLink (linux musl) and start a single-node RE endpoint on :1985.
# Runs inside the Linux nest: WSL (preferred) or a qemu-system guest.
set -euo pipefail

PORT="${RBE_PORT:-1985}"
CACHE="${NATIVELINK_CACHE:-$HOME/.cache/rules_cuda-rbe}"
VER="${NATIVELINK_VERSION:-1.6.4}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$CACHE/bin" /tmp/nativelink/work \
  /tmp/nativelink/data-worker-test/content_path-ac \
  /tmp/nativelink/data-worker-test/tmp_path-ac \
  /tmp/nativelink/data-worker-test/content_path-cas \
  /tmp/nativelink/data-worker-test/tmp_path-cas

BIN="$CACHE/bin/nativelink"
if [[ ! -x "$BIN" ]]; then
  TGZ="nativelink-${VER}-x86_64-unknown-linux-musl.tar.gz"
  URL="https://github.com/TraceMachina/nativelink/releases/download/v${VER}/${TGZ}"
  echo "Downloading $URL"
  curl -fL --retry 5 -o "$CACHE/$TGZ" "$URL"
  tar -xzf "$CACHE/$TGZ" -C "$CACHE/bin"
  if [[ ! -x "$BIN" ]]; then
    found=$(find "$CACHE/bin" -type f -name nativelink | head -n 1)
    if [[ -n "$found" ]]; then
      ln -sfn "$found" "$BIN"
    fi
  fi
  chmod +x "$BIN" || true
fi
"$BIN" -V 2>/dev/null || "$BIN" --version 2>/dev/null || ls -la "$CACHE/bin"

CONFIG="${NATIVELINK_CONFIG:-$SCRIPT_DIR/basic_cas.json5}"
if [[ ! -f "$CONFIG" ]]; then
  echo "NativeLink config not found: $CONFIG" >&2
  exit 1
fi

# Allow overriding the public port via env without editing the checked-in config.
RUNTIME_CONFIG="$CACHE/basic_cas.runtime.json5"
sed -e "s/0.0.0.0:1985/0.0.0.0:${PORT}/g" "$CONFIG" >"$RUNTIME_CONFIG"

echo "Starting nativelink on 0.0.0.0:${PORT} config=${RUNTIME_CONFIG}"
exec "$BIN" "$RUNTIME_CONFIG"
