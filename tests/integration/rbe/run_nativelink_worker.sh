#!/usr/bin/env bash
# Long-lived NativeLink entrypoint for Windows Start-Process -> wsl.exe.
# Logging stays inside WSL; RBE_PORT is honored by bootstrap_nativelink_linux.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="${NATIVELINK_LOG:-$HOME/.cache/rules_cuda-rbe/nativelink.log}"
mkdir -p "$(dirname "$LOG")" /tmp/nativelink/work
exec bash "$SCRIPT_DIR/bootstrap_nativelink_linux.sh" >"$LOG" 2>&1
