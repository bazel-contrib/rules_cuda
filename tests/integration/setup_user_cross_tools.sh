#!/usr/bin/env bash
# Install user-local aarch64 cross-gcc + qemu-aarch64 (no root).
# Optionally register binfmt via user namespace if possible.
set -euo pipefail

TOOLS="${CROSS_TOOLS_DIR:-$HOME/.cache/rules_cuda-cross-tools}"
mkdir -p "$TOOLS"
cd "$TOOLS"

log() { echo "[setup] $*"; }

# --- qemu-aarch64 static (multiarch release) ---
QEMU_VER="v7.2.0-1"
QEMU_BIN="$TOOLS/qemu-aarch64-static"
if [[ ! -x "$QEMU_BIN" ]]; then
  log "downloading qemu-aarch64-static ${QEMU_VER}"
  curl -fsSL -o "$QEMU_BIN" \
    "https://github.com/multiarch/qemu-user-static/releases/download/${QEMU_VER}/qemu-aarch64-static"
  chmod +x "$QEMU_BIN"
fi
"$QEMU_BIN" -version | head -n 1 || true

# --- Arm GNU Toolchain aarch64-none-linux-gnu (user extractable) ---
# Use a recent x86_64 host toolchain targeting aarch64-linux-gnu.
ARM_NAME="arm-gnu-toolchain-13.3.rel1-x86_64-aarch64-none-linux-gnu"
ARM_TGZ="${ARM_NAME}.tar.xz"
ARM_URL="https://developer.arm.com/-/media/Files/downloads/gnu/13.3.rel1/binrel/${ARM_TGZ}"
ARM_DIR="$TOOLS/${ARM_NAME}"
if [[ ! -x "$ARM_DIR/bin/aarch64-none-linux-gnu-g++" ]]; then
  if [[ ! -f "$ARM_TGZ" ]]; then
    log "downloading Arm GNU toolchain (large)..."
    curl -fL --retry 3 -o "$ARM_TGZ" "$ARM_URL"
  fi
  log "extracting ${ARM_TGZ}"
  tar -xf "$ARM_TGZ"
fi

# Symlinks with the names our cc_toolchain expects
BIN="$TOOLS/bin"
mkdir -p "$BIN"
for t in gcc g++ ar ld nm objcopy objdump strip cpp as; do
  src="$ARM_DIR/bin/aarch64-none-linux-gnu-${t}"
  # g++ package uses g++ not c++
  if [[ "$t" == "cpp" && ! -x "$src" ]]; then
    src="$ARM_DIR/bin/aarch64-none-linux-gnu-g++"
  fi
  if [[ -x "$src" ]]; then
    ln -sfn "$src" "$BIN/aarch64-linux-gnu-${t}"
  fi
done
# Also alias cpp -> g++ if missing
[[ -x "$BIN/aarch64-linux-gnu-cpp" ]] || ln -sfn "$BIN/aarch64-linux-gnu-g++" "$BIN/aarch64-linux-gnu-cpp"
[[ -x "$BIN/aarch64-linux-gnu-g++" ]] || ln -sfn "$ARM_DIR/bin/aarch64-none-linux-gnu-g++" "$BIN/aarch64-linux-gnu-g++"
[[ -x "$BIN/aarch64-linux-gnu-gcc" ]] || ln -sfn "$ARM_DIR/bin/aarch64-none-linux-gnu-gcc" "$BIN/aarch64-linux-gnu-gcc"

log "cross gcc: $($BIN/aarch64-linux-gnu-gcc --version | head -n 1)"
log "cross g++: $($BIN/aarch64-linux-gnu-g++ --version | head -n 1)"

# Export helper file for other scripts
cat > "$TOOLS/env.sh" <<EOF
export CROSS_TOOLS_DIR="$TOOLS"
export PATH="$BIN:\$PATH"
export QEMU_AARCH64="$QEMU_BIN"
# Sysroot for the Arm toolchain (needed for linking)
export AARCH64_SYSROOT="$ARM_DIR/aarch64-none-linux-gnu/libc"
EOF

# Try user-namespace binfmt registration (best-effort)
register_binfmt() {
  if [[ -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]]; then
    log "binfmt qemu-aarch64 already registered"
    return 0
  fi
  if ! command -v unshare >/dev/null 2>&1; then
    log "no unshare; skip binfmt"
    return 1
  fi
  # Magic for ELF aarch64 executable
  # See qemu-user-static packaging
  local magic='\\x7fELF\\x02\\x01\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x02\\x00\\xb7\\x00'
  local mask='\\xff\\xff\\xff\\xff\\xff\\xff\\xff\\x00\\xff\\xff\\xff\\xff\\xff\\xff\\xff\\xff\\xfe\\xff\\xff\\xff'
  log "attempting binfmt register via unshare (may fail without privileges)"
  if unshare --user --map-root-user --mount bash -c "
    mount -t binfmt_misc binfmt_misc /proc/sys/fs/binfmt_misc 2>/dev/null || true
    if [[ -w /proc/sys/fs/binfmt_misc/register ]]; then
      echo -1 > /proc/sys/fs/binfmt_misc/qemu-aarch64 2>/dev/null || true
      printf ':qemu-aarch64:M::${magic}:${mask}:${QEMU_BIN}:CF' > /proc/sys/fs/binfmt_misc/register
      echo ok
    else
      echo no-write
      exit 1
    fi
  " 2>/dev/null; then
    log "binfmt register attempt finished"
  else
    log "binfmt register failed (expected without root) — case 2 will use explicit qemu wrapper path if needed"
    return 1
  fi
}

register_binfmt || true

log "TOOLS ready at $TOOLS"
echo "$TOOLS"
