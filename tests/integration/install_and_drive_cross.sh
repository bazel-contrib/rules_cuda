#!/usr/bin/env bash
# Install cross-compile deps via apt (sudo) and drive cases 1–2.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INT="${ROOT}/tests/integration"
LOG_DIR="${LOG_DIR:-/tmp/rules_cuda_cross}"
mkdir -p "${LOG_DIR}"

# Prefer non-interactive sudo; password via SUDO_PASS or default local convention.
SUDO_PASS="${SUDO_PASS:-cloud}"
export DEBIAN_FRONTEND=noninteractive

log() { echo "[$(date -Iseconds)] $*"; }

sudo_run() {
  if sudo -n true 2>/dev/null; then
    sudo "$@"
  else
    printf '%s\n' "${SUDO_PASS}" | sudo -S -p '' "$@"
  fi
}

log "=== apt install cross deps ==="
sudo_run apt-get update -qq
sudo_run apt-get install -y --no-install-recommends \
  g++-aarch64-linux-gnu \
  gcc-aarch64-linux-gnu \
  binutils-aarch64-linux-gnu \
  qemu-user-static \
  binfmt-support \
  file \
  binutils

log "aarch64-g++: $(command -v aarch64-linux-gnu-g++)"
log "qemu-aarch64: $(command -v qemu-aarch64-static || command -v qemu-aarch64 || true)"
aarch64-linux-gnu-g++ --version | head -n 1
qemu-aarch64-static --version 2>/dev/null | head -n 1 || qemu-aarch64 --version 2>/dev/null | head -n 1 || true

# Ensure binfmt for aarch64 is enabled (best-effort)
if command -v update-binfmts >/dev/null 2>&1; then
  sudo_run update-binfmts --enable qemu-aarch64 2>/dev/null || true
fi
if [[ -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]]; then
  log "binfmt qemu-aarch64: registered"
  head -n 5 /proc/sys/fs/binfmt_misc/qemu-aarch64 || true
else
  log "WARN: binfmt qemu-aarch64 not visible; case 2 may need wrappers"
fi

# Guest aarch64 ELFs request /lib/ld-linux-aarch64.so.1 and libs under
# /lib/aarch64-linux-gnu. Map those to the multiarch paths from
# libc6-arm64-cross so qemu-user-static binfmt can run sbsa nvcc.
LD_SRC="$(find /usr/aarch64-linux-gnu -name 'ld-linux-aarch64.so.1' 2>/dev/null | head -n 1 || true)"
if [[ -n "${LD_SRC}" ]]; then
  sudo_run mkdir -p /lib
  sudo_run ln -sfn "${LD_SRC}" /lib/ld-linux-aarch64.so.1
  if [[ -d /usr/aarch64-linux-gnu/lib ]]; then
    # Replace mistaken nested dirs from prior attempts
    if [[ -d /lib/aarch64-linux-gnu && ! -L /lib/aarch64-linux-gnu ]]; then
      sudo_run rm -rf /lib/aarch64-linux-gnu
    fi
    sudo_run ln -sfn /usr/aarch64-linux-gnu/lib /lib/aarch64-linux-gnu
  fi
  log "mapped aarch64 guest loader: /lib/ld-linux-aarch64.so.1 -> ${LD_SRC}"
fi

# Restore distro tool paths for the Starlark cc toolchain
cat > "${INT}/platforms/local_tool_paths.bzl" <<'EOF'
"""Distro aarch64 cross tool paths (apt packages)."""

AARCH64_TOOL_PATHS = {
    "gcc": "/usr/bin/aarch64-linux-gnu-gcc",
    "g++": "/usr/bin/aarch64-linux-gnu-g++",
    "cpp": "/usr/bin/aarch64-linux-gnu-g++",
    "ar": "/usr/bin/aarch64-linux-gnu-ar",
    "ld": "/usr/bin/aarch64-linux-gnu-ld",
    "nm": "/usr/bin/aarch64-linux-gnu-nm",
    "objcopy": "/usr/bin/aarch64-linux-gnu-objcopy",
    "objdump": "/usr/bin/aarch64-linux-gnu-objdump",
    "strip": "/usr/bin/aarch64-linux-gnu-strip",
    "sysroot": "",
}
EOF

export USE_BAZEL_VERSION="${USE_BAZEL_VERSION:-9.2.0}"
export CUDA_REDIST_VERSION_OVERRIDE="${CUDA_REDIST_VERSION_OVERRIDE:-12.6.3}"

if command -v bazelisk >/dev/null 2>&1; then
  BAZEL=bazelisk
else
  BAZEL=bazel
fi

PLATFORMS_PKG="@rules_cuda//tests/integration/platforms"
AARCH64_CC_TC="${PLATFORMS_PKG}:aarch64_linux_cc_toolchain"

assert_arch() {
  local dir="$1"
  local expect_re="$2"
  local desc="$3"
  local f m hit=0
  while IFS= read -r -d '' f; do
    m=$(readelf -h "$f" 2>/dev/null | awk -F: '/Machine:/{gsub(/^[ \t]+/,"",$2); print $2; exit}')
    log "  artifact: ${f#"$dir"/} -> ${m} | $(file -b "$f" | head -c 100)"
    if echo "$m $(file -b "$f")" | grep -qiE "${expect_re}"; then
      hit=1
    fi
  done < <(find -L "$dir" -type f \( -name '*.o' -o -name '*.a' -o -name '*.pic.o' \) -print0 2>/dev/null)
  if [[ "$hit" -ne 1 ]]; then
    echo "ASSERT FAIL (${desc}): no /${expect_re}/ under ${dir}" >&2
    exit 1
  fi
  log "ASSERT OK (${desc})"
}

run_case1() {
  log "=== CASE 1: exec linux-x86_64 / target linux-sbsa ==="
  pushd "${INT}/toolchain_redist_cross_lx64_exec_lsbsa_tgt" >/dev/null
  local flags=(
    --enable_bzlmod
    --platforms="${PLATFORMS_PKG}:linux_sbsa"
    --@rules_cuda//cuda:exec_platform=linux-x86_64
    --@rules_cuda//cuda:aarch64=sbsa
    --@rules_cuda//cuda:enable=True
    --extra_toolchains="${AARCH64_CC_TC}"
    --verbose_failures
  )
  ${BAZEL} build "${flags[@]}" //:use_library //:use_rule 2>&1 | tee "${LOG_DIR}/apt_case1_build.log"
  ${BAZEL} aquery "${flags[@]}" //:use_rule >"${LOG_DIR}/apt_case1_aquery.txt" 2>/dev/null || true
  grep -q cuda_nvcc_linux_x86_64 "${LOG_DIR}/apt_case1_aquery.txt"
  log "ASSERT OK exec: cuda_nvcc_linux_x86_64"
  ${BAZEL} cquery "${flags[@]}" 'deps(//:use_library)' >"${LOG_DIR}/apt_case1_cquery.txt" 2>/dev/null || true
  grep -q linux_sbsa "${LOG_DIR}/apt_case1_cquery.txt"
  log "ASSERT OK target: linux_sbsa"
  assert_arch "$(readlink -f bazel-bin)" "AArch64|aarch64" "case1 artifacts aarch64"
  ${BAZEL} shutdown || true
  popd >/dev/null
  log "CASE 1 PASSED"
}

run_case2() {
  log "=== CASE 2: exec linux-sbsa (qemu) / target linux-x86_64 ==="
  pushd "${INT}/toolchain_redist_cross_lsbsa_exec_lx64_tgt" >/dev/null
  local flags=(
    --enable_bzlmod
    --platforms="${PLATFORMS_PKG}:linux_x86_64"
    --@rules_cuda//cuda:exec_platform=linux-sbsa
    --@rules_cuda//cuda:aarch64=sbsa
    --@rules_cuda//cuda:enable=True
    --verbose_failures
  )

  # If binfmt works, plain build is enough. If not, wrap tools with qemu.
  ${BAZEL} fetch "${flags[@]}" //:use_rule 2>&1 | tee "${LOG_DIR}/apt_case2_fetch.log" || true
  local ob
  ob=$(${BAZEL} info output_base)

  # Probe whether aarch64 ELF runs via binfmt + guest loader mapping
  local sample
  sample=$(find "${ob}/external" -path '*linux_sbsa*/nvcc/bin/nvcc' -type f 2>/dev/null | head -n 1 || true)
  if [[ -n "${sample}" ]]; then
    log "probe nvcc: ${sample} ($(file -b "${sample}"))"
    if "${sample}" --version >"${LOG_DIR}/apt_case2_nvcc_probe.log" 2>&1; then
      log "binfmt OK: aarch64 nvcc runs under qemu-user-static"
      cat "${LOG_DIR}/apt_case2_nvcc_probe.log"
    else
      log "binfmt probe failed; retry with QEMU_LD_PREFIX"
      cat "${LOG_DIR}/apt_case2_nvcc_probe.log" || true
      export QEMU_LD_PREFIX=/usr/aarch64-linux-gnu
      if ! QEMU_LD_PREFIX=/usr/aarch64-linux-gnu "${sample}" --version >"${LOG_DIR}/apt_case2_nvcc_probe.log" 2>&1; then
        log "applying explicit qemu -L wrappers"
        local qemu
        qemu=$(command -v qemu-aarch64-static || command -v qemu-aarch64)
        while IFS= read -r -d '' f; do
          if ! file -b "$f" | grep -qiE 'ELF 64-bit LSB (pie )?executable, ARM aarch64'; then
            continue
          fi
          if [[ -e "${f}.real-aarch64" ]]; then
            continue
          fi
          mv "$f" "${f}.real-aarch64"
          cat >"$f" <<WRAP
#!/bin/bash
exec "${qemu}" -L /usr/aarch64-linux-gnu "${f}.real-aarch64" "\$@"
WRAP
          chmod +x "$f"
        done < <(find "${ob}/external" -type f \( -name nvcc -o -name cicc -o -name ptxas -o -name nvlink -o -name fatbinary -o -name bin2c -o -name 'cudafe++' -o -name '__nvcc_device_query' \) -print0)
        flags+=(--spawn_strategy=local --strategy=CudaCompile=local --experimental_check_external_repository_files=false)
        "${qemu}" -L /usr/aarch64-linux-gnu "${sample}.real-aarch64" --version 2>&1 | tee "${LOG_DIR}/apt_case2_nvcc_probe.log" || true
      fi
    fi
  fi

  ${BAZEL} build "${flags[@]}" //:use_library //:use_rule 2>&1 | tee "${LOG_DIR}/apt_case2_build.log"
  ${BAZEL} aquery "${flags[@]}" //:use_rule >"${LOG_DIR}/apt_case2_aquery.txt" 2>/dev/null || true
  grep -q cuda_nvcc_linux_sbsa "${LOG_DIR}/apt_case2_aquery.txt"
  log "ASSERT OK exec: cuda_nvcc_linux_sbsa"
  ${BAZEL} cquery "${flags[@]}" 'deps(//:use_library)' >"${LOG_DIR}/apt_case2_cquery.txt" 2>/dev/null || true
  grep -q linux_x86_64 "${LOG_DIR}/apt_case2_cquery.txt"
  log "ASSERT OK target: linux_x86_64"
  assert_arch "$(readlink -f bazel-bin)" "X86-64|x86-64|x86_64|Advanced Micro Devices X86-64" "case2 artifacts x86_64"
  ${BAZEL} shutdown || true
  popd >/dev/null
  log "CASE 2 PASSED"
}

main() {
  log "host=$(uname -m) root=${ROOT}"
  sudo_run true
  log "sudo OK"
  run_case1
  run_case2
  log "=== ALL PASSED (apt-backed) ==="
}

main "$@"
