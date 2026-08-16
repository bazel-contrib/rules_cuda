#!/usr/bin/env bash
# Full-build redist_json cross-compile integration tests (bzlmod only).
#
# Keep all four combinations. Two are required (must stay green when runnable):
#
#   REQUIRED-A (case 3): windows host, linux-x86_64 exec, linux-sbsa target
#   REQUIRED-B (case 2): linux-x86_64 host, linux-sbsa exec, linux-x86_64 target
#
# Optional extras:
#   case 1: linux-x86_64 host, linux-x86_64 exec, linux-sbsa target
#   case 4: windows host, linux-sbsa exec, linux-x86_64 target
#
# Env:
#   CROSS_REMOTE_BAZEL_FLAGS  Required for Windows cases (3–4), e.g.
#     --remote_executor=grpc://127.0.0.1:1985
#     (Linux RE worker, often qemu-system-x86_64 guest + hostfwd; WSL not required)
#   CUDA_REDIST_VERSION_OVERRIDE  Optional CUDA redist version pin.
#
# Flags:
#   --no-1 .. --no-4   skip individual cases
#   --no-linux / --no-windows
#   --required-only    run only REQUIRED-A and REQUIRED-B (cases 3 and 2)

set -euo pipefail

this_dir=$(cd "$(dirname "$0")" && pwd)
LOG_DIR="${LOG_DIR:-${TMPDIR:-/tmp}/rules_cuda_cross}"
mkdir -p "${LOG_DIR}"

skip_1=false
skip_2=false
skip_3=false
skip_4=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-1) skip_1=true; shift ;;
        --no-2) skip_2=true; shift ;;
        --no-3) skip_3=true; shift ;;
        --no-4) skip_4=true; shift ;;
        --no-linux) skip_1=true; skip_2=true; shift ;;
        --no-windows) skip_3=true; skip_4=true; shift ;;
        --required-only)
            # REQUIRED-B=case2, REQUIRED-A=case3
            skip_1=true
            skip_4=true
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--no-1] [--no-2] [--no-3] [--no-4] [--no-linux] [--no-windows] [--required-only]" >&2
            exit 2
            ;;
    esac
done

is_windows=false
if [[ "${RUNNER_OS:-}" == "Windows" ]] || [[ "$(uname -s 2>/dev/null || true)" =~ MINGW|MSYS|CYGWIN ]]; then
    is_windows=true
fi

# shellcheck disable=SC2206
remote_flags=( ${CROSS_REMOTE_BAZEL_FLAGS:-} )

PLATFORMS_PKG="@rules_cuda//tests/integration/platforms"
AARCH64_CC_TC="${PLATFORMS_PKG}:aarch64_linux_cc_toolchain"

assert_redist_platforms() {
    local expect_exec_plat="$1"
    local expect_tgt_plat="$2"
    local tag="$3"
    shift 3

    local aq_file cq_file
    aq_file="${LOG_DIR}/${tag}_aquery.txt"
    cq_file="${LOG_DIR}/${tag}_cquery.txt"

    bazel aquery "$@" //:use_rule >"${aq_file}" 2>"${LOG_DIR}/${tag}_aquery.err" || true
    if ! grep -q "${expect_exec_plat}" "${aq_file}"; then
        echo "ASSERT FAIL (exec redist): expected '${expect_exec_plat}' in aquery //:use_rule" >&2
        grep -E 'cuda_nvcc_linux_|/nvcc/bin/nvcc' "${aq_file}" | head -n 20 >&2 || true
        exit 1
    fi
    echo "ASSERT OK (exec redist): ${expect_exec_plat}"

    bazel cquery "$@" 'deps(//:use_library)' >"${cq_file}" 2>"${LOG_DIR}/${tag}_cquery.err" || true
    if ! grep -q "${expect_tgt_plat}" "${cq_file}"; then
        echo "ASSERT FAIL (target redist): expected '${expect_tgt_plat}' in deps(//:use_library)" >&2
        tail -n 40 "${cq_file}" >&2
        exit 1
    fi
    echo "ASSERT OK (target redist): ${expect_tgt_plat}"
}

assert_artifact_machine() {
    # $1 = expect regex for readelf Machine / file(1)
    local expect_re="$1"
    local bb
    bb=$(readlink -f bazel-bin 2>/dev/null || bazel info bazel-bin)
    local f m hit=0
    while IFS= read -r -d '' f; do
        m=$(readelf -h "$f" 2>/dev/null | awk -F: '/Machine:/{gsub(/^[ \t]+/,"",$2); print $2; exit}' || true)
        echo "  artifact: ${f#"$bb"/} -> ${m} | $(file -b "$f" 2>/dev/null | head -c 100)"
        if echo "$m $(file -b "$f" 2>/dev/null || true)" | grep -qiE "${expect_re}"; then
            hit=1
        fi
    done < <(find -L "$bb" -type f \( -name '*.o' -o -name '*.a' -o -name 'smoke' -o -name 'smoke.exe' \) -print0 2>/dev/null)
    if [[ "$hit" -ne 1 ]]; then
        echo "ASSERT FAIL: no artifact matching /${expect_re}/ under ${bb}" >&2
        exit 1
    fi
    echo "ASSERT OK (artifact arch): /${expect_re}/"
}

run_case() {
    local name="$1"
    local dir="$2"
    local platforms_flag="$3"
    local exec_platform="$4"
    local expect_exec_plat="$5"
    local expect_tgt_plat="$6"
    local tag="$7"
    shift 7

    cat <<-EOF

============================================================
=== CROSS TEST: ${name}
============================================================
EOF

    pushd "${this_dir}/${dir}" >/dev/null

    local common_flags=(
        --enable_bzlmod
        --platforms="${platforms_flag}"
        --@rules_cuda//cuda:exec_platform="${exec_platform}"
        --@rules_cuda//cuda:aarch64=sbsa
        --@rules_cuda//cuda:enable=True
        "$@"
    )
    if [[ ${#remote_flags[@]} -gt 0 ]]; then
        common_flags+=("${remote_flags[@]}")
    fi

    bazel build "${common_flags[@]}" //:use_library
    bazel build "${common_flags[@]}" //:use_rule

    assert_redist_platforms "${expect_exec_plat}" "${expect_tgt_plat}" "${tag}" "${common_flags[@]}"

    popd >/dev/null
}

# --- Optional case 1 ---
if [[ "$skip_1" == false ]]; then
    if [[ "$is_windows" == true ]]; then
        echo "SKIP case 1 (optional): requires linux-x86_64 host"
    else
        run_case \
            "optional: linux x64 host / linux x64 exec / linux-sbsa target" \
            "toolchain_redist_cross_lx64_exec_lsbsa_tgt" \
            "${PLATFORMS_PKG}:linux_sbsa" \
            "linux-x86_64" \
            "cuda_nvcc_linux_x86_64" \
            "linux_sbsa" \
            "case1" \
            --extra_toolchains="${AARCH64_CC_TC}"
        pushd "${this_dir}/toolchain_redist_cross_lx64_exec_lsbsa_tgt" >/dev/null
        assert_artifact_machine "AArch64|aarch64"
        bazel shutdown || true
        popd >/dev/null
    fi
fi

# --- REQUIRED-B (case 2) ---
if [[ "$skip_2" == false ]]; then
    if [[ "$is_windows" == true ]]; then
        echo "SKIP REQUIRED-B (case 2): requires linux-x86_64 host"
    else
        if ! command -v qemu-aarch64-static >/dev/null 2>&1 && \
           ! command -v qemu-aarch64 >/dev/null 2>&1 && \
           ! [[ -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]]; then
            echo "WARN: qemu-aarch64 / binfmt not detected; REQUIRED-B may fail" >&2
        fi
        run_case \
            "REQUIRED-B: linux x64 host / linux-sbsa exec / linux x64 target" \
            "toolchain_redist_cross_lsbsa_exec_lx64_tgt" \
            "${PLATFORMS_PKG}:linux_x86_64" \
            "linux-sbsa" \
            "cuda_nvcc_linux_sbsa" \
            "linux_x86_64" \
            "case2_required_b"

        pushd "${this_dir}/toolchain_redist_cross_lsbsa_exec_lx64_tgt" >/dev/null
        local_flags=(
            --enable_bzlmod
            --platforms="${PLATFORMS_PKG}:linux_x86_64"
            --@rules_cuda//cuda:exec_platform=linux-sbsa
            --@rules_cuda//cuda:aarch64=sbsa
            --@rules_cuda//cuda:enable=True
        )
        if [[ ${#remote_flags[@]} -gt 0 ]]; then
            local_flags+=("${remote_flags[@]}")
        fi
        # Build + run host-arch smoke (no CUDA device).
        bazel build "${local_flags[@]}" //:smoke
        assert_artifact_machine "X86-64|x86-64|x86_64|Advanced Micro Devices X86-64"
        smoke_bin=$(readlink -f bazel-bin/smoke)
        echo "RUN ${smoke_bin}"
        out=$("${smoke_bin}")
        echo "${out}"
        grep -q rules_cuda_cross_smoke_ok <<<"${out}"
        echo "ASSERT OK (REQUIRED-B smoke run)"
        bazel shutdown || true
        popd >/dev/null
    fi
fi

# --- REQUIRED-A (case 3): Windows host + Linux x64 exec + sbsa target ---
if [[ "$skip_3" == false ]]; then
    if [[ "$is_windows" != true ]]; then
        echo "SKIP REQUIRED-A (case 3): requires windows-x86_64 host (use Windows bazelisk + CROSS_REMOTE_BAZEL_FLAGS)"
    elif [[ ${#remote_flags[@]} -eq 0 ]]; then
        echo "SKIP REQUIRED-A (case 3): set CROSS_REMOTE_BAZEL_FLAGS to a Linux RE endpoint"
        echo "  e.g. start tests/integration/rbe/start_qemu_worker (hostfwd :1985) then:"
        echo "  CROSS_REMOTE_BAZEL_FLAGS='--remote_executor=grpc://127.0.0.1:1985' $0 --required-only"
    else
        run_case \
            "REQUIRED-A: windows x64 host / linux x64 exec / linux-sbsa target" \
            "toolchain_redist_cross_win_lx64_exec_lsbsa_tgt" \
            "${PLATFORMS_PKG}:linux_sbsa" \
            "linux-x86_64" \
            "cuda_nvcc_linux_x86_64" \
            "linux_sbsa" \
            "case3_required_a" \
            --extra_toolchains="${AARCH64_CC_TC}"
        pushd "${this_dir}/toolchain_redist_cross_win_lx64_exec_lsbsa_tgt" >/dev/null
        bazel shutdown || true
        popd >/dev/null
    fi
fi

# --- Optional case 4 ---
if [[ "$skip_4" == false ]]; then
    if [[ "$is_windows" != true ]]; then
        echo "SKIP case 4 (optional): requires windows-x86_64 host"
    elif [[ ${#remote_flags[@]} -eq 0 ]]; then
        echo "SKIP case 4 (optional): set CROSS_REMOTE_BAZEL_FLAGS (aarch64-capable Linux RE)"
    else
        run_case \
            "optional: windows x64 host / linux-sbsa exec / linux x64 target" \
            "toolchain_redist_cross_win_lsbsa_exec_lx64_tgt" \
            "${PLATFORMS_PKG}:linux_x86_64" \
            "linux-sbsa" \
            "cuda_nvcc_linux_sbsa" \
            "linux_x86_64" \
            "case4"
        pushd "${this_dir}/toolchain_redist_cross_win_lsbsa_exec_lx64_tgt" >/dev/null
        bazel shutdown || true
        popd >/dev/null
    fi
fi

echo
echo "All requested cross-compile integration tests finished."
