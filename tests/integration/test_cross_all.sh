#!/usr/bin/env bash
# Full-build redist_json cross-compile integration tests (bzlmod only).
#
# The workspaces are keyed by exec/target platforms. The host is supplied by
# the machine running this script:
#
#   linux-x86_64 exec / linux-sbsa target
#   linux-sbsa exec / linux-x86_64 target
#
# Env:
#   CROSS_REMOTE_BAZEL_FLAGS  Required on a non-Linux host, e.g.
#     --remote_executor=grpc://127.0.0.1:1985
#     The included setup runs NativeLink under WSL (rbe/start_wsl_worker.ps1).
#     rbe/start_qemu_worker.ps1 can use a qemu-system guest instead.
#   CUDA_REDIST_VERSION_OVERRIDE  Optional CUDA redist version pin.
#
# Flags:
#   --no-lx64-exec     skip linux-x86_64 exec / linux-sbsa target
#   --no-lsbsa-exec    skip linux-sbsa exec / linux-x86_64 target

set -euo pipefail

this_dir=$(cd "$(dirname "$0")" && pwd)
LOG_DIR="${LOG_DIR:-${TMPDIR:-/tmp}/rules_cuda_cross}"
mkdir -p "${LOG_DIR}"

skip_lx64_exec=false
skip_lsbsa_exec=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-lx64-exec) skip_lx64_exec=true; shift ;;
        --no-lsbsa-exec) skip_lsbsa_exec=true; shift ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--no-lx64-exec] [--no-lsbsa-exec]" >&2
            exit 2
            ;;
    esac
done

host_kernel=$(uname -s 2>/dev/null || true)
is_linux=false
if [[ "${RUNNER_OS:-}" == "Linux" ]] || [[ "$host_kernel" == "Linux" ]]; then
    is_linux=true
fi

remote_flags=()
if [[ -n "${CROSS_REMOTE_BAZEL_FLAGS:-}" ]]; then
    read -r -a remote_flags <<<"${CROSS_REMOTE_BAZEL_FLAGS}"
fi

if [[ "$is_linux" != true && ${#remote_flags[@]} -eq 0 ]]; then
    echo "CROSS_REMOTE_BAZEL_FLAGS is required when the Bazel client is not running on Linux" >&2
    exit 1
fi

PLATFORMS_PKG="@rules_cuda//tests/integration/platforms"
AARCH64_CC_TC="${PLATFORMS_PKG}:aarch64_linux_cc_toolchain"
# Hermetic deliverable toolkits expose this target; MODULE only registers the
# host alias (nvcc-local-toolchain). A non-Linux host needs the Linux toolchain
# registered explicitly for remote execution.
NVCC_LINUX_TC="@cuda//toolchain:nvcc-linux-toolchain"

assert_redist_platforms() {
    local expect_exec_plat="$1"
    local expect_tgt_plat="$2"
    local tag="$3"
    shift 3

    local aq_file cq_file
    aq_file="${LOG_DIR}/${tag}_aquery.txt"
    cq_file="${LOG_DIR}/${tag}_cquery.txt"

    if ! bazel aquery "$@" //:use_rule >"${aq_file}" 2>"${LOG_DIR}/${tag}_aquery.err"; then
        cat "${LOG_DIR}/${tag}_aquery.err" >&2
        exit 1
    fi
    if ! grep -q "${expect_exec_plat}" "${aq_file}"; then
        echo "ASSERT FAIL (exec redist): expected '${expect_exec_plat}' in aquery //:use_rule" >&2
        grep -E 'cuda_nvcc_linux_|/nvcc/bin/nvcc' "${aq_file}" | head -n 20 >&2 || true
        exit 1
    fi
    echo "ASSERT OK (exec redist): ${expect_exec_plat}"

    if ! bazel cquery "$@" 'deps(//:use_library)' >"${cq_file}" 2>"${LOG_DIR}/${tag}_cquery.err"; then
        cat "${LOG_DIR}/${tag}_cquery.err" >&2
        exit 1
    fi
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

remote_toolchain_flags=()
if [[ ${#remote_flags[@]} -gt 0 ]]; then
    remote_toolchain_flags=(
        --extra_toolchains="${NVCC_LINUX_TC}"
        --extra_execution_platforms="${PLATFORMS_PKG}:linux_x86_64"
        --host_platform="${PLATFORMS_PKG}:linux_x86_64"
    )
fi

if [[ "$skip_lx64_exec" == false ]]; then
    run_case \
        "linux x64 exec / linux-sbsa target" \
        "toolchain_redist_cross_lx64_exec_lsbsa_tgt" \
        "${PLATFORMS_PKG}:linux_sbsa" \
        "linux-x86_64" \
        "cuda_nvcc_linux_x86_64" \
        "linux_sbsa" \
        "lx64_exec_lsbsa_tgt" \
        --extra_toolchains="${AARCH64_CC_TC}" \
        "${remote_toolchain_flags[@]}"

    if [[ "$is_linux" == true && ${#remote_flags[@]} -eq 0 ]]; then
        pushd "${this_dir}/toolchain_redist_cross_lx64_exec_lsbsa_tgt" >/dev/null
        assert_artifact_machine "AArch64|aarch64"
        bazel shutdown || true
        popd >/dev/null
    fi
fi

if [[ "$skip_lsbsa_exec" == false ]]; then
    if [[ "$is_linux" == true && ${#remote_flags[@]} -eq 0 ]] && \
       ! command -v qemu-aarch64-static >/dev/null 2>&1 && \
       ! command -v qemu-aarch64 >/dev/null 2>&1 && \
       ! [[ -e /proc/sys/fs/binfmt_misc/qemu-aarch64 ]]; then
        echo "WARN: qemu-aarch64 / binfmt not detected; linux-sbsa tools may not run" >&2
    fi

    run_case \
        "linux-sbsa exec / linux x64 target" \
        "toolchain_redist_cross_lsbsa_exec_lx64_tgt" \
        "${PLATFORMS_PKG}:linux_x86_64" \
        "linux-sbsa" \
        "cuda_nvcc_linux_sbsa" \
        "linux_x86_64" \
        "lsbsa_exec_lx64_tgt" \
        "${remote_toolchain_flags[@]}"

    if [[ "$is_linux" == true && ${#remote_flags[@]} -eq 0 ]]; then
        pushd "${this_dir}/toolchain_redist_cross_lsbsa_exec_lx64_tgt" >/dev/null
        local_flags=(
            --enable_bzlmod
            --platforms="${PLATFORMS_PKG}:linux_x86_64"
            --@rules_cuda//cuda:exec_platform=linux-sbsa
            --@rules_cuda//cuda:aarch64=sbsa
            --@rules_cuda//cuda:enable=True
        )
        bazel build "${local_flags[@]}" //:smoke
        assert_artifact_machine "X86-64|x86-64|x86_64|Advanced Micro Devices X86-64"
        smoke_bin=$(readlink -f bazel-bin/smoke)
        echo "RUN ${smoke_bin}"
        out=$("${smoke_bin}")
        echo "${out}"
        grep -q rules_cuda_cross_smoke_ok <<<"${out}"
        echo "ASSERT OK (smoke run)"
        bazel shutdown || true
        popd >/dev/null
    fi
fi

echo
echo "Cross-compilation integration tests passed."
