load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl", "tool_path")

def _impl(ctx):
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "flag-test-toolchain",
        host_system_name = "local",
        target_system_name = "aarch64-linux-gnu",
        target_cpu = "aarch64",
        compiler = "gcc",
        target_libc = "glibc-2.2.2",
        # abi_version = "gcc",
        # abi_libc_version = abi_libc_version,
        cc_target_os = None,
        builtin_sysroot = "/sysroot/for/flag/test",
        tool_paths = [
            tool_path(name = "gcc", path = "/usr/bin/aarch64-linux-gnu-gcc"),
            tool_path(name = "ld", path = "/usr/bin/aarch64-linux-gnu-ld"),
            tool_path(name = "ar", path = "/usr/bin/aarch64-linux-gnu-ar"),
            tool_path(name = "cpp", path = "/usr/bin/aarch64-linux-gnu-g++"),
            tool_path(name = "strip", path = "/usr/bin/aarch64-linux-gnu-strip"),
            tool_path(name = "nm", path = "/usr/bin/aarch64-linux-gnu-nm"),
            tool_path(name = "objdump", path = "/usr/bin/aarch64-linux-gnu-objdump"),
            tool_path(name = "objcopy", path = "/usr/bin/aarch64-linux-gnu-objcopy"),
        ],
    )

flag_test_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)
