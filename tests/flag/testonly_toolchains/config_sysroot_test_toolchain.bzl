load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl", "tool_path")

def _impl(ctx):
    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = ctx.attr.toolchain_identifier,
        host_system_name = "local",
        target_system_name = "aarch64-linux-gnu",
        target_cpu = "aarch64",
        compiler = "gcc",
        target_libc = "glibc-2.2.2",
        cc_target_os = None,
        builtin_sysroot = ctx.attr.builtin_sysroot if ctx.attr.builtin_sysroot else None,
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
    attrs = {
        "toolchain_identifier": attr.string(),
        "builtin_sysroot": attr.string(),
    },
    provides = [CcToolchainConfigInfo],
)

def config_sysroot_test_toolchain(identifier, sysroot):
    toolchain_config_name = "toolchain-" + identifier + "-test-config"
    cc_toolchain_name = identifier + "-test-cc-toolchain"
    toolchain_name = identifier + "-test-toolchain"
    platform_name = identifier + "-test-platform"

    constraint_name = identifier + "constraint"
    constraint_meet = identifier + "constraint-meet"

    native.constraint_setting(name = constraint_name)
    native.constraint_value(
        name = constraint_meet,
        constraint_setting = ":" + constraint_name,
    )

    optional_sysroot = {"builtin_sysroot": sysroot} if sysroot else {}
    flag_test_toolchain_config(
        name = toolchain_config_name,
        toolchain_identifier = identifier,
        **optional_sysroot,
    )

    native.cc_toolchain(
        name = cc_toolchain_name,
        all_files = ":empty",
        compiler_files = ":empty",
        dwp_files = ":empty",
        linker_files = ":empty",
        objcopy_files = ":empty",
        strip_files = ":empty",
        supports_param_files = 1,
        toolchain_config = ":" + toolchain_config_name,
    )

    native.toolchain(
        name = toolchain_name,
        exec_compatible_with = [
            "@platforms//cpu:x86_64",
            "@platforms//os:linux",
        ],
        target_compatible_with = [
            "@platforms//cpu:aarch64",
            "@platforms//os:linux",
            ":" + constraint_meet,
        ],
        toolchain = ":" + cc_toolchain_name,
        toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
    )

    native.platform(
        name = platform_name,
        constraint_values = [
            "@platforms//cpu:aarch64",
            "@platforms//os:linux",
            ":" + constraint_meet,
        ],
    )
