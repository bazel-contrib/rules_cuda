"""Toolchain configuration rule for detecting sysroot from C++ toolchain and providing it to CUDA toolchain."""

load("@bazel_tools//tools/build_defs/cc:action_names.bzl", CC_ACTION_NAMES = "ACTION_NAMES")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("//cuda/private:providers.bzl", "CudaCcSysrootInfo")

def _cuda_cc_sysroot_impl(ctx):
    """Implementation of cuda_cc_sysroot rule."""
    cc_toolchain = find_cpp_toolchain(ctx)

    cc_sysroot = None
    if ctx.attr.path:
        cc_sysroot = ctx.attr.path
    elif getattr(cc_toolchain, "sysroot", None):
        cc_sysroot = cc_toolchain.sysroot
    else:
        # Fallback to extracting sysroot from compiler flags
        feature_configuration = cc_common.configure_features(
            ctx = ctx,
            cc_toolchain = cc_toolchain,
            requested_features = ctx.features,
            unsupported_features = ctx.disabled_features,
        )

        variables = cc_common.create_compile_variables(
            cc_toolchain = cc_toolchain,
            feature_configuration = feature_configuration,
        )

        cc_flags = cc_common.get_memory_inefficient_command_line(
            feature_configuration = feature_configuration,
            action_name = CC_ACTION_NAMES.cpp_compile,
            variables = variables,
        )

        for flag in cc_flags:
            if flag.startswith("--sysroot="):
                cc_sysroot = flag.removeprefix("--sysroot=")
                break


    return [CudaCcSysrootInfo(cc_sysroot = cc_sysroot)]

cuda_cc_sysroot = rule(
    implementation = _cuda_cc_sysroot_impl,
    doc = """Detects sysroot from C++ toolchain and provides it to CUDA toolchain.

    This rule should be used in the toolchain configuration to avoid running
    sysroot detection on every CUDA target action.
    """,
    attrs = {
        "path": attr.string(
            mandatory = False,
            doc="The explicit value for sysroot, if not filled, will figure out from cc toolchain."),
        "_cc_toolchain": attr.label(
            default = "@bazel_tools//tools/cpp:current_cc_toolchain",  # legacy behaviour
        ),
    },
    provides = [CudaCcSysrootInfo],
    toolchains = use_cpp_toolchain(),
    fragments = ["cpp"],
)
