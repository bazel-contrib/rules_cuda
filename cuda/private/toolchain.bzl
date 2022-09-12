load("//cuda/private:providers.bzl", "CudaToolchainConfigInfo", "CudaToolkitInfo")
load("//cuda/private:toolchain_config_lib.bzl", "config_helper")

def _cuda_toolchain_impl(ctx):
    cuda_toolchain_config = ctx.attr.toolchain_config[CudaToolchainConfigInfo]
    selectables_info = config_helper.collect_selectables_info(cuda_toolchain_config.action_configs + cuda_toolchain_config.features)
    must_have_selectables = []
    for name in must_have_selectables:
        if not config_helper.is_configured(selectables_info, name):
            fail(name, "is not configured (not exists) in the provided toolchain_config")

    artifact_name_patterns = {}
    for pattern in cuda_toolchain_config.artifact_name_patterns:
        artifact_name_patterns[pattern.category_name] = pattern

    return [
        platform_common.ToolchainInfo(
            name = ctx.label.name,
            compiler_executable = ctx.attr.compiler_executable,
            selectables_info = selectables_info,
            artifact_name_patterns = artifact_name_patterns,
            cuda_toolkit = cuda_toolchain_config.cuda_toolkit,
        ),
    ]

cuda_toolchain = rule(
    implementation = _cuda_toolchain_impl,
    attrs = {
        "toolchain_config": attr.label(
            mandatory = True,
            providers = [CudaToolchainConfigInfo],
        ),
        "compiler_executable": attr.string(
            mandatory = True,
        ),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),
    },
)

CPP_TOOLCHAIN_TYPE = "@bazel_tools//tools/cpp:toolchain_type"
CUDA_TOOLCHAIN_TYPE = "//cuda:toolchain_type"

def use_cpp_toolchain(mandatory = True):
    """Helper to depend on the c++ toolchain.

    Copied from https://github.com/bazelbuild/bazel/blob/ac48e65f70/tools/cpp/toolchain_utils.bzl#L53-L72
    """
    return [CPP_TOOLCHAIN_TYPE]

def use_cuda_toolchain():
    return [CUDA_TOOLCHAIN_TYPE]

def find_cuda_toolchain(ctx):
    return ctx.toolchains[CUDA_TOOLCHAIN_TYPE]

def find_cuda_toolkit(ctx):
    return ctx.toolchains[CUDA_TOOLCHAIN_TYPE].cuda_toolkit[CudaToolkitInfo]

# buildifier: disable=unnamed-macro
def register_detected_cuda_toolchains():
    """Helper to register the automatically detected CUDA toolchain(s).

User can setup their own toolchain if needed and ignore the detected ones by not calling this macro.
"""
    native.register_toolchains(
        "@local_cuda//toolchain:nvcc-local-toolchain",
        "@local_cuda//toolchain/clang:clang-local-toolchain",
    )
