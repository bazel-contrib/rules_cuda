"""Regression tests for CUDA toolchain environments and runtime libraries."""

load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load("//cuda/private:action_names.bzl", "ACTION_NAMES")
load("//cuda/private:providers.bzl", "CudaToolchainConfigInfo", "CudaToolkitInfo")
load("//cuda/private:toolchain_config_lib.bzl", "config_helper")

def _environment_test_impl(ctx):
    env = unittest.begin(ctx)
    config = ctx.attr.config[CudaToolchainConfigInfo]
    features = config_helper.configure_features(
        selectables = config.features,
        requested_features = ["nvcc_compile_env", "nvcc_device_link_env"],
    )
    for action in [ACTION_NAMES.cuda_compile, ACTION_NAMES.device_link]:
        action_env = config_helper.get_environment_variables(features, action, struct())
        path = action_env["PATH"]
        separator = ctx.attr.separator
        asserts.true(env, "C:/CUDA/bin" + separator in path, path)
        tool_dir = config.cuda_toolkit[CudaToolkitInfo].bin2c.dirname
        asserts.true(env, tool_dir in path.split(separator), path)
        if separator == ";":
            # Splitting on ':' would corrupt drive-qualified Windows paths.
            asserts.true(env, "C:/CUDA/bin" in path.split(separator), path)
            asserts.true(env, "C:/Windows/system32" in path.split(separator), path)
        else:
            asserts.false(env, "C:/Windows/system32" in path, path)
    return unittest.end(env)

environment_test = unittest.make(
    _environment_test_impl,
    attrs = {
        "config": attr.label(mandatory = True, providers = [CudaToolchainConfigInfo]),
        "separator": attr.string(mandatory = True),
    },
)

def _runtime_libraries_test_impl(ctx):
    env = unittest.begin(ctx)
    toolkit = ctx.attr.toolkit[CudaToolkitInfo]
    asserts.equals(env, ["cudadevrt.lib", "libcudadevrt.a", "libculibos.a"], sorted([
        f.basename
        for f in toolkit.device_runtime_static_libs.to_list()
    ]))
    return unittest.end(env)

runtime_libraries_test = unittest.make(
    _runtime_libraries_test_impl,
    attrs = {
        "toolkit": attr.label(mandatory = True, providers = [CudaToolkitInfo]),
    },
)
