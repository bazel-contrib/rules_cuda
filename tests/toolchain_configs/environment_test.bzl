"""Regression tests for CUDA toolchain environments and runtime libraries."""

load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load("//cuda/private:providers.bzl", "CudaToolkitInfo")

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
