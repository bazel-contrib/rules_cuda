"""Tests verifying synthetic Clang cuda.h roots are never propagated to consumers."""

load("@bazel_skylib//lib:unittest.bzl", "analysistest", "asserts")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _all_include_dirs(cc_info):
    ctx = cc_info.compilation_context
    return (
        ctx.includes.to_list() +
        ctx.system_includes.to_list() +
        ctx.quote_includes.to_list() +
        ctx.external_includes.to_list()
    )

def _no_synthetic_include_test_impl(ctx):
    env = analysistest.begin(ctx)
    target = analysistest.target_under_test(env)
    cc_info = target[CcInfo]

    include_dirs = _all_include_dirs(cc_info)
    for inc in include_dirs:
        # <component_repo_root>/nvvm/include is the real nvvm header directory;
        # <component_repo_root>/include holds the synthetic cuda.h for Clang and must
        # never be propagated.
        is_synthetic_nvvm_inc = inc.endswith("/nvvm/include") and not inc.endswith("/nvvm/nvvm/include")
        asserts.false(
            env,
            is_synthetic_nvvm_inc,
            "Synthetic <component_repo_root>/include directory must not be propagated: {}".format(inc),
        )
        asserts.false(
            env,
            "_clang_cuda_root" in inc,
            "Synthetic nvcc _clang_cuda_root directory must not be propagated: {}".format(inc),
        )

    if ctx.attr.require_nvvm_inner_include:
        asserts.true(
            env,
            any([inc.endswith("/nvvm/nvvm/include") for inc in include_dirs]),
            "Expected <component_repo_root>/nvvm/include in propagated includes: {}".format(include_dirs),
        )

    for hdr in cc_info.compilation_context.headers.to_list():
        if hdr.basename == "cuda.h":
            asserts.true(
                env,
                ctx.attr.allow_cudart_cuda_h and "/cudart/include/cuda.h" in hdr.path,
                "Unexpected cuda.h in propagated headers: {}".format(hdr.path),
            )

    return analysistest.end(env)

no_synthetic_include_test = analysistest.make(
    _no_synthetic_include_test_impl,
    attrs = {
        "allow_cudart_cuda_h": attr.bool(default = False),
        "require_nvvm_inner_include": attr.bool(default = False),
    },
)
