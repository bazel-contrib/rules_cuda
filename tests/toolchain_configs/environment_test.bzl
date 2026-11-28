"""Regression tests for CUDA toolchain environments and runtime libraries."""

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_skylib//lib:unittest.bzl", "asserts", "unittest")
load("//cuda/private:action_names.bzl", "ACTION_NAMES")
load("//cuda/private:providers.bzl", "CudaToolchainConfigInfo", "CudaToolkitInfo")
load("//cuda/private:toolchain_config_lib.bzl", "config_helper", "eval_feature")

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

def _msvc_device_link_flags_test_impl(ctx):
    env = unittest.begin(ctx)
    config = ctx.attr.config[CudaToolchainConfigInfo]
    features = config_helper.configure_features(
        selectables = config.features,
        requested_features = [ACTION_NAMES.device_link, ACTION_NAMES.cuda_compile] + ctx.attr.requested,
    )
    value = struct(host_compiler = "C:/VC/bin/cl.exe")
    crt_feature = ctx.attr.requested[-1]
    for action in [ACTION_NAMES.cuda_compile, ACTION_NAMES.device_link]:
        # Evaluate only the host compiler and CRT features; the full command line needs many more variables.
        flags = []
        for feature in config.features:
            if feature.name in ["host_compiler_path", crt_feature]:
                asserts.true(env, config_helper.is_enabled(features, feature.name), feature.name)
                flags.extend(eval_feature(feature, value, action, features))
        asserts.true(env, _has_pair(flags, "-ccbin", "C:/VC/bin/cl.exe"), "%s: %s" % (action, flags))
        asserts.true(env, _has_pair(flags, "-Xcompiler", ctx.attr.crt_flag), "%s: %s" % (action, flags))
    return unittest.end(env)

def _has_pair(flags, first, second):
    for i in range(len(flags) - 1):
        if flags[i] == first and flags[i + 1] == second:
            return True
    return False

msvc_device_link_flags_test = unittest.make(
    _msvc_device_link_flags_test_impl,
    attrs = {
        "config": attr.label(mandatory = True, providers = [CudaToolchainConfigInfo]),
        "crt_flag": attr.string(mandatory = True),
        "requested": attr.string_list(mandatory = True),
    },
)

def _assembled_root_test_impl(ctx):
    env = unittest.begin(ctx)
    toolkit = ctx.attr.toolkit[CudaToolkitInfo]
    asserts.equals(env, sorted([
        "bin/" + toolkit.ptxas.basename,
        "bin/" + toolkit.fatbinary.basename,
        "include/cuda.h",
        "nvvm/libdevice/libdevice.10.bc",
    ]), sorted([
        paths.relativize(f.path, toolkit.path)
        for f in ctx.attr.toolkit[DefaultInfo].files.to_list()
    ]))
    return unittest.end(env)

assembled_root_test = unittest.make(
    _assembled_root_test_impl,
    attrs = {
        "toolkit": attr.label(mandatory = True, providers = [CudaToolkitInfo]),
    },
)

def _clang_cuda_path_test_impl(ctx):
    env = unittest.begin(ctx)
    config = ctx.attr.config[CudaToolchainConfigInfo]
    features = config_helper.configure_features(selectables = config.features, requested_features = ["cuda_path"])
    flags = []
    for feature in config.features:
        if feature.name == "cuda_path":
            flags.extend(eval_feature(feature, struct(), ACTION_NAMES.cuda_compile, features))
    asserts.equals(env, ["--cuda-path=" + config.cuda_toolkit[CudaToolkitInfo].path], flags)
    return unittest.end(env)

clang_cuda_path_test = unittest.make(
    _clang_cuda_path_test_impl,
    attrs = {
        "config": attr.label(mandatory = True, providers = [CudaToolchainConfigInfo]),
    },
)

def _objdir_as_tempdir_test_impl(ctx):
    env = unittest.begin(ctx)
    config = ctx.attr.config[CudaToolchainConfigInfo]
    features = config_helper.configure_features(selectables = config.features, requested_features = ["nvcc_objdir_as_tempdir"])
    flags = []
    for feature in config.features:
        if feature.name == "nvcc_objdir_as_tempdir":
            flags.extend(eval_feature(feature, struct(), ACTION_NAMES.cuda_compile, features))
            asserts.equals(env, [], eval_feature(feature, struct(), ACTION_NAMES.device_link, features))
    asserts.equals(env, ctx.attr.expected, flags)
    return unittest.end(env)

objdir_as_tempdir_test = unittest.make(
    _objdir_as_tempdir_test_impl,
    attrs = {
        "config": attr.label(mandatory = True, providers = [CudaToolchainConfigInfo]),
        "expected": attr.string_list(),
    },
)
