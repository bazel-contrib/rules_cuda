load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", CC_ACTION_NAMES = "ACTION_NAMES")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")
load("//cuda/private:providers.bzl", "CudaToolkitInfo")

def nvcc_version_ge(ctx, major, minor):
    if ctx.attr.toolchain_identifier != "nvcc":
        return False

    return (ctx.attr.nvcc_version_major, ctx.attr.nvcc_version_minor) >= (major, minor)

def _non_empty(list):
    return [elem for elem in list if elem]

def _unique(list):
    unique_list = []
    for elem in list:
        if elem not in unique_list:
            unique_list.append(elem)
    return unique_list

def collect_paths(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    cc_feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    host_compiler = cc_common.get_tool_for_action(
        feature_configuration = cc_feature_configuration,
        action_name = CC_ACTION_NAMES.cpp_compile,
    )
    cc_compile_variables = cc_common.create_compile_variables(
        feature_configuration = cc_feature_configuration,
        cc_toolchain = cc_toolchain,
    )
    env = cc_common.get_environment_variables(
        feature_configuration = cc_feature_configuration,
        action_name = CC_ACTION_NAMES.cpp_compile,
        variables = cc_compile_variables,
    )

    cicc_dir = ctx.attr.cuda_toolkit[CudaToolkitInfo].cicc.dirname if ctx.attr.cuda_toolkit[CudaToolkitInfo].cicc else None
    libdevice_dir = ctx.attr.cuda_toolkit[CudaToolkitInfo].libdevice.dirname if ctx.attr.cuda_toolkit[CudaToolkitInfo].libdevice else None

    path_separator = ctx.configuration.host_path_separator
    env_paths = [paths.dirname(host_compiler)]

    if libdevice_dir:
        # <libdevice_dir>/libdevice.10.bc
        # <nvvm_root>/nvvm/libdevice/libdevice.10.bc
        # <nvvm_root>/nvvm/bin
        env_paths.append(paths.join(paths.dirname(libdevice_dir), "bin"))
    if ctx.attr.cuda_toolkit[CudaToolkitInfo].path:
        env_paths.append(paths.join(ctx.attr.cuda_toolkit[CudaToolkitInfo].path, "bin"))
    for tool_name in ["cicc", "bin2c", "nvlink", "fatbinary", "ptxas"]:
        tool_file = getattr(ctx.attr.cuda_toolkit[CudaToolkitInfo], tool_name, None)
        if tool_file:
            env_paths.append(tool_file.dirname)

    if env.get("PATH"):
        env_paths.extend(env.get("PATH").split(path_separator))
    if ctx.configuration.default_shell_env.get("PATH"):
        env_paths.extend(ctx.configuration.default_shell_env.get("PATH").split(path_separator))

    is_windows = "windows" in ctx.var["TARGET_CPU"]
    if is_windows:
        env_paths.append("C:/Windows/system32")

    env_paths = _unique(_non_empty(env_paths))

    env_includes = cc_toolchain.built_in_include_directories

    return env_paths, env_includes, cicc_dir, libdevice_dir
