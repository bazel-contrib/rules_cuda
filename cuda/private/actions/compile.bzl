load("//cuda/private:action_names.bzl", "ACTION_NAMES")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:rules/common.bzl", "ALLOW_CUDA_SRCS")

def compile(
        ctx,
        cuda_toolchain,
        cc_toolchain,
        srcs,
        common,
        pic = False,
        rdc = False):
    """Perform CUDA compilation, return compiled object files.

    Notes:
        If `rdc` is set to `True`, then an additional step of device link must be performed.

    Args:
        ctx: A [context object](https://bazel.build/rules/lib/ctx).
        cuda_toolchain: A `platform_common.ToolchainInfo` of a cuda toolchain, Can be obtained with `find_cuda_toolchain(ctx)`.
        cc_toolchain: A `CcToolchainInfo`. Can be obtained with `find_cpp_toolchain(ctx)`.
        srcs: A list of `File`s to be compiled.
        common: A cuda common object. Can be obtained with `cuda_helper.create_common(ctx)`
        pic: Whether the `srcs` are compiled for position independent code.
        rdc: Whether the `srcs` are compiled for relocatable device code.

    Returns:
        An compiled object `File`.
    """
    actions = ctx.actions
    host_compiler = cc_toolchain.compiler_executable
    cuda_compiler = cuda_toolchain.compiler_executable

    cuda_feature_config = cuda_helper.configure_features(ctx, cuda_toolchain, requested_features = [ACTION_NAMES.cuda_compile])
    artifact_category_name = cuda_helper.get_artifact_category_from_action(ACTION_NAMES.cuda_compile, pic, rdc)

    ret = []
    for src in srcs:
        # this also filter out all header files
        basename = cuda_helper.get_basename_without_ext(src.basename, ALLOW_CUDA_SRCS, fail_if_not_match = False)
        if not basename:
            continue

        filename = cuda_helper.get_artifact_name(cuda_toolchain, artifact_category_name, basename)
        obj_file = actions.declare_file("_objs/{}/{}".format(ctx.attr.name, filename))
        ret.append(obj_file)

        var = cuda_helper.create_compile_variables(
            cuda_toolchain,
            cuda_feature_config,
            common.cuda_archs_info,
            source_file = src.path,
            output_file = obj_file.path,
            host_compiler = host_compiler,
            compile_flags = common.compile_flags,
            host_compile_flags = common.host_compile_flags,
            include_paths = common.includes,
            quote_include_paths = common.quote_includes,
            system_include_paths = common.system_includes,
            defines = common.local_defines + common.defines,
            host_defines = common.host_local_defines + common.host_defines,
            ptxas_flags = common.ptxas_flags,
            use_pic = pic,
            use_rdc = rdc,
        )
        cmd = cuda_helper.get_command_line(cuda_feature_config, ACTION_NAMES.cuda_compile, var)
        env = cuda_helper.get_environment_variables(cuda_feature_config, ACTION_NAMES.cuda_compile, var)

        args = actions.args()
        args.add_all(cmd)

        actions.run(
            executable = cuda_compiler,
            arguments = [args],
            outputs = [obj_file],
            inputs = depset([src], transitive = [common.headers, cc_toolchain.all_files, cuda_toolchain.all_files]),
            env = env,
            mnemonic = "CudaCompile",
            progress_message = "Compiling %s" % src.path,
        )
    return ret
