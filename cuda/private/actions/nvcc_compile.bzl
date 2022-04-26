load("//cuda/private:action_names.bzl", "ACTION_NAMES")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")
load("//cuda/private:providers.bzl", "CudaArchsInfo")
load("//cuda/private:rules/common.bzl", "ALLOW_CUDA_SRCS")

def compile(
        ctx,
        cuda_toolchain,
        cc_toolchain,
        srcs,
        common,
        pic = False,
        rdc = False):
    ""
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
            ctx.attr._default_cuda_archs[CudaArchsInfo],
            source_file = src.path,
            output_file = obj_file.path,
            host_compiler = host_compiler,
            user_compile_flags = common.compile_flags,
            include_paths = common.includes,
            quote_include_paths = common.quote_includes,
            system_include_paths = common.system_includes,
            defines = common.local_defines + common.defines,
            host_defines = common.host_local_defines + common.host_defines,
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
            inputs = depset([src], transitive = [common.headers, cc_toolchain.all_files]),
            env = env,
            mnemonic = "CudaCompile",
            progress_message = "Compiling %s" % src.path,
        )
    return ret
