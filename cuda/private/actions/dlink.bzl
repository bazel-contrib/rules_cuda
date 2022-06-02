""

load("//cuda/private:action_names.bzl", "ACTION_NAMES")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")

def device_link(
        ctx,
        cuda_toolchain,
        cc_toolchain,
        objects,
        common,
        pic = False,
        rdc = False,
        dlto = False):
    """perform device link, return a dlink-ed object file"""
    if not rdc:
        fail("device link is only meaningful on building relocatable device code")

    actions = ctx.actions
    host_compiler = cc_toolchain.compiler_executable
    cuda_compiler = cuda_toolchain.compiler_executable

    cuda_feature_config = cuda_helper.configure_features(ctx, cuda_toolchain, requested_features = [ACTION_NAMES.device_link])
    artifact_category_name = cuda_helper.get_artifact_category_from_action(ACTION_NAMES.device_link, pic, rdc)
    basename = ctx.attr.name + "_dlink"
    filename = cuda_helper.get_artifact_name(cuda_toolchain, artifact_category_name, basename)

    obj_file = actions.declare_file("_objs/{}/{}".format(ctx.attr.name, filename))

    var = cuda_helper.create_device_link_variables(
        cuda_toolchain,
        cuda_feature_config,
        common.cuda_archs_info,
        output_file = obj_file.path,
        host_compiler = host_compiler,
        host_compile_flags = common.host_compile_flags,
        user_link_flags = common.link_flags,
        use_pic = pic,
    )
    cmd = cuda_helper.get_command_line(cuda_feature_config, ACTION_NAMES.device_link, var)
    env = cuda_helper.get_environment_variables(cuda_feature_config, ACTION_NAMES.device_link, var)
    args = actions.args()
    args.add_all(cmd)
    args.add_all(objects)

    actions.run(
        executable = cuda_compiler,
        arguments = [args],
        outputs = [obj_file],
        inputs = depset(transitive = [objects, cc_toolchain.all_files]),
        env = env,
        mnemonic = "CudaDeviceLink",
        progress_message = "Device linking %{output}",
    )
    return obj_file
