""

load("//cuda/private:action_names.bzl", "ACTION_NAMES")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")

def create_library(ctx, cuda_toolchain, cc_toolchain, objects, pic = False):
    """create static library"""

    actions = ctx.actions
    host_compiler = cc_toolchain.compiler_executable
    cuda_compiler = cuda_toolchain.compiler_executable

    cuda_feature_config = cuda_helper.configure_features(ctx, cuda_toolchain, requested_features = [ACTION_NAMES.create_library] + ctx.attr.features)
    artifact_category_name = cuda_helper.get_artifact_category_from_action(ACTION_NAMES.create_library, pic)
    basename = ctx.attr.name
    filename = cuda_helper.get_artifact_name(cuda_toolchain, artifact_category_name, basename)

    lib_file = actions.declare_file(filename)

    var = struct(output_file = lib_file.path, host_compiler = host_compiler)
    cmd = cuda_helper.get_command_line(cuda_feature_config, ACTION_NAMES.create_library, var)
    env = cuda_helper.get_environment_variables(cuda_feature_config, ACTION_NAMES.create_library, var)
    args = actions.args()
    args.add_all(cmd)
    args.add_all(objects)

    actions.run(
        executable = cuda_compiler,
        arguments = [args],
        outputs = [lib_file],
        inputs = depset(transitive = [objects, cc_toolchain.all_files]),
        env = env,
        mnemonic = "CreateLibrary",
        progress_message = "Creating library %{output}",
    )
    return lib_file
