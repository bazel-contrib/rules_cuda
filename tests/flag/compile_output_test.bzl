load("@bazel_skylib//lib:unittest.bzl", "analysistest", "asserts")

def _cuda_compile_output_directory_test_impl(ctx):
    env = analysistest.begin(ctx)
    actions = analysistest.target_actions(env)
    outputs_by_directory = {}
    output_count = 0

    for action in actions:
        if action.mnemonic != "CudaCompile":
            continue
        for output in action.outputs.to_list():
            previous_output = outputs_by_directory.get(output.dirname)
            asserts.equals(
                env,
                None,
                previous_output,
                "CudaCompile outputs '{}' and '{}' share directory '{}'".format(
                    previous_output,
                    output.path,
                    output.dirname,
                ),
            )
            outputs_by_directory[output.dirname] = output.path
            output_count += 1

    asserts.equals(env, 2, output_count)
    return analysistest.end(env)

cuda_compile_output_directory_test = analysistest.make(
    _cuda_compile_output_directory_test_impl,
)
