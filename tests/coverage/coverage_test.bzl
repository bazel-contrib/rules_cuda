"""Analysis tests for coverage support in cuda_library and cuda_objects."""

load("@bazel_skylib//lib:unittest.bzl", "analysistest", "asserts")

_COVERAGE = {
    "//command_line_option:collect_code_coverage": True,
    "//command_line_option:instrumentation_filter": "/tests/coverage[/:]",
}

def _compile_actions(env):
    return [a for a in analysistest.target_actions(env) if a.mnemonic == "CudaCompile"]

def _keep_outputs(action):
    return [f for f in action.outputs.to_list() if f.basename.endswith(".keep")]

def _assert_keeps(env, action):
    asserts.true(env, "--keep" in action.argv, "--keep missing from {}".format(action.argv))
    asserts.true(env, "--keep-dir" in action.argv, "--keep-dir missing from {}".format(action.argv))
    asserts.false(env, "--objdir-as-tempdir" in action.argv, "--objdir-as-tempdir overrides --keep-dir in {}".format(action.argv))
    keep_dirs = _keep_outputs(action)
    asserts.equals(env, 1, len(keep_dirs))
    if keep_dirs:
        asserts.equals(env, keep_dirs[0].path, action.argv[action.argv.index("--keep-dir") + 1])

def _assert_not_keeps(env, action):
    asserts.false(env, "--keep" in action.argv, "unexpected --keep in {}".format(action.argv))
    asserts.equals(env, [], _keep_outputs(action))

def _assert_instrumented_files(env, target, keep_dirs):
    info = target[InstrumentedFilesInfo]
    asserts.equals(
        env,
        sorted(["kernel.cu", "kernel.cuh"]),
        sorted([f.basename for f in info.instrumented_files.to_list()]),
    )
    asserts.equals(
        env,
        sorted([f.path for f in keep_dirs]),
        sorted([f.path for f in info.metadata_files.to_list()]),
    )

def _coverage_library_test_impl(ctx):
    env = analysistest.begin(ctx)
    actions = _compile_actions(env)
    asserts.true(env, len(actions) > 0, "no CudaCompile actions")
    keep_dirs = []
    for action in actions:
        _assert_keeps(env, action)
        keep_dirs.extend(_keep_outputs(action))
    _assert_instrumented_files(env, analysistest.target_under_test(env), keep_dirs)
    return analysistest.end(env)

coverage_library_test = analysistest.make(
    _coverage_library_test_impl,
    config_settings = _COVERAGE,
)

def _coverage_objects_test_impl(ctx):
    env = analysistest.begin(ctx)
    actions = _compile_actions(env)
    rdc_count = len([a for a in actions if "-rdc=true" in a.argv])
    asserts.true(env, rdc_count > 0, "no rdc CudaCompile actions")
    asserts.true(env, rdc_count < len(actions), "no non-rdc CudaCompile actions")
    keep_dirs = []
    for action in actions:
        if "-rdc=true" in action.argv:
            _assert_keeps(env, action)
            keep_dirs.extend(_keep_outputs(action))
        else:
            _assert_not_keeps(env, action)
    _assert_instrumented_files(env, analysistest.target_under_test(env), keep_dirs)
    return analysistest.end(env)

coverage_objects_test = analysistest.make(
    _coverage_objects_test_impl,
    config_settings = _COVERAGE,
)

def _coverage_disabled_test_impl(ctx):
    env = analysistest.begin(ctx)
    actions = _compile_actions(env)
    asserts.true(env, len(actions) > 0, "no CudaCompile actions")
    for action in actions:
        _assert_not_keeps(env, action)
    return analysistest.end(env)

coverage_disabled_test = analysistest.make(_coverage_disabled_test_impl)
