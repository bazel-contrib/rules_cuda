load("@bazel_skylib//lib:partial.bzl", "partial")
load("@bazel_skylib//lib:unittest.bzl", "analysistest", "asserts", "unittest")
load("//cuda/private:providers.bzl", "CudaArchsInfo")
load("//cuda/private:cuda_helper.bzl", "cuda_helper")

def _num_actions_test_impl(ctx):
    env = analysistest.begin(ctx)
    target_under_test = analysistest.target_under_test(env)
    actions = analysistest.target_actions(env)
    if ctx.attr.num_actions > 0:
        asserts.equals(env, ctx.attr.num_actions, len(actions))
    return analysistest.end(env)

num_actions_test = analysistest.make(
    _num_actions_test_impl,
    attrs = {
        "num_actions": attr.int(),
    },
)

def cuda_library_flag_test_impl(ctx):
    env = analysistest.begin(ctx)
    target_under_test = analysistest.target_under_test(env)
    actions = analysistest.target_actions(env)

    asserts.true(env, len(ctx.attr.contain_flags) + len(ctx.attr.not_contain_flags) > 0, "Invalid test config")

    has_matched_action = False
    for action in actions:
        if ctx.attr.action_mnemonic == action.mnemonic:
            has_matched_action = True
            cmd = " ".join(action.argv)
            for flag in ctx.attr.contain_flags:
                asserts.true(env, (" " + flag + " ") in cmd, 'flag "{}" not in command line "{}"'.format(flag, cmd))
            for flag in ctx.attr.not_contain_flags:
                asserts.true(env, (" " + flag + " ") not in cmd, 'flag "{}" in command line "{}"'.format(flag, cmd))

    asserts.true(env, has_matched_action, 'target "{}" do not have action with mnemonic "{}"'.format(
        str(target_under_test),
        ctx.attr.action_mnemonic,
    ))

    return analysistest.end(env)

def _create_cuda_library_flag_test(config_settings):
    return analysistest.make(
        cuda_library_flag_test_impl,
        config_settings = config_settings,
        attrs = {
            "action_mnemonic": attr.string(mandatory = True),
            "contain_flags": attr.string_list(),
            "not_contain_flags": attr.string_list(),
        },
    )


cuda_library_flag_test = _create_cuda_library_flag_test({})

config_settings_dbg = {"//command_line_option:compilation_mode": "dbg"}
config_settings_fastbuild = {"//command_line_option:compilation_mode": "fastbuild"}
config_settings_opt = {"//command_line_option:compilation_mode": "opt"}

cuda_library_c_dbg_flag_test = _create_cuda_library_flag_test(config_settings_dbg)
cuda_library_c_fastbuild_flag_test = _create_cuda_library_flag_test(config_settings_fastbuild)
cuda_library_c_opt_flag_test = _create_cuda_library_flag_test(config_settings_opt)

# NOTE: @rules_cuda//cuda:archs does not work
config_settings_sm61 = {"@//cuda:archs": "sm_61"}
config_settings_compute60 = {"@//cuda:archs": "compute_60"}
config_settings_compute60_sm61 = {"@//cuda:archs": "compute_60,sm_61"}
config_settings_compute61_sm61 = {"@//cuda:archs": "compute_61,sm_61"}

cuda_library_sm61_flag_test = _create_cuda_library_flag_test(config_settings_sm61)
cuda_library_compute60_flag_test = _create_cuda_library_flag_test(config_settings_compute60)
cuda_library_compute60_sm61_flag_test = _create_cuda_library_flag_test(config_settings_compute60_sm61)
cuda_library_compute61_sm61_flag_test = _create_cuda_library_flag_test(config_settings_compute61_sm61)
