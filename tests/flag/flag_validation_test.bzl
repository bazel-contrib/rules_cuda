load("@bazel_skylib//lib:unittest.bzl", "analysistest", "asserts")

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

    def has_flag(cmd, single_flag):
        if (" " + single_flag + " ") in cmd:
            return True
        if cmd.endswith(" " + single_flag):
            return True
        return False

    has_matched_action = False
    has_name_match = True
    for action in actions:
        if ctx.attr.action_mnemonic == action.mnemonic:
            if ctx.attr.output_name != "":
                has_name_match = False
                for output in action.outputs.to_list():
                    has_name_match = has_name_match or output.basename == ctx.attr.output_name
                if not has_name_match:
                    continue

            has_matched_action = True
            cmd = " ".join(action.argv)
            for flag in ctx.attr.contain_flags:
                asserts.true(env, has_flag(cmd, flag), 'flag "{}" not in command line "{}"'.format(flag, cmd))
            for flag in ctx.attr.not_contain_flags:
                asserts.true(env, not has_flag(cmd, flag), 'flag "{}" in command line "{}"'.format(flag, cmd))

    msg = "" if has_name_match else ' has output named "{}"'.format(ctx.attr.output_name)
    asserts.true(env, has_matched_action, 'target "{}" do not have action with mnemonic "{}"'.format(
        str(target_under_test),
        ctx.attr.action_mnemonic,
    ) + msg)

    return analysistest.end(env)

def _rules_cuda_target(target):
    # https://github.com/bazelbuild/bazel/issues/19286#issuecomment-1684325913
    # must only apply to rules_cuda related labels when bzlmod is enabled
    is_bzlmod_enabled = str(Label("//:invalid")).startswith("@@")
    label_str = "@//" + target
    if is_bzlmod_enabled:
        return str(Label(label_str))
    else:
        return label_str

def _create_cuda_library_flag_test(*config_settings):
    merged_config_settings = {}
    for cs in config_settings:
        for k, v in cs.items():
            merged_config_settings[k] = v
    return analysistest.make(
        cuda_library_flag_test_impl,
        config_settings = merged_config_settings,
        attrs = {
            "action_mnemonic": attr.string(mandatory = True),
            "output_name": attr.string(),
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

static_link_msvcrt = {"//command_line_option:features": ["static_link_msvcrt"]}

cuda_library_c_dbg_static_msvcrt_flag_test = _create_cuda_library_flag_test(config_settings_dbg, static_link_msvcrt)
cuda_library_c_fastbuild_static_msvcrt_flag_test = _create_cuda_library_flag_test(config_settings_fastbuild, static_link_msvcrt)
cuda_library_c_opt_static_msvcrt_flag_test = _create_cuda_library_flag_test(config_settings_opt, static_link_msvcrt)

# NOTE: @rules_cuda//cuda:archs does not work
config_settings_sm61 = {_rules_cuda_target("cuda:archs"): "sm_61"}
config_settings_compute60 = {_rules_cuda_target("cuda:archs"): "compute_60"}
config_settings_compute60_sm61 = {_rules_cuda_target("cuda:archs"): "compute_60,sm_61"}
config_settings_compute61_sm61 = {_rules_cuda_target("cuda:archs"): "compute_61,sm_61"}
config_settings_sm90a = {_rules_cuda_target("cuda:archs"): "sm_90a"}
config_settings_sm90a_sm90 = {_rules_cuda_target("cuda:archs"): "sm_90a,sm_90"}

cuda_library_sm61_flag_test = _create_cuda_library_flag_test(config_settings_sm61)
cuda_library_sm90a_flag_test = _create_cuda_library_flag_test(config_settings_sm90a)
cuda_library_sm90a_sm90_flag_test = _create_cuda_library_flag_test(config_settings_sm90a_sm90)
cuda_library_compute60_flag_test = _create_cuda_library_flag_test(config_settings_compute60)
cuda_library_compute60_sm61_flag_test = _create_cuda_library_flag_test(config_settings_compute60_sm61)
cuda_library_compute61_sm61_flag_test = _create_cuda_library_flag_test(config_settings_compute61_sm61)
