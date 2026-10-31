"""Analysis-test factories configured for the clang CUDA toolchain."""

load("//tests/flag:flag_validation_test.bzl", "create_cuda_library_flag_test")

def _rules_cuda_target(target):
    # https://github.com/bazelbuild/bazel/issues/19286#issuecomment-1684325913
    # Only canonicalize rules_cuda labels when bzlmod is enabled.
    is_bzlmod_enabled = str(Label("//:invalid")).startswith("@@")
    label_str = "@//" + target
    if is_bzlmod_enabled:
        return str(Label(label_str))
    return label_str

config_settings_clang = {_rules_cuda_target("cuda:compiler"): "clang"}
config_settings_dbg = {"//command_line_option:compilation_mode": "dbg"}
config_settings_fastbuild = {"//command_line_option:compilation_mode": "fastbuild"}
config_settings_opt = {"//command_line_option:compilation_mode": "opt"}

clang_cuda_library_flag_test = create_cuda_library_flag_test(config_settings_clang)
clang_cuda_library_c_dbg_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_dbg)
clang_cuda_library_c_fastbuild_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_fastbuild)
clang_cuda_library_c_opt_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_opt)

config_settings_sm61 = {_rules_cuda_target("cuda:archs"): "sm_61"}
config_settings_compute60 = {_rules_cuda_target("cuda:archs"): "compute_60"}
config_settings_compute60_sm61 = {_rules_cuda_target("cuda:archs"): "compute_60,sm_61"}
config_settings_sm90a = {_rules_cuda_target("cuda:archs"): "sm_90a"}
config_settings_sm90a_sm90 = {_rules_cuda_target("cuda:archs"): "sm_90a,sm_90"}
config_settings_sm100_sm100a = {_rules_cuda_target("cuda:archs"): "sm_100;sm_100a"}
config_settings_sm110_sm110a_sm110f = {_rules_cuda_target("cuda:archs"): "compute_110:sm_110,sm_110a,sm_110f"}

clang_cuda_library_sm61_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_sm61)
clang_cuda_library_compute60_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_compute60)
clang_cuda_library_compute60_sm61_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_compute60_sm61)
clang_cuda_library_sm90a_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_sm90a)
clang_cuda_library_sm90a_sm90_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_sm90a_sm90)
clang_cuda_library_sm100_sm100a_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_sm100_sm100a)
clang_cuda_library_sm110_sm110a_sm110f_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_sm110_sm110a_sm110f)

config_settings_use_copts = {
    "//command_line_option:features": ["cuda_host_use_copts"],
    "//command_line_option:copt": ["-DRULES_CUDA_TEST_COPT_PROBE"],
}
config_settings_use_cxxopts = {
    "//command_line_option:features": ["cuda_host_use_cxxopts"],
    "//command_line_option:cxxopt": ["-DRULES_CUDA_TEST_CXXOPT_PROBE"],
}
config_settings_use_linkopts = {
    "//command_line_option:features": ["cuda_host_use_linkopts"],
    "//command_line_option:linkopt": ["-DRULES_CUDA_TEST_LINKOPT_PROBE"],
}
config_settings_copts_without_feature = {
    "//command_line_option:features": ["-cuda_host_use_copts"],
    "//command_line_option:copt": ["-DRULES_CUDA_TEST_COPT_PROBE"],
}
config_settings_cxxopts_without_feature = {
    "//command_line_option:features": ["-cuda_host_use_cxxopts"],
    "//command_line_option:cxxopt": ["-DRULES_CUDA_TEST_CXXOPT_PROBE"],
}
config_settings_linkopts_without_feature = {
    "//command_line_option:features": ["-cuda_host_use_linkopts"],
    "//command_line_option:linkopt": ["-DRULES_CUDA_TEST_LINKOPT_PROBE"],
}

clang_cuda_library_use_copts_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_use_copts)
clang_cuda_library_use_cxxopts_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_use_cxxopts)
clang_cuda_library_use_linkopts_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_use_linkopts)
clang_cuda_library_copts_without_feature_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_copts_without_feature)
clang_cuda_library_cxxopts_without_feature_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_cxxopts_without_feature)
clang_cuda_library_linkopts_without_feature_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_linkopts_without_feature)

config_settings_external_include_paths = {
    "//command_line_option:features": ["external_include_paths"],
}
clang_cuda_library_external_include_paths_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_external_include_paths)

config_settings_toolchain_resolution = {"//command_line_option:incompatible_enable_cc_toolchain_resolution": "1"}
config_settings_platform_sysroot_test = {"//command_line_option:platforms": _rules_cuda_target("tests/flag/testonly_toolchains:sysroot-test-platform")}
config_settings_platform_no_sysroot_test = {"//command_line_option:platforms": _rules_cuda_target("tests/flag/testonly_toolchains:no-sysroot-test-platform")}
config_settings_platform_copt_sysroot_test = {"//command_line_option:copt": ["--sysroot=/sysroot/from/copt"]}

clang_cuda_library_platform_sysroot_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_platform_sysroot_test, config_settings_toolchain_resolution)
clang_cuda_library_platform_no_sysroot_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_platform_no_sysroot_test, config_settings_toolchain_resolution)
clang_cuda_library_platform_no_sysroot_but_copt_flag_test = create_cuda_library_flag_test(config_settings_clang, config_settings_platform_no_sysroot_test, config_settings_platform_copt_sysroot_test)
