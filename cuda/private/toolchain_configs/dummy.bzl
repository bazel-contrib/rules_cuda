def _dummy_toolchain_config_impl(_ctx):
    return [platform_common.ToolchainInfo()]

dummy_toolchain_config = rule(_dummy_toolchain_config_impl, attrs = {})
