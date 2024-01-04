def _disabled_toolchain_config_impl(_ctx):
    return [platform_common.ToolchainInfo()]

disabled_toolchain_config = rule(_disabled_toolchain_config_impl, attrs = {})
