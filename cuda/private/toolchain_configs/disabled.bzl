"""disabled_toolchain_config implementation"""

def _disabled_toolchain_config_impl(_ctx):
    return [platform_common.ToolchainInfo()]

disabled_toolchain_config = rule(
    doc = "A variant of `cuda_toolchain_configuration` rules that represents a disabled toolchain.",
    implementation = _disabled_toolchain_config_impl,
    attrs = {},
)
