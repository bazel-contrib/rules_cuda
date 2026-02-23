def _unsupported_cuda_version_impl(ctx):
    fail("CUDA component '{}' is not available for the selected CUDA version. Available versions: {}".format(
        ctx.attr.component,
        ", ".join(ctx.attr.available_versions),
    ))

unsupported_cuda_version = rule(
    implementation = _unsupported_cuda_version_impl,
    attrs = {
        "component": attr.string(mandatory = True),
        "available_versions": attr.string_list(mandatory = True),
    },
)

def _unsupported_cuda_platform_impl(ctx):
    fail("CUDA component '{}' is not available for the selected platform. Available platforms: {}".format(
        ctx.attr.component,
        ", ".join(ctx.attr.available_platforms),
    ))

unsupported_cuda_platform = rule(
    implementation = _unsupported_cuda_platform_impl,
    attrs = {
        "component": attr.string(mandatory = True),
        "available_platforms": attr.string_list(mandatory = True),
    },
)
