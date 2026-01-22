def nvcc_version_ge(ctx, major, minor):
    if ctx.attr.toolchain_identifier != "nvcc":
        return False

    return (ctx.attr.nvcc_version_major, ctx.attr.nvcc_version_minor) >= (major, minor)
