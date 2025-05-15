"""NVCC toolchain utilities"""

def nvcc_version_ge(ctx, major, minor):
    """Check if `ctx` represents a `nvcc` toolchain that is greater than the given constraints.

    Args:
        ctx (ctx): The rule's context object
        major (int): The major version to compare against
        minor (int): The minor version to compare against

    Returns:
        True: If `ctx` is a greater version than what's given.
    """
    if ctx.attr.toolchain_identifier != "nvcc":
        return False
    if ctx.attr.nvcc_version_major < major:
        return False
    if ctx.attr.nvcc_version_minor < minor:
        return False
    return True
