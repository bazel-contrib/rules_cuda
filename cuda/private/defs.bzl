"""private defs"""

def requires_cuda():
    """Returns constraint_setting that is satisfied if:

    * CUDA is enabled and
    * CUDA toolchain is found.

    Add to 'target_compatible_with' attribute to mark a target incompatible when
    @rules_cuda//cuda:is_enabled_and_cuda_found is not set. Incompatible targets are excluded
    from bazel target wildcards and fail to build if requested explicitly.
    """
    return select({
        "@rules_cuda//cuda:is_enabled_and_cuda_found": [],
        "//conditions:default": ["@platforms//:incompatible"],
    })
