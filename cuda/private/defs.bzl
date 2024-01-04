"""private"""

def _requires_is_enabled():
    return select({
        "@rules_cuda//cuda:is_enabled": [],
        "//conditions:default": ["@rules_cuda//cuda:cuda_must_be_enabled"],
    })

def _requires_cuda_found():
    return select({
        "@rules_cuda//cuda:is_cuda_found": [],
        "//conditions:default": ["@rules_cuda//cuda:cuda_must_be_found"],
    })

def requires_cuda():
    """Returns constraint_setting that is satisfied if:

    * CUDA is enabled and
    * CUDA toolchain is found.

    Add to 'target_compatible_with' attribute to mark a target incompatible when
    the conditions are not satisfied. Incompatible targets are excluded
    from bazel target wildcards and fail to build if requested explicitly.
    """
    return _requires_is_enabled() + _requires_cuda_found()
