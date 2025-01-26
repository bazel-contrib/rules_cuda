load("//cuda/private:rules/cuda_library.bzl", _cuda_library = "cuda_library")

def cuda_test(name, **attrs):
    """A macro wraps cuda_library and cc_test to ensure the test is compiled with the CUDA compiler.

    Args:
        name: A unique name for this target (cc_test).
        **attrs: attrs of cc_test and cuda_library.
    """
    cuda_library_only_attrs = ["deps", "srcs", "hdrs", "testonly", "alwayslink"]
    cuda_library_only_attrs_defaults = {
        "testonly": True,
        "alwayslink": True,
    }

    # https://bazel.build/reference/be/common-definitions?hl=en#common-attributes-tests
    cc_test_only_attrs = ["args", "env", "env_inherit", "size", "timeout", "flaky", "shard_count", "local", "data"]

    cuda_library_attrs = {k: v for k, v in attrs.items() if k not in cc_test_only_attrs}
    for attr in cuda_library_only_attrs_defaults:
        if attr not in cuda_library_attrs:
            cuda_library_attrs[attr] = cuda_library_only_attrs_defaults[attr]

    cuda_library_name = "_" + name
    _cuda_library(
        name = cuda_library_name,
        **cuda_library_attrs
    )

    native.cc_test(
        name = name,
        deps = [cuda_library_name],
        **{k: v for k, v in attrs.items() if k not in cuda_library_only_attrs}
    )
