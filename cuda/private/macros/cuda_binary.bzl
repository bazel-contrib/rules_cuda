load("//cuda/private:rules/cuda_library.bzl", _cuda_library = "cuda_library")

def cuda_binary(name, **attrs):
    """A macro wraps cuda_library and cc_binary to ensure the binary is compiled with the CUDA compiler.

    Args:
        name: A unique name for this target (cc_binary).
        **attrs: attrs of cc_binary and cuda_library.
    """
    cuda_library_only_attrs = ["deps", "srcs", "hdrs", "alwayslink", "rdc"]
    cuda_library_only_attrs_defaults = {
        "alwayslink": True,
    }

    # https://bazel.build/reference/be/common-definitions?hl=en#common-attributes-binaries
    cc_binary_only_attrs = ["args", "env", "output_licenses"]

    cuda_library_attrs = {k: v for k, v in attrs.items() if k not in cc_binary_only_attrs}
    for attr in cuda_library_only_attrs_defaults:
        if attr not in cuda_library_attrs:
            cuda_library_attrs[attr] = cuda_library_only_attrs_defaults[attr]

    cuda_library_name = "_" + name
    _cuda_library(
        name = cuda_library_name,
        **cuda_library_attrs
    )

    native.cc_binary(
        name = name,
        deps = [cuda_library_name],
        **{k: v for k, v in attrs.items() if k not in cuda_library_only_attrs}
    )
