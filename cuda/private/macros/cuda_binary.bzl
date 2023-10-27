load("//cuda/private:rules/cuda_library.bzl", _cuda_library = "cuda_library")

def cuda_binary(name, **attrs):
    """Wrapper to ensure the binary is compiled with the CUDA compiler."""
    cuda_library_only_attrs = ["deps", "srcs", "hdrs"]

    # https://bazel.build/reference/be/common-definitions?hl=en#common-attributes-binaries
    cc_binary_only_attrs = ["args", "env", "output_licenses"]

    cuda_library_name = "_" + name

    _cuda_library(
        name = cuda_library_name,
        **{k: v for k, v in attrs.items() if k not in cc_binary_only_attrs}
    )

    native.cc_binary(
        name = name,
        deps = [cuda_library_name],
        **{k: v for k, v in attrs.items() if k not in cuda_library_only_attrs}
    )
