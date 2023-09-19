load("//cuda/private:rules/cuda_library.bzl", _cuda_library = "cuda_library")

def cuda_test(**attrs):
    """Wrapper to ensure the test is compiled with the CUDA compiler."""
    cuda_library_name = "_" + getattr(attrs, "name", "")

    _cuda_library(
        name = cuda_library_name,
        srcs = getattr(attrs, "srcs", []),
        copts = getattr(attrs, "copts", []),
        deps = getattr(attrs, "deps", []),
        testonly = True,
    )

    native.cc_test(
        deps = [cuda_library_name],
        **attrs
    )
