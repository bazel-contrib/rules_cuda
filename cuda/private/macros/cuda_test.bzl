load("//cuda/private:rules/cuda_library.bzl", _cuda_library = "cuda_library")

def cuda_test(name, **attrs):
    """Wrapper to ensure the test is compiled with the CUDA compiler."""
    cuda_library_name = "_" + name

    _cuda_library(
        name = cuda_library_name,
        testonly = True,
        **attrs
    )

    native.cc_test(
        name = name,
        deps = [cuda_library_name],
        **{k: v for k, v in attrs.items() if k != "deps"}
    )
