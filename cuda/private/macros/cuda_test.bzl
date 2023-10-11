load("//cuda/private:rules/cuda_library.bzl", _cuda_library = "cuda_library")

def cuda_test(name, **attrs):
    """Wrapper to ensure the test is compiled with the CUDA compiler."""
    cuda_library_only_attrs = ["deps"]

    # https://bazel.build/reference/be/common-definitions?hl=en#common-attributes-tests
    cc_test_only_attrs = ["args", "env", "env_inherit", "size", "timeout", "flaky", "shard_count", "local"]

    cuda_library_name = "_" + name

    _cuda_library(
        name = cuda_library_name,
        testonly = True,
        **{k: v for k, v in attrs.items() if k not in cc_test_only_attrs}
    )

    native.cc_test(
        name = name,
        deps = [cuda_library_name],
        **{k: v for k, v in attrs.items() if k not in cuda_library_only_attrs}
    )
