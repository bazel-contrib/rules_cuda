load(
    "@rules_cuda//cuda:defs.bzl",
    _cuda_toolchain = "cuda_toolchain",
    _cuda_toolchain_config_clang = "cuda_toolchain_config_clang",
    _cuda_toolchain_config_nvcc = "cuda_toolchain_config_nvcc",
    _cuda_toolchain_config_nvcc_msvc = "cuda_toolchain_config_nvcc_msvc",
    _cuda_toolkit = "cuda_toolkit",
)

cuda_toolkit = _cuda_toolkit
cuda_toolchain = _cuda_toolchain
cuda_toolchain_config_clang = _cuda_toolchain_config_clang
cuda_toolchain_config_nvcc_msvc = _cuda_toolchain_config_nvcc_msvc
cuda_toolchain_config_nvcc = _cuda_toolchain_config_nvcc
