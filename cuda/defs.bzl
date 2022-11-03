"""
Core rules for building CUDA projects.
"""

load("//cuda/private:providers.bzl", _CudaArchsInfo = "CudaArchsInfo", _cuda_archs = "cuda_archs")
load("//cuda/private:rules/cuda_objects.bzl", _cuda_objects = "cuda_objects")
load("//cuda/private:rules/cuda_library.bzl", _cuda_library = "cuda_library")
load("//cuda/private:rules/cuda_toolkit.bzl", _cuda_toolkit = "cuda_toolkit")
load(
    "//cuda/private:toolchain.bzl",
    _cuda_toolchain = "cuda_toolchain",
    _find_cuda_toolchain = "find_cuda_toolchain",
    _use_cuda_toolchain = "use_cuda_toolchain",
)
load("//cuda/private:toolchain_configs/clang.bzl", _cuda_toolchain_config_clang = "cuda_toolchain_config")
load("//cuda/private:toolchain_configs/nvcc.bzl", _cuda_toolchain_config_nvcc = "cuda_toolchain_config")
load("//cuda/private:toolchain_configs/nvcc_msvc.bzl", _cuda_toolchain_config_nvcc_msvc = "cuda_toolchain_config")

cuda_toolkit = _cuda_toolkit
cuda_toolchain = _cuda_toolchain
find_cuda_toolchain = _find_cuda_toolchain
use_cuda_toolchain = _use_cuda_toolchain
cuda_toolchain_config_clang = _cuda_toolchain_config_clang
cuda_toolchain_config_nvcc_msvc = _cuda_toolchain_config_nvcc_msvc
cuda_toolchain_config_nvcc = _cuda_toolchain_config_nvcc

cuda_archs = _cuda_archs
CudaArchsInfo = _CudaArchsInfo

cuda_objects = _cuda_objects
cuda_library = _cuda_library
