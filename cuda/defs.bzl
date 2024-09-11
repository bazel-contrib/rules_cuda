"""
Core rules for building CUDA projects.
"""

load("//cuda/private:defs.bzl", _requires_cuda = "requires_cuda")
load("//cuda/private:macros/cuda_binary.bzl", _cuda_binary = "cuda_binary")
load("//cuda/private:macros/cuda_test.bzl", _cuda_test = "cuda_test")
load("//cuda/private:os_helpers.bzl", _cc_import_versioned_sos = "cc_import_versioned_sos", _if_linux = "if_linux", _if_windows = "if_windows")
load("//cuda/private:providers.bzl", _CudaArchsInfo = "CudaArchsInfo", _cuda_archs = "cuda_archs")
load("//cuda/private:rules/cuda_library.bzl", _cuda_library = "cuda_library")
load("//cuda/private:rules/cuda_objects.bzl", _cuda_objects = "cuda_objects")
load("//cuda/private:rules/cuda_toolkit.bzl", _cuda_toolkit = "cuda_toolkit")
load(
    "//cuda/private:toolchain.bzl",
    _cuda_toolchain = "cuda_toolchain",
    _find_cuda_toolchain = "find_cuda_toolchain",
    _use_cuda_toolchain = "use_cuda_toolchain",
)
load("//cuda/private:toolchain_configs/clang.bzl", _cuda_toolchain_config_clang = "cuda_toolchain_config")
load("//cuda/private:toolchain_configs/disabled.bzl", _cuda_toolchain_config_disabled = "disabled_toolchain_config")
load("//cuda/private:toolchain_configs/nvcc.bzl", _cuda_toolchain_config_nvcc = "cuda_toolchain_config")
load("//cuda/private:toolchain_configs/nvcc_msvc.bzl", _cuda_toolchain_config_nvcc_msvc = "cuda_toolchain_config")

cuda_toolkit = _cuda_toolkit
cuda_toolchain = _cuda_toolchain
find_cuda_toolchain = _find_cuda_toolchain
use_cuda_toolchain = _use_cuda_toolchain
cuda_toolchain_config_clang = _cuda_toolchain_config_clang
cuda_toolchain_config_disabled = _cuda_toolchain_config_disabled
cuda_toolchain_config_nvcc_msvc = _cuda_toolchain_config_nvcc_msvc
cuda_toolchain_config_nvcc = _cuda_toolchain_config_nvcc

cuda_archs = _cuda_archs
CudaArchsInfo = _CudaArchsInfo

# rules
cuda_objects = _cuda_objects
cuda_library = _cuda_library

# macros
cuda_binary = _cuda_binary
cuda_test = _cuda_test

if_linux = _if_linux
if_windows = _if_windows

cc_import_versioned_sos = _cc_import_versioned_sos

requires_cuda = _requires_cuda
