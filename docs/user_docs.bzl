load("@rules_cuda//cuda:defs.bzl", _cuda_library = "cuda_library", _cuda_objects = "cuda_objects")
load("@rules_cuda//cuda:repositories.bzl", _register_detected_cuda_toolchains = "register_detected_cuda_toolchains", _rules_cuda_dependencies = "rules_cuda_dependencies")
load("@rules_cuda//cuda/private:rules/flags.bzl", _cuda_archs_flag = "cuda_archs_flag")

cuda_library = _cuda_library
cuda_objects = _cuda_objects

cuda_archs = _cuda_archs_flag

register_detected_cuda_toolchains = _register_detected_cuda_toolchains
rules_cuda_dependencies = _rules_cuda_dependencies
