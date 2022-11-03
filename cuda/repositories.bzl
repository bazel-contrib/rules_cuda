load("//cuda/private:repositories.bzl", _rules_cuda_dependencies = "rules_cuda_dependencies")
load("//cuda/private:toolchain.bzl", _register_detected_cuda_toolchains = "register_detected_cuda_toolchains")

rules_cuda_dependencies = _rules_cuda_dependencies
register_detected_cuda_toolchains = _register_detected_cuda_toolchains
