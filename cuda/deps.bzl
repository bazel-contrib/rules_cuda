load("//cuda/private:repositories.bzl", _rules_cuda_deps = "rules_cuda_deps")
load("//cuda/private:cuda_toolkit.bzl", _register_detected_cuda_toolchains = "register_detected_cuda_toolchains")

rules_cuda_deps = _rules_cuda_deps
register_detected_cuda_toolchains = _register_detected_cuda_toolchains
