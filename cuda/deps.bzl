load("//cuda/private:repositories.bzl", _rules_cuda_deps = "rules_cuda_deps")
load("//cuda/private:cuda_toolkit.bzl", _detect_cuda_toolkit = "detect_cuda_toolkit")

rules_cuda_deps = _rules_cuda_deps
detect_cuda_toolkit = _detect_cuda_toolkit
