load(
    "//cuda/private:repositories.bzl",
    _local_cuda = "local_cuda",
    _rules_cuda_dependencies = "rules_cuda_dependencies",
    _rules_cuda_toolchains = "rules_cuda_toolchains",
)
load("//cuda/private:toolchain.bzl", _register_detected_cuda_toolchains = "register_detected_cuda_toolchains")

local_cuda = _local_cuda
rules_cuda_dependencies = _rules_cuda_dependencies
rules_cuda_toolchains = _rules_cuda_toolchains
register_detected_cuda_toolchains = _register_detected_cuda_toolchains
