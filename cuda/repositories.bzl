load(
    "//cuda/private:repositories.bzl",
    _cuda_component = "cuda_component",
    _cuda_redist_json = "cuda_redist_json",
    _default_components_mapping = "default_components_mapping",
    _local_cuda = "local_cuda",
    _rules_cuda_dependencies = "rules_cuda_dependencies",
    _rules_cuda_toolchains = "rules_cuda_toolchains",
)
load("//cuda/private:toolchain.bzl", _register_detected_cuda_toolchains = "register_detected_cuda_toolchains")

# rules
cuda_component = _cuda_component
cuda_redist_json = _cuda_redist_json
local_cuda = _local_cuda

# macros
rules_cuda_dependencies = _rules_cuda_dependencies
rules_cuda_toolchains = _rules_cuda_toolchains
register_detected_cuda_toolchains = _register_detected_cuda_toolchains
default_components_mapping = _default_components_mapping
