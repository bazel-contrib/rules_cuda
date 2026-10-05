# CUDA repository hierarchy

| Configuration                   | `@cuda//toolchain`                                  | Implementations                      |
| ------------------------------- | --------------------------------------------------- | ------------------------------------ |
| Local CTK                       | Implementations and registrations                   | Inside `@cuda`                       |
| One `redist_json` release       | Versioned and default registrations                 | One `@cuda_toolchain_<version>` repo |
| Multiple `redist_json` releases | Registrations for every release; highest is default | One repo per release                 |

The [template README](https://github.com/bazel-contrib/rules_cuda/blob/main/cuda/private/templates/README.md)
shows the complete generated layouts. Redistribution facades have no `defs.bzl`;
version helpers belong to the component and implementation repositories.

## Component selection

For releases 12.8.1 and 13.0.1 declared on Linux x86_64 and SBSA, the component
aliases resolve as follows. Intermediate execution-platform and ARM aliases are
omitted.

```mermaid
flowchart TD
    cudanvcc["@cuda//:nvcc<br/>alias to @cuda_nvcc//:nvcc"]
    cudanvcc_nvcc["@cuda_nvcc//:nvcc<br/>platform and version aliases"]
    x64["@cuda_nvcc//:linux_x86_64_nvcc"]
    arm["@cuda_nvcc//:linux_sbsa_nvcc"]
    x64_win["@rules_cuda//cuda/dummy:dummy<br/>Windows not declared"]
    x64_12["@cuda_nvcc_linux_x86_64_12_8_1//:nvcc"]
    x64_13["@cuda_nvcc_linux_x86_64_13_0_1//:nvcc"]
    arm_12["@cuda_nvcc_linux_sbsa_12_8_1//:nvcc"]
    arm_13["@cuda_nvcc_linux_sbsa_13_0_1//:nvcc"]

    cudanvcc --> cudanvcc_nvcc
    cudanvcc_nvcc -- "linux-x86_64" --> x64
    cudanvcc_nvcc -- "linux-sbsa" --> arm
    cudanvcc_nvcc -- "windows-x86_64" --> x64_win
    x64 -- "version_is_nvcc_12_8_1" --> x64_12
    x64 -- "version_is_nvcc_13_0_1 or default" --> x64_13
    arm -- "version_is_nvcc_12_8_1" --> arm_12
    arm -- "version_is_nvcc_13_0_1 or default" --> arm_13
```

`nvcc` and `nvvm` follow the execution platform; libraries follow the target
platform. A release-specific alias such as `@cuda_nvcc_toolchain_12_8_1` uses the
same platform selection but points only to 12.8.1 component repositories.

## Toolchain selection

Generated BUILD excerpts (unrelated attributes omitted):

```starlark
# @cuda//toolchain/BUILD
config_setting(
    name = "toolkit_version_is_12_8_1",
    flag_values = {"@rules_cuda//cuda:version": "12.8.1"},
)
toolchain(
    name = "nvcc-linux-12_8_1-toolchain",
    target_settings = [
        "@rules_cuda//cuda:is_enabled",
        "@rules_cuda//cuda:compiler_is_nvcc",
        ":toolkit_version_is_12_8_1",
    ],
    toolchain = "@cuda_toolchain_12_8_1//toolchain:nvcc-linux",
    toolchain_type = "@rules_cuda//cuda:toolchain_type",
    # platform constraints and visibility omitted
)

# @cuda_toolchain_12_8_1//toolchain/BUILD
alias(name = "nvcc-linux", actual = "//toolchain/nvcc:nvcc-local", visibility = ["//visibility:private"])

# @cuda_toolchain_12_8_1//toolchain/nvcc/BUILD
cuda_toolchain(
    name = "nvcc-local",
    compiler_label = "@cuda_nvcc_toolchain_12_8_1//:nvcc",
    toolchain_config = ":nvcc-local-config",
    # compiler_files omitted
)
```

With `--@rules_cuda//cuda:version=12.8.1 --@rules_cuda//cuda:compiler=nvcc`,
**analysis** selects this toolchain and resolves its compiler label through the
platform aliases. **Execution** runs that compiler; selection does not happen at
runtime. An unset version selects the highest declared release; an undeclared
version fails toolchain resolution when CUDA is enabled.
