## Template files

- `BUILD.cuda_shared`: For `cuda` repo (CTK + toolchain) or `cuda_%{component_name}`
- `BUILD.lctk_cuda`: For `cuda` repo only in locally installed CTK case, expand separately.
- `BUILD.dctk_cuda`: For `cuda` repo only in deliverable CTK case, expand separately.
- `BUILD.dctk_comp`: For component repos only in deliverable CTK case, expand separately.
- `BUILD.cuda_build_setting`: For `cuda` repo (CTK + toolchain) build_setting
- `BUILD.cuda_disabled`: For creating a dummy local configuration.
- `BUILD.toolchain_disabled`: For creating a dummy local toolchain.
- `BUILD.toolchain_clang`: For Clang device compilation toolchain.
- `BUILD.toolchain_nvcc`: For NVCC device compilation toolchain.
- `BUILD.toolchain_nvcc_msvc`: For NVCC device compilation with MSVC as host compiler.
- Otherwise, each `BUILD.*` corresponds to a component in CUDA Toolkit.

## Repository organization

These layouts use `cuda.toolkit(name = "cuda")` on a Linux host. `:name` denotes a
target in the enclosing BUILD file; braces abbreviate multiple names.

### 1. Local CTK

`@cuda` contains the toolkit files, implementations, and registrations directly.

```text
@cuda//
├── cuda/ → local toolkit directory
├── BUILD                         # BUILD.lctk_cuda + component fragments
├── defs.bzl                      # detected toolkit version
└── toolchain/
    ├── BUILD                     # BUILD.toolchain_nvcc (nvcc_msvc on Windows)
    │   ├── :cuda-toolkit
    │   ├── :nvcc-local-config
    │   ├── :nvcc-local
    │   └── :nvcc-local-toolchain
    ├── clang/BUILD
    │   ├── :cuda-toolkit
    │   ├── :clang-local-config
    │   ├── :clang-local
    │   └── :clang-local-toolchain
    └── disabled/BUILD
        ├── :cuda_is_disabled
        ├── :disabled-local
        └── :disabled-local-toolchain
```

### 2. Single deliverable CTK from `redist_json`

Even one release creates a public facade and a separate implementation repository.
For CUDA 12.6.3:

```text
@cuda//
├── BUILD                         # component aliases + BUILD.dctk_cuda
└── toolchain/                    # no defs.bzl in this facade
    ├── BUILD
    │   ├── :toolkit_version_is_{12_6_3,default}
    │   ├── :nvcc-{linux,windows}-12_6_3-toolchain
    │   ├── :nvcc-12_6_3-toolchain → :nvcc-linux-12_6_3-toolchain
    │   ├── :nvcc-{linux,windows}-toolchain
    │   └── :nvcc-local-toolchain → :nvcc-linux-toolchain
    ├── clang/BUILD
    │   ├── :toolkit_version_is_{12_6_3,default}
    │   └── :clang-{12_6_3,local}-toolchain
    └── disabled/BUILD            # same three disabled targets as above

@cuda_toolchain_12_6_3//
├── BUILD                         # component aliases fixed to 12.6.3
├── defs.bzl                      # version = (12, 6)
└── toolchain/
    ├── BUILD                     # aliases into implementation packages
    │   ├── :cuda-toolkit → //toolchain/nvcc:cuda-toolkit
    │   ├── :nvcc-linux{,-config,-toolchain} → //toolchain/nvcc:nvcc-local{,-config,-toolchain}
    │   ├── :nvcc-windows{,-config,-toolchain} → //toolchain/nvcc_msvc:nvcc-local{,-config,-toolchain}
    │   └── :nvcc-local-toolchain → :nvcc-linux-toolchain
    ├── nvcc/BUILD                # BUILD.toolchain_nvcc
    │   └── :cuda-toolkit, :nvcc-local-config, :nvcc-local, :nvcc-local-toolchain
    ├── nvcc_msvc/BUILD           # BUILD.toolchain_nvcc_msvc
    │   └── :cuda-toolkit, :nvcc-local-config, :nvcc-local, :nvcc-local-toolchain
    ├── clang/BUILD
    │   └── :cuda-toolkit, :clang-local-config, :clang-local, :clang-local-toolchain
    └── disabled/BUILD            # same three disabled targets
```

The explicit `12.6.3` and empty-version registrations both use this implementation.
The `nvcc-local-toolchain` aliases point to Windows registrations on a Windows host.

### 3. Multiple deliverable CTKs from `redist_json`

Each release gets the implementation layout above. The facade declares toolchains for each
and uses the highest declared release when `cuda:version` is empty.

```text
@cuda//toolchain/
├── BUILD
│   ├── :toolkit_version_is_{12_8_1,13_0_1,default}
│   ├── :nvcc-{linux,windows}-12_8_1-toolchain
│   ├── :nvcc-{linux,windows}-13_0_1-toolchain
│   ├── :nvcc-12_8_1-toolchain → :nvcc-linux-12_8_1-toolchain
│   ├── :nvcc-13_0_1-toolchain → :nvcc-linux-13_0_1-toolchain
│   ├── :nvcc-{linux,windows}-toolchain              # default: 13.0.1
│   └── :nvcc-local-toolchain → :nvcc-linux-toolchain
├── clang/BUILD
│   ├── :toolkit_version_is_{12_8_1,13_0_1,default}
│   ├── :clang-{12_8_1,13_0_1}-toolchain
│   └── :clang-local-toolchain                     # default: 13.0.1
└── disabled/BUILD

@cuda_toolchain_12_8_1//           # fixed 12.8.1 implementation
@cuda_toolchain_13_0_1//           # fixed 13.0.1 implementation
```

`register_toolchains("@cuda//toolchain/...")` registers the facade's declarations. The
implementation repositories are referenced by those declarations, not registered
recursively themselves. An undeclared version fails toolchain resolution when
CUDA is enabled.

### Downloaded components

`registry.bzl` maps redistribution component names to the targets exposed by the
toolkit. Both the public facade and the versioned implementations refer to the
same downloaded repositories:

```text
@cuda_nvcc_linux_x86_64_12_8_1//
├── nvcc/                         # extracted archive
│   ├── bin/
│   ├── include/
│   └── nvvm/
├── BUILD                         # BUILD.dctk_comp + BUILD.nvcc + BUILD.nvvm
└── defs.bzl                      # containing toolkit version = (12, 8)
```

Public component aliases select platform and version. An implementation's
component aliases select platforms within its fixed release. With only one
platform, that intermediate alias repository is omitted. Local and manually
assembled toolkits retain their own `defs.bzl`; redistribution facades have none.

## Generated connections and selection

The following excerpts come from the generated CUDA 12.8.1/13.0.1 repositories.
Loads, visibility, and unrelated attributes are omitted where indicated.

### Public component aliases

```starlark
# @cuda//BUILD
alias(name = "nvcc", actual = "@cuda_nvcc//:nvcc")

# @cuda_nvcc//BUILD.bazel: after platform selection, Linux x86_64 selects a release.
alias(
    name = "linux_x86_64_nvcc",
    actual = select({
        ":version_is_nvcc_13_0_1": "@cuda_nvcc_linux_x86_64_13_0_1//:nvcc",
        ":version_is_nvcc_12_8_1": "@cuda_nvcc_linux_x86_64_12_8_1//:nvcc",
        "//conditions:default": "@cuda_nvcc_linux_x86_64_13_0_1//:nvcc",
    }),
    # visibility omitted
)
```

### Toolchain registration to compiler

```starlark
# @cuda//toolchain/BUILD
config_setting(
    name = "toolkit_version_is_12_8_1",
    flag_values = {"@rules_cuda//cuda:version": "12.8.1"},
)

toolchain(
    name = "nvcc-linux-12_8_1-toolchain",
    exec_compatible_with = ["@platforms//os:linux"],
    target_compatible_with = ["@platforms//os:linux"],
    target_settings = [
        "@rules_cuda//cuda:is_enabled",
        "@rules_cuda//cuda:compiler_is_nvcc",
        ":toolkit_version_is_12_8_1",
    ],
    toolchain = "@cuda_toolchain_12_8_1//toolchain:nvcc-linux",
    toolchain_type = "@rules_cuda//cuda:toolchain_type",
    # visibility omitted
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

# @cuda_nvcc_toolchain_12_8_1//BUILD.bazel: platform alias fixed to one release.
alias(
    name = "linux_x86_64_nvcc",
    actual = select({
        ":version_is_nvcc_12_8_1": "@cuda_nvcc_linux_x86_64_12_8_1//:nvcc",
        "//conditions:default": "@cuda_nvcc_linux_x86_64_12_8_1//:nvcc",
    }),
    # visibility omitted
)
```

For `--@rules_cuda//cuda:version=12.8.1 --@rules_cuda//cuda:compiler=nvcc`, Bazel
selects this registration and resolves its component aliases during **analysis**.
During **execution**, the compile action runs the selected archive's
`nvcc/bin/nvcc`; there is no runtime version switch. An empty version selects the
13.0.1 registration in this example.

During BUILD loading, version checks such as `if_cuda_toolkit_version_ge` read
the repository's own `defs.bzl`. This keeps CUDA 12 header layouts independent of
CUDA 13 declarations.
