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
- `BUILD.toolchain_nvcc_msvc`: For NVCC device compilation with (MSVC as host compiler) toolchain.
- Otherwise, each `BUILD.*` corresponds to a component in CUDA Toolkit.

## Repository organization

We organize the generated repo as follows, for both `cuda` and `cuda_<component_repo_name>`

```
<repo_root>              # bazel unconditionally creates a directory for us
├── %{component_name}/   # cuda for local ctk, component name otherwise
│   ├── include/         #
│   └── %{libpath}/      # lib or lib64, platform dependent
├── defs.bzl             # generated
├── BUILD                # generated with template_helper
└── WORKSPACE            # generated
```

If the repo is `cuda`, we additionally generate toolchain config as follows

```
<repo_root>
└── toolchain/
    ├── BUILD            # the default nvcc toolchain
    ├── clang/           # the optional clang toolchain
    │   └── BUILD        #
    └── disabled/        # the fallback toolchain
        └── BUILD        #
```

## How are component repositories and `@cuda` connected?

The `registry.bzl` file holds mappings from our (`rules_cuda`) components name to various things.

The registry serve the following purpose:

1. maps our component names to full component names used `redistrib.json` file.

   This is purely for looking up the json files.

2. maps our component names to target names to be exposed under `@cuda` repo.

   To expose those targets, we use a `components_mapping` attr from our component names to labels of component
   repository (for example, `@cuda_nvcc`) as follows

```starlark
# in registry.bzl
...
    "cudart": ["cuda", "cuda_runtime", "cuda_runtime_static"],
...

# in WORKSPACE.bazel
cuda_component(
    name = "cuda_cudart_v12.6.77",
    component_name = "cudart",
    ...
)

cuda_toolkit(
    name = "cuda",
    components_mapping = {"cudart": "@cuda_cudart_v12.6.77"},
    ...
)
```

This basically means the component `cudart` has `cuda`, `cuda_runtime` and `cuda_runtime_static` targets defined.

- In locally installed CTK, we setup the targets in `@cuda` directly.
- In a deliverable CTK, we setup the targets in `@cuda_cudart_v12.6.77` repo. And alias all targets to
  `@cuda` as follows

```starlark
alias(name = "cuda", actual = "@cuda_cudart_v12.6.77//:cuda")
alias(name = "cuda_runtime", actual = "@cuda_cudart_v12.6.77//:cuda_runtime")
alias(name = "cuda_runtime_static", actual = "@cuda_cudart_v12.6.77//:cuda_runtime_static")
```

`cuda_component` is in charge of setting up the repo `@cuda_cudart_v12.6.77`.
