## Template files

- `BUILD.local_cuda_shared`: For `local_cuda` repo (CTK + toolchain) or `local_cuda_%{component_name}`
- `BUILD.local_cuda_headers`: For `local_cuda` repo (CTK + toolchain) or `local_cuda_%{component_name}` headers
- `BUILD.local_cuda_build_setting`: For `local_cuda` repo (CTK + toolchain) build_setting
- `BUILD.local_cuda_disabled`: For creating a dummy local configuration.
- `BUILD.local_toolchain_disabled`: For creating a dummy local toolchain.
- `BUILD.local_toolchain_clang`: For Clang device compilation toolchain.
- `BUILD.local_toolchain_nvcc`: For NVCC device compilation toolchain.
- `BUILD.local_toolchain_nvcc_msvc`: For NVCC device compilation with (MSVC as host compiler) toolchain.
- Otherwise, each `BUILD.*` corresponds to a component in CUDA Toolkit.

## Repository organization

We organize the generated repo as follows, for both `local_cuda` and `local_cuda_<component_repo_name>`

```
<repo_root>              # bazel unconditionally creates a directory for us
├── %{component_name}/   # cuda for local ctk, component name otherwise
│   ├── include/         #
│   └── %{libpath}/      # lib or lib64, platform dependent
├── defs.bzl             # generated
├── BUILD                # generated from BUILD.local_cuda and one/all of the component(s)
└── WORKSPACE            # generated
```

If the repo is `local_cuda`, we additionally generate toolchain config as follows

```
<repo_root>
└── toolchain/
    ├── BUILD            # the default nvcc toolchain
    ├── clang/           # the optional clang toolchain
    │   └── BUILD        #
    └── disabled/        # the fallback toolchain
        └── BUILD        #
```

## How are component repositories and `@local_cuda` connected?

The `registry.bzl` file holds mappings from our (`rules_cuda`) components name to various things.

The registry serve the following purpose:

1. maps our component names to full component names used `redistrib.json` file.

   This is purely for looking up the json files.

2. maps our component names to target names to be exposed under `@local_cuda` repo.

   To expose those targets, we use a `components_mapping` attr from our component names to labels of component
   repository (for example, `@local_cuda_nvcc`) as follows

```starlark
# in registry.bzl
...
    "cudart": ["cuda", "cuda_runtime", "cuda_runtime_static"],
...

# in WORKSPACE.bazel
cuda_component(
    name = "local_cuda_cudart_v12.6.77",
    component_name = "cudart",
    ...
)

local_cuda(
    name = "local_cuda",
    components_mapping = {"cudart": "@local_cuda_cudart_v12.6.77"},
    ...
)
```

This basically means the component `cudart` has `cuda`, `cuda_runtime` and `cuda_runtime_static` targets defined.

- In locally installed CTK, we setup the targets in `@local_cuda` directly.
- In a deliverable CTK, we setup the targets in `@local_cuda_cudart_v12.6.77` repo. And alias all targets to
  `@local_cuda` as follows

```starlark
alias(name = "cuda", actual = "@local_cuda_cudart_v12.6.77//:cuda")
alias(name = "cuda_runtime", actual = "@local_cuda_cudart_v12.6.77//:cuda_runtime")
alias(name = "cuda_runtime_static", actual = "@local_cuda_cudart_v12.6.77//:cuda_runtime_static")
```

`cuda_component` is in charge of setting up the repo `@local_cuda_cudart_v12.6.77`.
