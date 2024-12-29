- `BUILD.local_cuda_shared`: For local_cuda repo (CTK + toolchain) or local_cuda_%{component_name}
- `BUILD.local_cuda_headers`: For local_cuda repo (CTK + toolchain) or local_cuda_%{component_name} headers
- `BUILD.local_cuda_build_setting`: For local_cuda repo (CTK + toolchain) build_setting
- `BUILD.local_cuda_disabled`: For creating a dummy local configuration.
- `BUILD.local_toolchain_disabled`: For creating a dummy local toolchain.
- `BUILD.local_toolchain_clang`: For Clang device compilation toolchain.
- `BUILD.local_toolchain_nvcc`: For NVCC device compilation toolchain.
- `BUILD.local_toolchain_nvcc_msvc`: For NVCC device compilation with (MSVC as host compiler) toolchain.
- Otherwise, each `BUILD.*` corresponds to a component in CUDA Toolkit.

We organize the generated repo as follows

```
<repo_root>              # bazel unconditionally create a directory for us
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
