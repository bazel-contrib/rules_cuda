- `BUILD.local_cuda_shared`: For local_cuda repo (CTK + toolchain) or local_cuda_%{component_name}
- `BUILD.local_cuda_headers`: For local_cuda repo (CTK + toolchain) or local_cuda_%{component_name} headers
- `BUILD.local_cuda_build_setting`: For local_cuda repo (CTK + toolchain) build_setting
- `BUILD.local_cuda_disabled`: For creating a dummy local configuration.
- `BUILD.local_toolchain_nvcc`:
- `BUILD.local_toolchain_nvcc_msvc`:
- `BUILD.local_toolchain_nvcc`:
- Otherwise, each `BUILD.*` corresponds to a component in CUDA Toolkit.

We organize the generated repo as follows

```
<repo_root>              # bazel unconditionally create a directory for us
├── %{component_name}    # placeholder
│   ├── include/         #
│   ├── %{libpath}/      # placeholder
│   └── version.json     # copy redistrib json to it if not exist and is a deliverable
├── defs.bzl             # generated
├── BUILD                # generated from BUILD.local_cuda and one/all of the component(s)
└── WORKSPACE            # generated
```

If the repo is `local_cuda`, we additionally generate toolchain config as follows

```
<repo_root>
└── toolchain
    ├── BUILD            # the default nvcc toolchain
    ├── clang            # the optional clang toolchain
    │   └── BUILD        #
    └── disabled         # the fallback toolchain
        └── BUILD        #
```
