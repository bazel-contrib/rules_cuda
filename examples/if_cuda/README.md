# if_cuda Example

This example demonstrates how to conditionally include CUDA targets in your build.

By default, _rules_cuda_ rules are enabled. Disabling rules_cuda rules can be accomplished by passing the `@rules_cuda//cuda:enable`
flag at the command-line or via `.bazelrc`. 

## Building Example with rules_cuda

From the `examples` directory, build the sample application:

```bash
bazel build //if_cuda:main
```

And run the binary:
```bash
./bazel-bin/if_cuda/main
```

If a valid GPU device is running on your development machine, the application will exit successfully and print:
```
cuda enabled
```

If running without a valid GPU device, the code, as written, will print a CUDA error and exit:
```
CUDA_VISIBLE_DEVICES=-1 ./bazel-bin/if_cuda/main
CUDA Error Code  : 100
     Error String: no CUDA-capable device is detected
```

## Building Example without rules_cuda

To build the binary without CUDA support, disable rules_cuda:

```bash
bazel build //if_cuda:main --@rules_cuda//cuda:enable`
```

And run the binary:
```bash
./bazel-bin/if_cuda/main
```

The binary will output:
```
cuda disabled
```

### rules_cuda targets

Any attempt to build a rules_cuda-defined rule (e.g. `cuda_library` or `cuda_objects`) will _FAIL_ if rules_cuda is disabled, as the CUDA toolchain will not be registered for that invocation.

```
bazel build //if_cuda:kernel --@rules_cuda//cuda:enable=false
ERROR: /home/ryan/devel/rules_cuda/examples/if_cuda/BUILD.bazel:3:13: While resolving toolchains for target //if_cuda:kernel: No matching toolchains found for types @rules_cuda//cuda:toolchain_type.
To debug, rerun with --toolchain_resolution_debug='@rules_cuda//cuda:toolchain_type'
If platforms or toolchains are a new concept for you, we'd encourage reading https://bazel.build/concepts/platforms-intro.
ERROR: Analysis of target '//if_cuda:kernel' failed; build aborted:
FAILED: Build did NOT complete successfully (0 packages loaded, 133 targets configured)
```

## Developing for CUDA- and CUDA-free targets

Note the `BUILD.bazel` file takes care to ensure that
- CUDA-related dependencies are excluded when CUDA is disabled
- Preprocessor variables are set to enable compile-time differentiated codepaths between CUDA and non-CUDA builds

Our example build (when CUDA is enabled) sets a `CUDA_ENABLED` preprocessor variable. This variable is then checked in `main.cpp` to determine
the type of compilation underway. You are free to set any set of preprocessor variables as needed by your particular project. Checking whether
rules_cuda or not can be achieved simply by using Bazel's `select` feature:

```
select({
        "@rules_cuda//cuda:is_enabled": [], # add whatever settings are required if using CUDA
        "//conditions:default": [] # add whatever settings are required if not using CUDA
    })
```

We use this same mechanism to include the `cuda_library` dependencies.