# Compute Sanitizer

Run GPU tests under [NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)
using Bazel's `--run_under` option. The launcher currently supports Linux and uses the
sanitizer from the configured CUDA toolkit, including its injection libraries and
process launch helpers. It does not search `PATH` for another installation.

## Run tests

```sh
bazel test --run_under=@rules_cuda//:compute_sanitizer //path/to:gpu_test
```

The same target is available as `@rules_cuda//tools/compute_sanitizer`.
The default tool is `memcheck`. Select other tools by adding options after the label:

```sh
bazel test --run_under='@rules_cuda//:compute_sanitizer --tool initcheck' //path/to:gpu_test
bazel test --run_under='@rules_cuda//:compute_sanitizer --tool racecheck' //path/to:gpu_test
bazel test --run_under='@rules_cuda//:compute_sanitizer --tool synccheck' //path/to:gpu_test
```

| Tool        | Checks                                           |
| ----------- | ------------------------------------------------ |
| `memcheck`  | Invalid and misaligned memory accesses           |
| `initcheck` | Uninitialized device global-memory reads         |
| `racecheck` | Shared-memory access hazards                     |
| `synccheck` | Invalid use of device synchronization primitives |

The launcher sets `--error-exitcode 1` so findings fail the Bazel test, even when the
application returns success. It also sets `--target-processes all` to check child
processes. Additional sanitizer options follow these defaults. Use `--test_arg` for
arguments to the test itself.

## Configure the toolkit

For a local CUDA toolkit, use the [toolchain setup](../index.md#toolchain-detection).
The installation must include `compute-sanitizer/compute-sanitizer` and its accompanying
support files.

For a hermetic toolkit, the `sanitizer` component maps to NVIDIA's `cuda_sanitizer_api`
redistributable. If your existing `cuda.redist_json` explicitly lists `components`, add
`"sanitizer"` to that list. For example, a CUDA 13 toolkit can use:

```starlark
cuda.redist_json(
    name = "cuda_13_0_2",
    version = "13.0.2",
    platforms = ["linux-x86_64"],
    components = ["cccl", "crt", "cudart", "nvcc", "nvvm", "sanitizer"],
)
```

Keep the existing `cuda.toolkit`, `use_repo`, and toolchain registration from your
toolchain setup. If `components` is omitted, all available components are included.

## Execution requirements

The test execution environment must provide a supported NVIDIA GPU and driver.
`--run_under` does not configure GPU scheduling, remote workers, or sandbox device
access. Use your existing GPU test execution configuration. Sanitized tests may need a
longer `--test_timeout` than ordinary runs.

## Validate the integration

The repository includes manual GPU fixtures. From a checkout configured with CUDA and
the sanitizer component, the clean fixture should pass:

```sh
bazel test --run_under=//:compute_sanitizer //tests/compute_sanitizer/gpu:clean
```

These two fixtures intentionally return application success after an invalid device
read or an uninitialized device read. Each command should fail with sanitizer diagnostics:

```sh
bazel test --run_under=//:compute_sanitizer //tests/compute_sanitizer/gpu:invalid_access
bazel test --run_under='//:compute_sanitizer --tool initcheck' //tests/compute_sanitizer/gpu:uninitialized
```

All three fixtures are tagged `manual` and require a GPU. The ordinary utility suite
also tests argument forwarding, exit-status propagation, and runfiles lookup without
requiring CUDA or a GPU.
