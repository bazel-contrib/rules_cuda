# CUDA redistributable cross-compilation tests

These tests check that a multi-platform `redist_json` setup selects CUDA tools
for the execution platform and runtime libraries for the target platform.
They use Bzlmod and perform complete builds rather than analysis-only checks.

In this document, `linux-sbsa` means server-class aarch64 Linux. The shared
platforms and aarch64 C++ toolchain are in
`@rules_cuda//tests/integration/platforms`.

## Matrix

The workspace depends only on the execution and target platforms. The same
workspace is used from every client host.

| CI      | Directory                                    | Host           | Exec           | Target         |
| ------- | -------------------------------------------- | -------------- | -------------- | -------------- |
| yes     | `toolchain_redist_cross_lx64_exec_lsbsa_tgt` | linux-x86_64   | linux-x86_64   | linux-sbsa     |
| yes     | `toolchain_redist_cross_lsbsa_exec_lx64_tgt` | linux-x86_64   | linux-sbsa     | linux-x86_64   |
| yes     | `toolchain_redist_cross_lx64_exec_lsbsa_tgt` | windows-x86_64 | linux-x86_64   | linux-sbsa     |
| yes     | `toolchain_redist_cross_lsbsa_exec_lx64_tgt` | windows-x86_64 | linux-sbsa     | linux-x86_64   |
| yes     | `toolchain_redist_json`                      | windows-x86_64 | windows-x86_64 | windows-x86_64 |
| not yet | `toolchain_redist_cross_lx64_exec_lsbsa_tgt` | macos          | linux-x86_64   | linux-sbsa     |

The native Windows row is also covered by the regular integration workflow.
GitHub provides macOS runners for public repositories, but a macOS client
still needs a Linux remote executor because it cannot run Linux CUDA tools
itself.

## Linux host

Linux runs the x86_64 tools directly. For the reverse direction, binfmt sends
the aarch64 tools through qemu-user.

```text
┌─ Linux x86_64 (Bazel client = host) ─────────────────────────────┐
│  --platforms=linux_x86_64          (target artifacts)            │
│  --@rules_cuda//cuda:exec_platform=linux-sbsa                    │
│                                                                  │
│  spawn linux-sbsa nvcc / cicc / …                                │
│       │                                                          │
│       │  binfmt → qemu-user-static (same host, no network)       │
│       v                                                          │
│  ┌─ qemu-user ────────────────────────────────────────────────┐  │
│  │  runs the aarch64 CUDA tools                              │  │
│  │  produces linux-x86_64 target artifacts                   │  │
│  └────────────────────────────────────────────────────────────┘  │
│  run //:smoke natively on x86_64                                 │
└──────────────────────────────────────────────────────────────────┘
```

Install the host packages, then run both directions:

```bash
sudo apt-get update
sudo apt-get install -y g++-aarch64-linux-gnu qemu-user-static
sudo ln -sfn /usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1 /lib/ld-linux-aarch64.so.1
if [[ -e /lib/aarch64-linux-gnu && ! -L /lib/aarch64-linux-gnu ]]; then
  sudo cp -asn /usr/aarch64-linux-gnu/lib/. /lib/aarch64-linux-gnu/
else
  sudo ln -sfn /usr/aarch64-linux-gnu/lib /lib/aarch64-linux-gnu
fi

cd tests/integration
bash ./test_cross_all.sh
```

`install_and_drive_cross.sh` and `drive_cross_wsl.sh` provide local setup
helpers.

## Windows host with Linux execution

Bazel runs on Windows and sends Linux actions to NativeLink in WSL. The x86_64
CUDA tools run directly in WSL. The aarch64 CUDA tools run through qemu-user.

```text
┌─ Windows x86_64 (Bazel client = host) ───────────────────────────┐
│  --platforms=linux_sbsa            (target artifacts)            │
│  --@rules_cuda//cuda:exec_platform=linux-x86_64                  │
│  --extra_toolchains=@cuda//toolchain:nvcc-linux-toolchain        │
│  --remote_executor=grpc://127.0.0.1:1985                         │
│       │                                                          │
│       │  grpc :1985 via netsh portproxy                          │
│       v                                                          │
│  ┌─ Linux x86_64 worker in WSL2 Ubuntu ───────────────────────┐  │
│  │  NativeLink runs linux-x86_64 nvcc directly               │  │
│  │  aarch64-linux-gnu-g++ produces linux-sbsa objects         │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

```text
┌─ Windows x86_64 (Bazel client = host) ───────────────────────────┐
│  --platforms=linux_x86_64          (target artifacts)            │
│  --@rules_cuda//cuda:exec_platform=linux-sbsa                    │
│  --extra_toolchains=@cuda//toolchain:nvcc-linux-toolchain        │
│  --remote_executor=grpc://127.0.0.1:1985                         │
│       │                                                          │
│       │  grpc :1985 via netsh portproxy                          │
│       v                                                          │
│  ┌─ Linux x86_64 worker in WSL2 Ubuntu ───────────────────────┐  │
│  │  NativeLink spawns linux-sbsa nvcc                         │  │
│  │       │                                                    │  │
│  │       │  binfmt → qemu-user                                │  │
│  │       v                                                    │  │
│  │  aarch64 CUDA tools produce linux-x86_64 target artifacts │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

WSL uses its default NAT network. Windows connects to the worker through a
loopback port proxy:

```text
Windows Bazel                 WSL2
─────────────                 ────
grpc 127.0.0.1:1985
       │
       │  netsh interface portproxy
       │  127.0.0.1:1985 → <wsl-ip>:1985
       v
       └──────────────────────► NativeLink 0.0.0.0:1985
                                      │
                                      │ worker API :1986
                                      v
                               local action runner
```

Mirrored WSL networking has caused DNS failures while installing Ubuntu
packages on GitHub-hosted runners, so the CI setup leaves it disabled.

Run the Windows setup and both platform directions with:

```powershell
pwsh tests/integration/drive_cross_windows.ps1
```

For a qemu-system worker instead of WSL, see
[`rbe/README.md`](rbe/README.md).

## Driver

```bash
cd tests/integration
bash ./test_cross_all.sh
bash ./test_cross_all.sh --no-lx64-exec
bash ./test_cross_all.sh --no-lsbsa-exec
```

For each direction, the driver:

1. Builds `//:use_library` and `//:use_rule`.
2. Checks the execution and target redistributable selections with
   `aquery` and `cquery`.
3. Checks the output architecture on a local Linux host.
4. Builds and runs `//:smoke` when the target is local x86_64 Linux.

## WSL distribution

The CI worker uses Ubuntu rather than Alpine. NVIDIA's Linux CUDA
redistributables are linked against glibc, while Alpine uses musl. Adding a
glibc compatibility environment and a separate aarch64 GNU cross-toolchain to
Alpine would make the worker more complicated and remove much of the image-size
advantage.

## CI

Workflow: [`.github/workflows/cross-compile-tests.yaml`](../../.github/workflows/cross-compile-tests.yaml)

| Job       | Runs on      | Execution                                 |
| --------- | ------------ | ----------------------------------------- |
| `linux`   | Ubuntu 24.04 | local Linux; qemu-user for aarch64 tools  |
| `windows` | windows-2025 | Linux actions in a WSL2 NativeLink worker |

The workflow can also be started manually with `workflow_dispatch`.

## Toolchain registration

Hermetic toolkits generate `nvcc-linux-toolchain` and
`nvcc-windows-toolchain`. `nvcc-local-toolchain` selects the client-host
default. A non-Linux client using Linux remote execution passes
`--extra_toolchains=@cuda//toolchain:nvcc-linux-toolchain`.
