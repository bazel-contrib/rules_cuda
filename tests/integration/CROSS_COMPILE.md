# redist_json cross-compile integration tests

Bzlmod-only full builds that exercise multi-platform CUDA redistrib selection
with distinct **exec** (nvcc/cicc/…) and **target** (cudart/…) platforms.

“linux aarch” means **`linux-sbsa`** (server-class ARM). Use
`--@rules_cuda//cuda:aarch64=aarch64` only if you add Tegra variants later.

Shared platforms / aarch64 C++ toolchain:
`@rules_cuda//tests/integration/platforms`.

## Matrix (keep all four)

| Case  | CI      | Directory                                        | Host           | Exec         | Target       |
| ----- | ------- | ------------------------------------------------ | -------------- | ------------ | ------------ |
| **2** | primary | `toolchain_redist_cross_lsbsa_exec_lx64_tgt`     | linux-x86_64   | linux-sbsa   | linux-x86_64 |
| **3** | primary | `toolchain_redist_cross_win_lx64_exec_lsbsa_tgt` | windows-x86_64 | linux-x86_64 | linux-sbsa   |
| 1     | extra   | `toolchain_redist_cross_lx64_exec_lsbsa_tgt`     | linux-x86_64   | linux-x86_64 | linux-sbsa   |
| 4     | extra   | `toolchain_redist_cross_win_lsbsa_exec_lx64_tgt` | windows-x86_64 | linux-sbsa   | linux-x86_64 |

### Case 2 — Linux host, sbsa exec via qemu-user

Same machine: Bazel and tools share one Linux kernel. When exec tools are
aarch64 ELFs, the kernel runs them through **qemu-user** (binfmt).

```text
┌─ Linux x86_64 (Bazel client = host) ─────────────────────────────┐
│  --platforms=linux_x86_64          (target artifacts)            │
│  --@rules_cuda//cuda:exec_platform=linux-sbsa                    │
│                                                                  │
│  spawn linux-sbsa nvcc / cicc / …                                │
│       │                                                          │
│       │  binfmt → qemu-user-static (same host, no network)       │
│       v                                                          │
│  ┌─ qemu-user (user-mode) ────────────────────────────────────┐  │
│  │  aarch64 ELF toolchain (linux-sbsa redist)                 │  │
│  │  writes x86_64 objects / links //:smoke                    │  │
│  └────────────────────────────────────────────────────────────┘  │
│  run //:smoke natively on x86_64                                 │
└──────────────────────────────────────────────────────────────────┘
```

### Case 3 — Windows host, Linux exec via RE (WSL preferred)

Bazel runs on Windows. Linux tools cannot exec locally, so actions go to a
**Linux remote-execution worker**. Preferred worker: **NativeLink inside WSL**
(native x86_64 Linux — no qemu for case 3 exec). Alternate: NativeLink inside
a **qemu-system** guest (see [`rbe/README.md`](rbe/README.md)).

```text
┌─ Windows x86_64 (Bazel client = host) ───────────────────────────┐
│  --platforms=linux_sbsa            (target artifacts)            │
│  --@rules_cuda//cuda:exec_platform=linux-x86_64                  │
│  --extra_toolchains=@cuda//toolchain:nvcc-linux-toolchain        │
│  --remote_executor=grpc://127.0.0.1:1985                         │
│       │                                                          │
│       │  grpc :1985                                              │
│       │    preferred: 127.0.0.1 with WSL networkingMode=mirrored │
│       │    fallback:  WSL eth IPv4 under default NAT             │
│       v                                                          │
│  ┌─ Linux x86_64 RE worker ───────────────────────────────────┐  │
│  │                                                            │  │
│  │  path A (preferred): WSL2 Ubuntu (host/mirrored network)   │  │
│  │    NativeLink listens 0.0.0.0:1985 (API) + :1986 (worker)  │  │
│  │    exec tools = linux-x86_64 nvcc  (native, no qemu-user)  │  │
│  │    target C++ = aarch64-linux-gnu-g++ → linux-sbsa objs    │  │
│  │                                                            │  │
│  │  path B (optional): qemu-system-x86_64 guest               │  │
│  │    same NativeLink + tool layout; port via QEMU hostfwd    │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

Network wiring (case 3, WSL path):

```text
  Windows Bazel                 WSL2 (networkingMode=mirrored)
  ─────────────                 ──────────────────────────────
       │  grpc 127.0.0.1:1985
       ├──────────────────────► NativeLink public  0.0.0.0:1985
       │                               │
       │                               │ worker_api 127.0.0.1:1986
       │                               v
       │                        local worker (runs actions)

  If mirrored is unavailable, the driver falls back to the WSL NAT IPv4
  (hostname -I) instead of 127.0.0.1.
```

### Case 1 — Linux host, same-arch exec, sbsa target

No qemu for tools: exec is linux-x86_64. Cross-gcc produces aarch64 objects.

### Case 4 — Windows host, sbsa exec (needs qemu-user on the Linux worker)

Same RE shape as case 3, but exec tools are linux-sbsa. On an x86_64 WSL/guest,
those tools run under **qemu-user** inside the Linux worker.

```text
┌─ Windows Bazel ──grpc:1985──► ┌─ WSL / qemu-system (x86_64 Linux) ──┐
│                               │  NativeLink                          │
│                               │    spawn sbsa nvcc                   │
│                               │      └─ qemu-user (aarch64 ELF)      │
│                               │  target = linux-x86_64               │
└───────────────────────────────┴──────────────────────────────────────┘
```

## Driver

```bash
cd tests/integration
bash ./test_cross_all.sh                 # all applicable cases
bash ./test_cross_all.sh --required-only # cases 2 + 3 only
# --no-1 --no-2 --no-3 --no-4 --no-linux --no-windows
```

Each case:

1. `bazel build //:use_library` and `//:use_rule` with
   `--platforms=…` and `--@rules_cuda//cuda:exec_platform=…`
2. aquery/cquery asserts exec vs target redist platform segments
3. On Linux cases, `readelf`/`file` checks artifact Machine
4. Case 2 also builds and runs `//:smoke`

## Linux host (cases 1 + 2)

```bash
sudo apt-get update
sudo apt-get install -y g++-aarch64-linux-gnu qemu-user-static
# map aarch64 guest loader so binfmt can run sbsa nvcc (case 2):
sudo ln -sfn /usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1 /lib/ld-linux-aarch64.so.1
sudo ln -sfn /usr/aarch64-linux-gnu/lib /lib/aarch64-linux-gnu

bash ./test_cross_all.sh --no-windows
# or only case 2:
bash ./test_cross_all.sh --required-only --no-windows
```

Local helpers (optional): `install_and_drive_cross.sh`, `drive_cross_wsl.sh`.

## Windows host (cases 3 + 4)

Preferred: **WSL-native NativeLink** (diagram under case 3).

```powershell
# One-shot: start WSL worker + run case 3
pwsh tests/integration/drive_cross_windows.ps1

# Or step by step:
pwsh tests/integration/rbe/start_wsl_worker.ps1
$env:CROSS_REMOTE_BAZEL_FLAGS = "--remote_executor=grpc://127.0.0.1:1985 --remote_default_exec_properties=OSFamily=Linux"
bash tests/integration/test_cross_all.sh --required-only --no-linux
```

Without `CROSS_REMOTE_BAZEL_FLAGS`, case 3 **fails** (no silent skip).

Optional qemu-system guest instead of WSL: [`rbe/README.md`](rbe/README.md).

## CI

Workflow: [`.github/workflows/cross-compile-tests.yaml`](../../.github/workflows/cross-compile-tests.yaml)

| Job         | Runs         | Nesting                                                                  |
| ----------- | ------------ | ------------------------------------------------------------------------ |
| **linux**   | Ubuntu 24.04 | host Linux → case 2 uses qemu-user on the same machine                   |
| **windows** | windows-2025 | host Windows → case 3 RE into WSL NativeLink (`drive_cross_windows.ps1`) |

`workflow_dispatch` is enabled for manual runs.

## Toolchain note

Hermetic (deliverable) toolkits generate both `nvcc-linux-toolchain` and
`nvcc-windows-toolchain`, but only `nvcc-local-toolchain` (host alias) is
registered by default. Windows-host cross cases pass
`--extra_toolchains=@cuda//toolchain:nvcc-linux-toolchain`. Local install
templates are unchanged.
