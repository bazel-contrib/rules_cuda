# redist_json cross-compile integration tests

Bzlmod-only full builds that exercise multi-platform CUDA redistrib selection
with distinct **exec** (nvcc/cicc/…) and **target** (cudart/…) platforms.

“linux aarch” means **`linux-sbsa`** (server-class ARM). Use
`--@rules_cuda//cuda:aarch64=aarch64` only if you add Tegra variants later.

Shared platforms / aarch64 C++ toolchain:
`@rules_cuda//tests/integration/platforms`.

## Matrix (keep all four)

| Case | Priority | Directory | Host | Exec | Target |
|------|----------|-----------|------|------|--------|
| **2 / B** | **REQUIRED** | `toolchain_redist_cross_lsbsa_exec_lx64_tgt` | linux-x86_64 | linux-sbsa | linux-x86_64 |
| **3 / A** | **REQUIRED** | `toolchain_redist_cross_win_lx64_exec_lsbsa_tgt` | windows-x86_64 | linux-x86_64 | linux-sbsa |
| 1 | optional | `toolchain_redist_cross_lx64_exec_lsbsa_tgt` | linux-x86_64 | linux-x86_64 | linux-sbsa |
| 4 | optional | `toolchain_redist_cross_win_lsbsa_exec_lx64_tgt` | windows-x86_64 | linux-sbsa | linux-x86_64 |

### REQUIRED-B (case 2)

- Linux x64 Bazel client.
- Exec tools are **linux-sbsa** (run under **qemu-user-static** / binfmt on the same machine).
- Target is **linux-x86_64** — intermediate aarch is only the tool env.
- Builds `//:use_rule`, asserts redists, checks object Machine is **X86-64**, builds and **runs** `//:smoke` (no CUDA device).

### REQUIRED-A (case 3)

- **Windows** Bazel client (true Windows host).
- Exec tools are **linux-x86_64** — cannot spawn locally; need a **Linux RE worker**.
- Preferred worker: **qemu-system-x86_64** guest + hostfwd (WSL not required). See [`rbe/README.md`](rbe/README.md).
- Target is **linux-sbsa** (aarch64 objects via cross gcc on the worker).

## Driver

```bash
cd tests/integration
bash ./test_cross_all.sh                 # all applicable cases
bash ./test_cross_all.sh --required-only # REQUIRED-A + REQUIRED-B only
# --no-1 --no-2 --no-3 --no-4 --no-linux --no-windows
```

Each case:

1. `bazel build //:use_library` and `//:use_rule` with
   `--platforms=…` and `--@rules_cuda//cuda:exec_platform=…`
2. aquery/cquery asserts exec vs target redist platform segments
3. On Linux cases, `readelf`/`file` checks artifact Machine
4. REQUIRED-B also builds and runs `//:smoke`

## Linux host (cases 1 + REQUIRED-B)

```bash
sudo apt-get update
sudo apt-get install -y g++-aarch64-linux-gnu qemu-user-static
# map aarch64 guest loader so binfmt can run sbsa nvcc (REQUIRED-B):
sudo ln -sfn /usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1 /lib/ld-linux-aarch64.so.1
sudo ln -sfn /usr/aarch64-linux-gnu/lib /lib/aarch64-linux-gnu

bash ./test_cross_all.sh --no-windows
# or only REQUIRED-B:
bash ./test_cross_all.sh --required-only --no-windows
```

Local helpers (optional): `install_and_drive_cross.sh`, `drive_cross_wsl.sh`.

## Windows host (REQUIRED-A + optional case 4)

1. Start Linux RE worker in qemu (see [`rbe/README.md`](rbe/README.md)).
2. Point Bazel at it and run:

```powershell
$env:CROSS_REMOTE_BAZEL_FLAGS = "--remote_executor=grpc://127.0.0.1:1985"
bash tests/integration/test_cross_all.sh --required-only
```

Without `CROSS_REMOTE_BAZEL_FLAGS`, Windows cases are **skipped** (not failed).

## CI

Ubuntu nvidia/nvcc jobs install apt cross/qemu packages and run
`test_cross_all.sh --no-windows` (optional case 1 + REQUIRED-B).

Windows PR CI does not run REQUIRED-A unless a remote executor is configured.
