# Linux remote execution for Windows-host cross (cases 3–4)

Windows cannot exec Linux CUDA tools. Case 3/4 send actions to a Linux
remote-execution (RE) worker. Two nestings are supported.

## Nesting A — NativeLink under WSL (preferred)

Keep **default WSL2 NAT** so DNS inside the distro works (apt/curl/GitHub).
Windows reaches NativeLink via `netsh interface portproxy`:

```text
127.0.0.1:1985  →  <wsl-eth-ip>:1985
```

Avoid `networkingMode=mirrored` on GitHub Actions: it often breaks WSL DNS
(`Temporary failure resolving 'archive.ubuntu.com'`).

```text
┌─ Windows host ──────────────────────────────────────────────────┐
│  bazelisk  --remote_executor=grpc://127.0.0.1:1985              │
│       │      (fallback: grpc://<wsl-eth-ip>:1985)               │
│       │                                                         │
│       │  netsh portproxy (loopback → WSL NAT IP)                │
│       v                                                         │
│  ┌─ WSL2 distro (Ubuntu, linux-x86_64, default NAT) ─────────┐  │
│  │                                                           │  │
│  │  NativeLink                                               │  │
│  │    public API   0.0.0.0:1985  ◄── Windows Bazel           │  │
│  │    worker API   0.0.0.0:1986  (internal)                  │  │
│  │         │                                                 │  │
│  │         v                                                 │  │
│  │    local worker runs actions:                             │  │
│  │      case 3: linux-x86_64 nvcc (native)                   │  │
│  │              aarch64-linux-gnu-g++ → linux-sbsa objects   │  │
│  │      case 4: linux-sbsa nvcc under qemu-user (binfmt)     │  │
│  │              host-arch link for linux-x86_64 target       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

```powershell
pwsh tests/integration/rbe/start_wsl_worker.ps1
# or full case-3 driver:
pwsh tests/integration/drive_cross_windows.ps1
```

`start_wsl_worker.ps1`:

1. Installs `g++-aarch64-linux-gnu` (+ qemu-user for case 4) in WSL if missing
2. Holds the distro open with `sleep infinity` (avoids WSL shutting down after a short `wsl` invocation and killing a background worker)
3. Starts NativeLink via a Windows-owned `wsl.exe` process (`basic_cas.json5` on `0.0.0.0:1985`)
4. Adds `netsh portproxy` `127.0.0.1:1985` → WSL eth IP
5. Probes localhost then WSL IP; sets `CROSS_REMOTE_BAZEL_FLAGS`

Then:

```powershell
$env:CROSS_REMOTE_BAZEL_FLAGS = "--remote_executor=grpc://127.0.0.1:1985 --remote_default_exec_properties=OSFamily=Linux"
bash tests/integration/test_cross_all.sh --required-only --no-linux
```

## Nesting B — NativeLink under qemu-system (optional)

Same RE protocol; Linux is a full VM instead of WSL.

```text
┌─ Windows host ──────────────────────────────────────────────────┐
│  bazelisk  --remote_executor=grpc://127.0.0.1:1985              │
│       │                                                         │
│       │  QEMU user-net hostfwd                                  │
│       │  hostfwd=tcp:127.0.0.1:1985-:1985                       │
│       v                                                         │
│  ┌─ qemu-system-x86_64 ──────────────────────────────────────┐  │
│  │  Linux guest                                              │  │
│  │    NativeLink  0.0.0.0:1985 / :1986                       │  │
│  │    same tool layout as WSL nesting                        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

```powershell
$env:QEMU_DISK = "D:\vms\rules-cuda-rbe.qcow2"
$env:RBE_PORT = "1985"
.\tests\integration\rbe\start_qemu_worker.ps1
```

Guest checklist:

1. Linux x86_64 disk with network
2. RE worker listening on `0.0.0.0:1985` (`bootstrap_nativelink_linux.sh`)
3. `g++-aarch64-linux-gnu` for sbsa target compiles
4. Network for CUDA redist download

## Which cases use this RE layer?

| Case | Host    | Exec         | Target       | Worker notes                              |
| ---- | ------- | ------------ | ------------ | ----------------------------------------- |
| 3    | Windows | linux-x86_64 | linux-sbsa   | WSL-native tools (no qemu-user)           |
| 4    | Windows | linux-sbsa   | linux-x86_64 | qemu-user inside WSL/guest for sbsa tools |
| 2    | Linux   | linux-sbsa   | linux-x86_64 | **no RE** — qemu-user on the Linux host   |

## Network checklist

| Check         | Expect                                        |
| ------------- | --------------------------------------------- |
| Worker listen | `0.0.0.0:1985` (public), `:1986` (worker API) |
| Windows probe | `Test-NetConnection 127.0.0.1 -Port 1985`     |
| Bazel flag    | `--remote_executor=grpc://127.0.0.1:1985`     |

If the port is closed, Bazel fails scheduling Linux actions — fix WSL/qemu
before debugging rules_cuda.
