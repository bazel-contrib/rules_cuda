# Simple remote execution for REQUIRED-A (Windows host)

## Goal

```text
Windows Bazel  (host)
    --remote_executor=grpc://127.0.0.1:1985
            |
            |  qemu user-net hostfwd
            v
qemu-system-x86_64 Linux guest
    RE worker (e.g. NativeLink)
    linux-x86_64 nvcc (exec)
    aarch64-linux-gnu-g++ (target cross)
```

**WSL is not required.** The guest is plain Linux under qemu.

## REQUIRED cases

| ID | Host | Exec | Target | Local driver |
|----|------|------|--------|--------------|
| **A** | Windows | linux-x86_64 | linux-sbsa | this RBE + `test_cross_all.sh` case 3 |
| **B** | Linux x64 | linux-sbsa | linux-x86_64 | qemu-user on the Linux host (no system qemu) |

## Guest setup (once)

1. Create a Linux x86_64 qcow2 (Debian/Ubuntu cloud image is fine).
2. Inside the guest:
   - Install `g++-aarch64-linux-gnu`, `qemu-user-static` (only if you also run optional case 4), `ca-certificates`, `curl`.
   - Install an RE implementation (e.g. [NativeLink](https://github.com/TraceMachina/nativelink) musl linux binary).
   - Point it at `nativelink-qemu.json` (or equivalent) listening on **`0.0.0.0:1985`**.
3. Ensure the guest firewall allows port 1985.

## Start worker from Windows

```powershell
$env:QEMU_DISK = "D:\vms\rules-cuda-rbe.qcow2"
$env:RBE_PORT = "1985"
.\tests\integration\rbe\start_qemu_worker.ps1
```

## Run REQUIRED-A from Windows

```powershell
$env:CROSS_REMOTE_BAZEL_FLAGS = "--remote_executor=grpc://127.0.0.1:1985"
# from Git bash in repo:
bash tests/integration/test_cross_all.sh --required-only
```

`--required-only` runs **B** (skipped on Windows) and **A** (runs when remote flags are set).

## Network checklist

| Check | Expect |
|-------|--------|
| hostfwd | `hostfwd=tcp:127.0.0.1:1985-:1985` |
| guest listen | `0.0.0.0:1985` |
| Windows probe | `Test-NetConnection 127.0.0.1 -Port 1985` |
| Bazel flag | `--remote_executor=grpc://127.0.0.1:1985` |

If the port is closed, Bazel will fail scheduling Linux actions — fix qemu/network before debugging rules_cuda.
