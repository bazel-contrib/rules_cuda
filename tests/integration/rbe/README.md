# Linux remote execution for REQUIRED-A (Windows host)

## Goal

```text
Windows Bazel  (host)
    --remote_executor=grpc://127.0.0.1:1985
            |
            |  localhost
            v
WSL Ubuntu (preferred)  — or qemu-system-x86_64 guest
    RE worker (NativeLink)
    linux-x86_64 nvcc (exec)
    aarch64-linux-gnu-g++ (target cross)
    qemu-user only if exec arch is aarch (optional case 4)
```

## REQUIRED cases

| ID | Host | Exec | Target | Local driver |
|----|------|------|--------|--------------|
| **A** | Windows | linux-x86_64 | linux-sbsa | WSL RE + `test_cross_all.sh` case 3 |
| **B** | Linux x64 | linux-sbsa | linux-x86_64 | qemu-user on the Linux host (no system qemu) |

## Preferred: WSL worker

```powershell
pwsh tests/integration/rbe/start_wsl_worker.ps1
# or the full driver:
pwsh tests/integration/drive_cross_windows.ps1
```

`start_wsl_worker.ps1`:

1. Installs `g++-aarch64-linux-gnu` (+ qemu-user for optional aarch exec) in WSL
2. Downloads NativeLink musl binary
3. Starts `basic_cas.json5` on `0.0.0.0:1985`
4. Waits until Windows can open `127.0.0.1:1985`

Then:

```powershell
$env:CROSS_REMOTE_BAZEL_FLAGS = "--remote_executor=grpc://127.0.0.1:1985 --remote_default_exec_properties=OSFamily=Linux"
bash tests/integration/test_cross_all.sh --required-only --no-linux
```

## Alternative: qemu-system guest

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

## Network checklist

| Check | Expect |
|-------|--------|
| WSL / guest listen | `0.0.0.0:1985` |
| Windows probe | `Test-NetConnection 127.0.0.1 -Port 1985` |
| Bazel flag | `--remote_executor=grpc://127.0.0.1:1985` |

If the port is closed, Bazel will fail scheduling Linux actions — fix WSL/qemu before debugging rules_cuda.
