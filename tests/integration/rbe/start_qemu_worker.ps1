# Start (or print how to start) NativeLink under qemu-system for cases 3–4.
#
# Alternate nesting (WSL-native is preferred — see start_wsl_worker.ps1):
#   Windows bazelisk  --remote_executor=grpc://127.0.0.1:1985
#         |
#         | hostfwd TCP 127.0.0.1:1985 -> guest:1985
#         v
#   qemu-system-x86_64 Linux guest running NativeLink
#
# Needs qemu-system-x86_64 on PATH (or QEMU_SYSTEM) and QEMU_DISK.
#
# Minimal usage:
#   $env:QEMU_DISK = "C:\path\to\linux.qcow2"
#   .\start_qemu_worker.ps1
#   $env:CROSS_REMOTE_BAZEL_FLAGS = "--remote_executor=grpc://127.0.0.1:1985"
#   bash tests/integration/test_cross_all.sh --required-only --no-linux

$ErrorActionPreference = "Stop"
$Port = if ($env:RBE_PORT) { [int]$env:RBE_PORT } else { 1985 }
$Qemu = if ($env:QEMU_SYSTEM) { $env:QEMU_SYSTEM } else { "qemu-system-x86_64" }
$Disk = $env:QEMU_DISK
$Mem = if ($env:QEMU_MEM) { $env:QEMU_MEM } else { "4G" }
$Cpus = if ($env:QEMU_CPUS) { $env:QEMU_CPUS } else { "4" }

Write-Host "qemu-system NativeLink worker launcher (cases 3–4)"
Write-Host "  remote_executor = grpc://127.0.0.1:$Port"
Write-Host "  qemu            = $Qemu"
Write-Host "  disk            = $Disk"

if (-not $Disk) {
    Write-Host @"

QEMU_DISK is not set. Provide a Linux x86_64 disk image that:
  1. Has network and SSH or console access
  2. Installs an RE worker listening on 0.0.0.0:$Port
     (example config: tests/integration/rbe/nativelink-qemu.json)
  3. Has g++-aarch64-linux-gnu for sbsa target compiles
  4. Can fetch CUDA redist packages (curl/ca-certificates)

Example once the image is ready:
  `$env:QEMU_DISK = 'D:\vms\rules-cuda-rbe.qcow2'
  `$env:RBE_PORT = '$Port'
  .\start_qemu_worker.ps1

Then from the repo (Git bash or pwsh calling bash):
  `$env:CROSS_REMOTE_BAZEL_FLAGS = '--remote_executor=grpc://127.0.0.1:$Port'
  bash tests/integration/test_cross_all.sh --required-only

"@
    exit 2
}

if (-not (Test-Path $Disk)) {
    throw "QEMU_DISK not found: $Disk"
}

$qemuCmd = Get-Command $Qemu -ErrorAction SilentlyContinue
if (-not $qemuCmd) {
    throw "qemu binary not found: $Qemu (install QEMU or set QEMU_SYSTEM)"
}

# user networking + hostfwd: Windows localhost:$Port -> guest:$Port
$fwd = "hostfwd=tcp:127.0.0.1:${Port}-:${Port}"
Write-Host "Starting qemu with -netdev user,$fwd"
& $Qemu `
    -machine q35,accel=whpx:tcg `
    -cpu max `
    -m $Mem `
    -smp $Cpus `
    -drive "file=$Disk,if=virtio,format=qcow2" `
    -netdev "user,id=net0,$fwd" `
    -device "virtio-net-pci,netdev=net0" `
    -nographic
