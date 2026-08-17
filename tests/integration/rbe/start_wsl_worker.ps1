# Start a Linux remote-execution worker inside WSL for REQUIRED-A.
#
# Shape:
#   Windows bazelisk  --remote_executor=grpc://127.0.0.1:<port>
#         |
#         | localhost (WSL2 mirrored / localhost relay)
#         v
#   WSL Ubuntu  (linux-x86_64 exec)
#     NativeLink RE worker
#     g++-aarch64-linux-gnu for linux-sbsa target
#     qemu-user only if exec arch needs it (not for REQUIRED-A x64 exec)
#
# Usage:
#   .\tests\integration\rbe\start_wsl_worker.ps1
#   $env:CROSS_REMOTE_BAZEL_FLAGS = "--remote_executor=grpc://127.0.0.1:1985"
#   bash tests/integration/test_cross_all.sh --required-only --no-linux

$ErrorActionPreference = "Stop"

$Port = if ($env:RBE_PORT) { [int]$env:RBE_PORT } else { 1985 }
$Distro = if ($env:WSL_DISTRO) { $env:WSL_DISTRO } else { $null }
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path

function Invoke-Wsl {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList
    )
    $prefix = @()
    if ($Distro) { $prefix += @("-d", $Distro) }
    & wsl (@($prefix) + $ArgumentList)
    if ($LASTEXITCODE -ne 0) {
        throw "wsl failed ($LASTEXITCODE): $($ArgumentList -join ' ')"
    }
}

function Invoke-WslBash {
    param([Parameter(Mandatory = $true)][string]$Command)
    Invoke-Wsl -ArgumentList @("-e", "bash", "-lc", $Command)
}

$WslRepo = (Invoke-Wsl -ArgumentList @("-e", "wslpath", "-a", $RepoRoot) | Select-Object -Last 1).ToString().Trim()
$SetupDeps = "$WslRepo/tests/integration/rbe/setup_wsl_cross_deps.sh"
$Bootstrap = "$WslRepo/tests/integration/rbe/bootstrap_nativelink_linux.sh"
$LogDir = if ($env:RBE_LOG_DIR) { $env:RBE_LOG_DIR } else { Join-Path $env:TEMP "rules_cuda_rbe" }
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$WorkerLog = Join-Path $LogDir "nativelink-wsl.log"
$WslLog = (Invoke-Wsl -ArgumentList @("-e", "wslpath", "-a", $WorkerLog) | Select-Object -Last 1).ToString().Trim()

Write-Host "REQUIRED-A WSL worker"
Write-Host "  repo (Windows) = $RepoRoot"
Write-Host "  repo (WSL)     = $WslRepo"
Write-Host "  remote_executor= grpc://127.0.0.1:$Port"
Write-Host "  worker log     = $WorkerLog"

# Ensure LF line endings (Windows checkouts may be CRLF).
Invoke-WslBash "sed -i 's/\r`$//' '$SetupDeps' '$Bootstrap' && chmod +x '$SetupDeps' '$Bootstrap'"

Write-Host "Installing WSL cross-compile deps..."
Invoke-WslBash "bash '$SetupDeps'"

# Stop any previous worker on this port (best effort).
# Avoid `pkill -f nativelink` — it can match the shell wrapper cmdline.
try {
    Invoke-WslBash "fuser -k ${Port}/tcp >/dev/null 2>&1 || true; fuser -k 1986/tcp >/dev/null 2>&1 || true; sleep 1"
} catch {
    # ignore cleanup failures
}

Write-Host "Starting NativeLink in WSL (background)..."
# Capture only the pid line; ignore fstab noise from wsl.exe.
$startCmd = "export RBE_PORT=$Port; nohup bash '$Bootstrap' >'$WslLog' 2>&1 & echo `$!"
$pidText = (Invoke-WslBash $startCmd | Select-Object -Last 1).ToString().Trim()
Write-Host "  nativelink pid (WSL) = $pidText"

# Wait until the public port accepts TCP from Windows.
$deadline = (Get-Date).AddMinutes(2)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(1000, $false)
        if ($ok -and $client.Connected) {
            $ready = $true
            $client.Close()
            break
        }
        $client.Close()
    } catch {
        # retry
    }
    Start-Sleep -Seconds 1
}

if (-not $ready) {
    Write-Host "Worker did not open 127.0.0.1:$Port in time. Log tail:"
    if (Test-Path $WorkerLog) {
        Get-Content $WorkerLog -Tail 80
    } else {
        try { Invoke-WslBash "tail -n 80 '$WslLog' || true" } catch { }
    }
    throw "WSL NativeLink worker not reachable on 127.0.0.1:$Port"
}

Write-Host "WSL RE worker is up on grpc://127.0.0.1:$Port"
Write-Host "Set CROSS_REMOTE_BAZEL_FLAGS before test_cross_all.sh:"
Write-Host "  `$env:CROSS_REMOTE_BAZEL_FLAGS = '--remote_executor=grpc://127.0.0.1:$Port --remote_default_exec_properties=OSFamily=Linux'"
