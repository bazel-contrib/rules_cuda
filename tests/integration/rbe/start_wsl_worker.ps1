# Start NativeLink under WSL for Windows-host cross (cases 3–4).
#
#   Windows bazelisk  --remote_executor=grpc://<host>:<port>
#         │  host network (mirrored) or WSL eth IP (NAT fallback)
#         v
#   WSL Ubuntu (linux-x86_64)
#     NativeLink :1985 (public) / :1986 (worker)
#     case 3: x64 nvcc native; case 4: sbsa nvcc via qemu-user
#
# Usage:
#   .\tests\integration\rbe\start_wsl_worker.ps1
#   # sets $env:CROSS_REMOTE_BAZEL_FLAGS and $env:CROSS_REMOTE_HOST
#   bash tests/integration/test_cross_all.sh --required-only --no-linux

$ErrorActionPreference = "Stop"

$Port = if ($env:RBE_PORT) { [int]$env:RBE_PORT } else { 1985 }
$Distro = if ($env:WSL_DISTRO) { $env:WSL_DISTRO } else { $null }
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
# Keep logs inside WSL so Windows TEMP / short-path mapping cannot break them.
$WslLog = "/tmp/rules_cuda_rbe/nativelink.log"

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

function Test-TcpPort {
    param(
        [Parameter(Mandatory = $true)][string]$HostName,
        [Parameter(Mandatory = $true)][int]$PortNumber,
        [int]$TimeoutMs = 1000
    )
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect($HostName, $PortNumber, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne($TimeoutMs, $false)
        if ($ok -and $client.Connected) {
            $client.Close()
            return $true
        }
        $client.Close()
    } catch {
        # not ready
    }
    return $false
}

function Enable-WslHostNetworking {
    # Mirrored mode shares the host network stack so Windows localhost reaches
    # listeners inside WSL (default WSL2 NAT does not). Requires modern WSL.
    $wslConfig = Join-Path $env:USERPROFILE ".wslconfig"
    $desired = @"
[wsl2]
networkingMode=mirrored
dnsTunneling=true
autoProxy=true
"@
    $existing = if (Test-Path $wslConfig) { Get-Content -Raw $wslConfig } else { "" }
    if ($existing -notmatch 'networkingMode\s*=\s*mirrored') {
        Write-Host "Writing $wslConfig (networkingMode=mirrored)"
        Set-Content -Path $wslConfig -Value $desired -Encoding ascii
        Write-Host "Restarting WSL to apply host networking..."
        & wsl --shutdown
        Start-Sleep -Seconds 3
    } else {
        Write-Host "WSL host networking already configured ($wslConfig)"
    }
}

function Get-WslIpv4 {
    # First non-loopback IPv4 from the distro (used when localhost is not shared).
    $raw = Invoke-WslBash "hostname -I 2>/dev/null || ip -4 -o addr show scope global | awk '{print `$4}' | cut -d/ -f1"
    $ips = @($raw -split '\s+' | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' -and $_ -notmatch '^127\.' })
    if ($ips.Count -gt 0) { return $ips[0] }
    return $null
}

Enable-WslHostNetworking

# Touch the distro so it is running after a possible shutdown.
Invoke-WslBash "true"

$WslRepo = (Invoke-Wsl -ArgumentList @("-e", "wslpath", "-a", $RepoRoot) | Select-Object -Last 1).ToString().Trim()
$SetupDeps = "$WslRepo/tests/integration/rbe/setup_wsl_cross_deps.sh"
$Bootstrap = "$WslRepo/tests/integration/rbe/bootstrap_nativelink_linux.sh"

Write-Host "WSL NativeLink worker (cases 3–4)"
Write-Host "  repo (Windows) = $RepoRoot"
Write-Host "  repo (WSL)     = $WslRepo"
Write-Host "  worker log     = $WslLog (inside WSL)"

# Ensure LF line endings (Windows checkouts may be CRLF).
Invoke-WslBash "sed -i 's/\r`$//' '$SetupDeps' '$Bootstrap' && chmod +x '$SetupDeps' '$Bootstrap'"

Write-Host "Installing WSL cross-compile deps..."
Invoke-WslBash "bash '$SetupDeps'"

# Stop any previous worker on this port (best effort).
try {
    Invoke-WslBash "fuser -k ${Port}/tcp >/dev/null 2>&1 || true; fuser -k 1986/tcp >/dev/null 2>&1 || true; sleep 1"
} catch {
    # ignore cleanup failures
}

Write-Host "Starting NativeLink in WSL (background)..."
$startCmd = "mkdir -p /tmp/rules_cuda_rbe; export RBE_PORT=$Port; nohup bash '$Bootstrap' >'$WslLog' 2>&1 & echo `$!"
$pidText = (Invoke-WslBash $startCmd | Select-Object -Last 1).ToString().Trim()
Write-Host "  nativelink pid (WSL) = $pidText"

$wslIp = Get-WslIpv4
Write-Host "  WSL IPv4           = $wslIp"

# Probe localhost first (mirrored / host network), then the WSL NAT address.
$candidates = @("127.0.0.1")
if ($wslIp) { $candidates += $wslIp }

$readyHost = $null
$deadline = (Get-Date).AddMinutes(2)
while ((Get-Date) -lt $deadline -and -not $readyHost) {
    # Confirm the process is still alive / log has "Ready".
    try {
        $logTail = (Invoke-WslBash "tail -n 5 '$WslLog' 2>/dev/null || true" | Out-String)
        if ($logTail -match 'Ready, listening') {
            foreach ($h in $candidates) {
                if (Test-TcpPort -HostName $h -PortNumber $Port -TimeoutMs 800) {
                    $readyHost = $h
                    break
                }
            }
        }
    } catch {
        # retry
    }
    if (-not $readyHost) { Start-Sleep -Seconds 1 }
}

if (-not $readyHost) {
    Write-Host "Worker not reachable from Windows. Log tail:"
    try { Invoke-WslBash "tail -n 80 '$WslLog' || true" } catch { }
    Write-Host "Candidates tried: $($candidates -join ', ')"
    throw "WSL NativeLink worker not reachable on port $Port"
}

$endpoint = "grpc://${readyHost}:$Port"
$env:CROSS_REMOTE_HOST = $readyHost
$env:CROSS_REMOTE_BAZEL_FLAGS = @(
    "--remote_executor=$endpoint",
    "--remote_default_exec_properties=OSFamily=Linux",
    "--remote_timeout=600"
) -join " "

Write-Host "WSL RE worker is up on $endpoint (via $(if ($readyHost -eq '127.0.0.1') { 'host/mirrored network' } else { 'WSL NAT IP' }))"
Write-Host "CROSS_REMOTE_BAZEL_FLAGS=$env:CROSS_REMOTE_BAZEL_FLAGS"
