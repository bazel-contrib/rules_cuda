# Start NativeLink under WSL for Windows-host cross (cases 3–4).
#
# Default WSL2 NAT keeps working DNS inside the distro. Windows reaches the
# worker via:
#   1) netsh portproxy 127.0.0.1:<port> -> <wsl-ip>:<port>  (preferred)
#   2) direct grpc://<wsl-ip>:<port>                       (fallback)
#
# Do not force networkingMode=mirrored here: on GitHub Actions runners that
# mode often breaks WSL DNS (cannot resolve archive.ubuntu.com).
#
# Usage:
#   .\tests\integration\rbe\start_wsl_worker.ps1
#   # sets $env:CROSS_REMOTE_BAZEL_FLAGS and $env:CROSS_REMOTE_HOST
#   bash tests/integration/test_cross_all.sh --required-only --no-linux

$ErrorActionPreference = "Stop"

$Port = if ($env:RBE_PORT) { [int]$env:RBE_PORT } else { 1985 }
$Distro = if ($env:WSL_DISTRO) { $env:WSL_DISTRO } else { $null }
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
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

function Get-WslIpv4 {
    $raw = Invoke-WslBash "hostname -I 2>/dev/null || ip -4 -o addr show scope global | awk '{print `$4}' | cut -d/ -f1"
    $ips = @($raw -split '\s+' | Where-Object { $_ -match '^\d+\.\d+\.\d+\.\d+$' -and $_ -notmatch '^127\.' })
    if ($ips.Count -gt 0) { return $ips[0] }
    return $null
}

function Enable-LocalhostPortProxy {
    param(
        [Parameter(Mandatory = $true)][string]$ConnectAddress,
        [Parameter(Mandatory = $true)][int]$PortNumber
    )
    # Map Windows loopback -> WSL NAT address so Bazel can use 127.0.0.1
    # without WSL mirrored networking (which breaks DNS on some runners).
    Write-Host "Configuring portproxy 127.0.0.1:$PortNumber -> ${ConnectAddress}:$PortNumber"
    & netsh interface portproxy delete v4tov4 listenaddress=127.0.0.1 listenport=$PortNumber 2>$null | Out-Null
    & netsh interface portproxy add v4tov4 `
        listenaddress=127.0.0.1 listenport=$PortNumber `
        connectaddress=$ConnectAddress connectport=$PortNumber
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARN: netsh portproxy add failed (exit $LASTEXITCODE); will use WSL IP directly"
        return $false
    }
    # Allow inbound on the listen port (no-op if rule exists / insufficient rights).
    & netsh advfirewall firewall delete rule name="rules_cuda WSL RE $PortNumber" 2>$null | Out-Null
    & netsh advfirewall firewall add rule `
        name="rules_cuda WSL RE $PortNumber" `
        dir=in action=allow protocol=TCP localport=$PortNumber 2>$null | Out-Null
    return $true
}

# Ensure distro is up.
Invoke-WslBash "true"

$WslRepo = (Invoke-Wsl -ArgumentList @("-e", "wslpath", "-a", $RepoRoot) | Select-Object -Last 1).ToString().Trim()
$SetupDeps = "$WslRepo/tests/integration/rbe/setup_wsl_cross_deps.sh"
$Bootstrap = "$WslRepo/tests/integration/rbe/bootstrap_nativelink_linux.sh"

Write-Host "WSL NativeLink worker (cases 3–4)"
Write-Host "  repo (Windows) = $RepoRoot"
Write-Host "  repo (WSL)     = $WslRepo"
Write-Host "  worker log     = $WslLog (inside WSL)"

# Sanity: DNS must work inside WSL for apt/curl (mirrored mode often breaks this).
try {
    Invoke-WslBash "getent hosts archive.ubuntu.com >/dev/null 2>&1 || getent hosts github.com >/dev/null 2>&1 || (echo 'WARN: WSL DNS probe failed' >&2; cat /etc/resolv.conf || true)"
} catch {
    Write-Host "WARN: WSL DNS probe threw; continuing"
}

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
if (-not $wslIp) {
    try { Invoke-WslBash "ip -br a || true; tail -n 40 '$WslLog' || true" } catch { }
    throw "Could not determine WSL IPv4 address"
}

$portproxyOk = Enable-LocalhostPortProxy -ConnectAddress $wslIp -PortNumber $Port

# Probe localhost (via portproxy) first, then the WSL NAT address.
$candidates = @()
if ($portproxyOk) { $candidates += "127.0.0.1" }
$candidates += $wslIp

$readyHost = $null
$deadline = (Get-Date).AddMinutes(2)
while ((Get-Date) -lt $deadline -and -not $readyHost) {
    try {
        $logTail = (Invoke-WslBash "tail -n 8 '$WslLog' 2>/dev/null || true" | Out-String)
        if ($logTail -match 'Ready, listening|listening on') {
            foreach ($h in $candidates) {
                if (Test-TcpPort -HostName $h -PortNumber $Port -TimeoutMs 800) {
                    $readyHost = $h
                    break
                }
            }
        } elseif ($logTail -match 'error|Error|panic|failed') {
            Write-Host "NativeLink log may indicate failure:"
            Write-Host $logTail
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
    try { & netsh interface portproxy show all } catch { }
    throw "WSL NativeLink worker not reachable on port $Port"
}

$endpoint = "grpc://${readyHost}:$Port"
$env:CROSS_REMOTE_HOST = $readyHost
$env:CROSS_REMOTE_BAZEL_FLAGS = @(
    "--remote_executor=$endpoint",
    "--remote_default_exec_properties=OSFamily=Linux",
    "--remote_timeout=600"
) -join " "

$via = if ($readyHost -eq "127.0.0.1") { "localhost portproxy -> $wslIp" } else { "WSL NAT IP" }
Write-Host "WSL RE worker is up on $endpoint (via $via)"
Write-Host "CROSS_REMOTE_BAZEL_FLAGS=$env:CROSS_REMOTE_BAZEL_FLAGS"
