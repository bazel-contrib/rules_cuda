# Start NativeLink under WSL for a Windows-host cross-compilation test.
#
# Default WSL2 NAT keeps working DNS inside the distro. Windows reaches the
# worker via:
#   1) netsh portproxy 127.0.0.1:<port> -> <wsl-ip>:<port>
#   2) direct grpc://<wsl-ip>:<port> if portproxy is unavailable
#
# Do not force networkingMode=mirrored here: on GitHub Actions runners that
# mode often breaks WSL DNS (cannot resolve archive.ubuntu.com).
#
# Also: do not start NativeLink with `wsl bash -lc 'nohup ... &'`. When that
# short-lived shell exits, WSL may shut the distro down, kill the worker, and
# wipe /tmp (log disappears; port never opens). Keep a sleep-infinity session
# and launch NativeLink via Start-Process so Windows owns the lifetime.
#
# Usage:
#   .\tests\integration\rbe\start_wsl_worker.ps1
#   # sets $env:CROSS_REMOTE_BAZEL_FLAGS and $env:CROSS_REMOTE_HOST
#   bash tests/integration/test_cross_all.sh

$ErrorActionPreference = "Stop"

$Port = if ($env:RBE_PORT) { [int]$env:RBE_PORT } else { 1985 }
$Distro = if ($env:WSL_DISTRO) { $env:WSL_DISTRO } else { $null }
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
# Durable path inside WSL (survives /tmp wipe if the distro restarts).
$WslLog = '$HOME/.cache/rules_cuda-rbe/nativelink.log'

function Get-WslPrefix {
    $prefix = @()
    if ($Distro) { $prefix += @("-d", $Distro) }
    return $prefix
}

function Invoke-Wsl {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList
    )
    & wsl (@(Get-WslPrefix) + $ArgumentList)
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
$WorkerRun = "$WslRepo/tests/integration/rbe/run_nativelink_worker.sh"

Write-Host "WSL NativeLink worker"
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
Invoke-WslBash "sed -i 's/\r`$//' '$SetupDeps' '$Bootstrap' '$WorkerRun' && chmod +x '$SetupDeps' '$Bootstrap' '$WorkerRun'"

Write-Host "Installing WSL cross-compile deps..."
Invoke-WslBash "bash '$SetupDeps'"

# Stop any previous worker on this port (best effort).
try {
    Invoke-WslBash "fuser -k ${Port}/tcp >/dev/null 2>&1 || true; fuser -k 1986/tcp >/dev/null 2>&1 || true; sleep 1"
} catch {
    # ignore cleanup failures
}

# Hold the distro open: a short-lived `wsl ... nohup &` session exiting can
# shut WSL down and kill the worker (CI symptom: missing log + closed port).
Write-Host "Starting WSL keep-alive (sleep infinity)..."
$keeperArgStr = if ($Distro) { "-d $Distro -e sleep infinity" } else { "-e sleep infinity" }
$script:WslKeeperProc = Start-Process -FilePath "wsl.exe" -ArgumentList $keeperArgStr -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 2

Write-Host "Starting NativeLink in WSL (Windows-owned process)..."
# Pass a simple -lc string; the worker script owns logging + exec.
$nlArgStr = if ($Distro) {
    "-d $Distro -e bash -lc `"export RBE_PORT=$Port; exec bash '$WorkerRun'`""
} else {
    "-e bash -lc `"export RBE_PORT=$Port; exec bash '$WorkerRun'`""
}
$script:WslNativeLinkProc = Start-Process -FilePath "wsl.exe" -ArgumentList $nlArgStr -PassThru -WindowStyle Hidden
Write-Host "  wsl keeper pid (Windows)     = $($script:WslKeeperProc.Id)"
Write-Host "  wsl nativelink pid (Windows) = $($script:WslNativeLinkProc.Id)"

$wslIp = Get-WslIpv4
Write-Host "  WSL IPv4                     = $wslIp"
if (-not $wslIp) {
    try { Invoke-WslBash "ip -br a || true; tail -n 40 $WslLog || true" } catch { }
    throw "Could not determine WSL IPv4 address"
}

$portproxyOk = Enable-LocalhostPortProxy -ConnectAddress $wslIp -PortNumber $Port

# Probe localhost (via portproxy) first, then the WSL NAT address.
$candidates = @()
if ($portproxyOk) { $candidates += "127.0.0.1" }
$candidates += $wslIp

$readyHost = $null
# First launch may download the NativeLink musl tarball.
$deadline = (Get-Date).AddMinutes(4)
while ((Get-Date) -lt $deadline -and -not $readyHost) {
    $script:WslNativeLinkProc.Refresh()
    if ($script:WslNativeLinkProc.HasExited) {
        Write-Host "NativeLink WSL process exited early (code $($script:WslNativeLinkProc.ExitCode)). Log tail:"
        try { Invoke-WslBash "tail -n 80 $WslLog || true; ls -la `$HOME/.cache/rules_cuda-rbe || true" } catch { }
        throw "WSL NativeLink process exited before becoming ready"
    }
    try {
        $logTail = (Invoke-WslBash "tail -n 12 $WslLog 2>/dev/null || true" | Out-String)
        if ($logTail -match 'Ready, listening|listening on|Starting nativelink') {
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
        # Also probe TCP even before a ready log line (download still in progress).
        if (-not $readyHost) {
            foreach ($h in $candidates) {
                if (Test-TcpPort -HostName $h -PortNumber $Port -TimeoutMs 400) {
                    $readyHost = $h
                    break
                }
            }
        }
    } catch {
        # retry
    }
    if (-not $readyHost) { Start-Sleep -Seconds 2 }
}

if (-not $readyHost) {
    Write-Host "Worker not reachable from Windows. Diagnostics:"
    try {
        Invoke-WslBash "echo '--- log ---'; tail -n 80 $WslLog || echo '(no log)'; echo '--- listen ---'; ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null || true; echo '--- procs ---'; ps -ef | grep -E '[n]ativelink|[b]ootstrap_nativelink' || true"
    } catch { }
    Write-Host "Candidates tried: $($candidates -join ', ')"
    Write-Host "NativeLink process HasExited=$($script:WslNativeLinkProc.HasExited) ExitCode=$($script:WslNativeLinkProc.ExitCode)"
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
