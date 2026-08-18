# Drive case 3 on a Windows host with Linux exec via WSL-native NativeLink.
#
#   ┌─ Windows bazelisk ──grpc://<host>:1985──► ┌─ WSL Ubuntu ─────────┐
#   │  host = windows-x86_64                    │  NativeLink RE       │
#   │  target = linux-sbsa                      │  exec = linux-x86_64 │
#   └───────────────────────────────────────────┴──────────────────────┘
# Endpoint is 127.0.0.1 via netsh portproxy (default WSL NAT), else WSL eth IPv4.

$ErrorActionPreference = "Stop"

$RepoRoot = if ($env:RULES_CUDA_ROOT) {
    $env:RULES_CUDA_ROOT
} else {
    (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
Set-Location $RepoRoot

$Port = if ($env:RBE_PORT) { [int]$env:RBE_PORT } else { 1985 }
$CudaVersion = if ($env:CUDA_REDIST_VERSION_OVERRIDE) {
    $env:CUDA_REDIST_VERSION_OVERRIDE
} else {
    "12.6.3"
}
$env:CUDA_REDIST_VERSION_OVERRIDE = $CudaVersion

if (-not $env:USE_BAZEL_VERSION -and (Test-Path ".bazelversion")) {
    $env:USE_BAZEL_VERSION = (Get-Content ".bazelversion" -Raw).Trim()
}

# Git-bash for the integration driver (test_cross_all.sh).
if (-not $env:BAZEL_SH) {
    $bashCandidates = @(
        "C:\Program Files\Git\bin\bash.exe",
        "C:\Program Files\Git\usr\bin\bash.exe"
    )
    foreach ($b in $bashCandidates) {
        if (Test-Path $b) {
            $env:BAZEL_SH = $b
            break
        }
    }
}
if (-not $env:BAZEL_SH) {
    throw "BAZEL_SH / Git bash not found (needed to run test_cross_all.sh on Windows)"
}

Write-Host "=== drive_cross_windows.ps1 (case 3: Windows host / WSL RE) ==="
Write-Host "  root=$RepoRoot cuda=$CudaVersion port=$Port"

& (Join-Path $RepoRoot "tests\integration\rbe\start_wsl_worker.ps1")

if (-not $env:CROSS_REMOTE_BAZEL_FLAGS) {
    throw "start_wsl_worker.ps1 did not set CROSS_REMOTE_BAZEL_FLAGS"
}
Write-Host "CROSS_REMOTE_BAZEL_FLAGS=$env:CROSS_REMOTE_BAZEL_FLAGS"

$driverWin = Join-Path $RepoRoot "tests\integration\test_cross_all.sh"
# Normalize CRLF for bash (path as Git-bash understands it).
$driverBash = ($driverWin -replace '\\', '/')
if ($driverBash -match '^([A-Za-z]):/') {
    $driverBash = "/$($Matches[1].ToLower())/$($driverBash.Substring(3))"
}
& $env:BAZEL_SH -lc "sed -i 's/\r`$//' '$driverBash' && chmod +x '$driverBash'"

Push-Location (Join-Path $RepoRoot "tests\integration")
try {
    & $env:BAZEL_SH ./test_cross_all.sh --required-only --no-linux
    if ($LASTEXITCODE -ne 0) {
        throw "test_cross_all.sh failed with exit $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host "=== case 3 finished ==="
