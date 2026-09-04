$ErrorActionPreference = "Stop"
$winRoot = "C:\Users\cloud\workspaces\rules_cuda"
$wslRoot = (wsl -e wslpath -a $winRoot).Trim()
$script = "$wslRoot/tests/integration/drive_cross_wsl.sh"
$log = "/tmp/rules_cuda_cross/drive.log"

# Normalize CRLF -> LF for the bash script
wsl -e bash -lc "sed -i 's/\r$//' '$script' && chmod +x '$script'"

Write-Host "Running drive_cross_wsl.sh under WSL (root=$wslRoot)"
# Long-running: no timeout at outer level beyond bazel downloads
wsl -e bash -lc "export LOG_DIR=/tmp/rules_cuda_cross; mkdir -p `$LOG_DIR; bash '$script' 2>&1 | tee `$LOG_DIR/drive.log; exit `${PIPESTATUS[0]}"
exit $LASTEXITCODE
