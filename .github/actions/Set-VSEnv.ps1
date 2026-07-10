param (
    [parameter(Mandatory = $false)]
    [ValidateSet(2022, 2019, 2017)][int]$Version = 2022,

    [parameter(Mandatory = $false)]
    [ValidateSet("all", "x86", "x64")][String]$Arch = "x64",

    # Pin a specific MSVC toolset prefix, or an ordered comma-separated list of
    # prefixes (major.minor, e.g. "14.44,14.43,14.42,14.41,14.40"). CUDA
    # toolkits lag behind the newest MSVC, so we must not let Bazel auto-detect
    # the latest VS on the runner (e.g. VS 2026 / MSVC 14.51 on the windows-2025
    # image, which nvcc's frontend cannot parse). Defaults to VS 2022 14.4x
    # candidates; pass "" to opt out and use the VS default toolset (not
    # recommended in CI).
    [parameter(Mandatory = $false)]
    [string]$ToolsetVersion = "14.44,14.43,14.42,14.41,14.40"
)

$ErrorActionPreference = "Stop"

# VS component id (for `setup.exe --add`) per pinned toolset. Extend as needed.
$toolset_component = @{
    "14.40" = "Microsoft.VisualStudio.Component.VC.14.40.17.10.x86.x64"
    "14.41" = "Microsoft.VisualStudio.Component.VC.14.41.17.11.x86.x64"
    "14.42" = "Microsoft.VisualStudio.Component.VC.14.42.17.12.x86.x64"
    "14.43" = "Microsoft.VisualStudio.Component.VC.14.43.17.13.x86.x64"
    "14.44" = "Microsoft.VisualStudio.Component.VC.14.44.17.14.x86.x64"
}

function Get-InstalledMsvcToolsets {
    [CmdletBinding()]
    param([string]$InstallPath)

    $msvc_dir = Join-Path $InstallPath "VC\Tools\MSVC"
    Get-ChildItem $msvc_dir -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
}

function Set-EnvFromCmdSet {
    [CmdletBinding()]
    param(
        [Parameter(ValueFromPipeline)]
        [string]$CmdSetResult
    )
    process {
        if ($CmdSetResult -Match "=") {
            $i = $CmdSetResult.IndexOf("=")
            $k = $CmdSetResult.Substring(0, $i)
            $v = $CmdSetResult.Substring($i + 1)
            Set-Item -Force -Path "Env:\$k" -Value "$v"
        }
    }
}

$vs_where = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'

$version_range = switch ($Version) {
    2022 { '[17,18)' }
    2019 { '[16,17)' }
    2017 { '[15,16)' }
}
$info = &$vs_where -version $version_range -format json | ConvertFrom-Json
$install_path = $info ? $info[0].installationPath : $null

if ($null -eq $install_path) {
    Write-Host -ForegroundColor Red "Visual Studio $Version is not installed."
    exit 1
}

# Ensure the pinned toolset is installed; CI images may only ship the newest.
$selected_toolset = ""
if ($ToolsetVersion) {
    $requested_toolsets = $ToolsetVersion.Split(",") |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }

    $installed_toolsets = @(Get-InstalledMsvcToolsets -InstallPath $install_path)
    foreach ($candidate in $requested_toolsets) {
        $match = $installed_toolsets |
            Where-Object { $_.Name.StartsWith("$candidate.") } |
            Select-Object -First 1
        if ($match) {
            $selected_toolset = $candidate
            break
        }
    }

    # If exactly one toolset was requested, try to install it. For candidate
    # lists, fail with diagnostics instead of guessing which version to install.
    if (-not $selected_toolset -and $requested_toolsets.Count -eq 1) {
        $component = $toolset_component[$requested_toolsets[0]]
        if (-not $component) {
            Write-Host -ForegroundColor Red "No known VS component id for MSVC toolset $($requested_toolsets[0]); add it to `$toolset_component."
            exit 1
        }
        Write-Host "MSVC toolset $($requested_toolsets[0]) not found; installing $component ..."
        $installer = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe'
        & $installer modify --installPath "$install_path" --add $component --quiet --norestart --nocache --wait
        if ($LASTEXITCODE -notin @(0, 3010)) {
            Write-Host -ForegroundColor Red "Failed to install MSVC toolset $($requested_toolsets[0]) (exit $LASTEXITCODE)."
            exit 1
        }
        $installed_toolsets = @(Get-InstalledMsvcToolsets -InstallPath $install_path)
        $match = $installed_toolsets |
            Where-Object { $_.Name.StartsWith("$($requested_toolsets[0]).") } |
            Select-Object -First 1
        if ($match) {
            $selected_toolset = $requested_toolsets[0]
        }
    }

    if (-not $selected_toolset) {
        $available = if ($installed_toolsets) { ($installed_toolsets.Name -join ", ") } else { "<none>" }
        Write-Host -ForegroundColor Red "None of the requested MSVC toolsets were found: $($requested_toolsets -join ', '). Available toolsets: $available"
        exit 1
    }
}

$vc_script = switch ($Arch) {
    "all" { 'Common7\Tools\VsDevCmd.bat' }
    "x64" { 'VC\Auxiliary\Build\vcvars64.bat' }
    "x86" { 'VC\Auxiliary\Build\vcvars32.bat' }
}
$path = Join-Path $install_path $vc_script
$vcvars_arg = if ($selected_toolset) { " -vcvars_ver=$selected_toolset" } else { "" }

C:/Windows/System32/cmd.exe /c "`"$path`"$vcvars_arg & set" | Set-EnvFromCmdSet

if ($selected_toolset -and (-not $env:VCToolsVersion -or -not $env:VCToolsVersion.StartsWith("$selected_toolset."))) {
    Write-Host -ForegroundColor Red "Requested MSVC toolset $selected_toolset but vcvars selected '$env:VCToolsVersion'."
    exit 1
}

Set-Item -Force -Path "Env:\BAZEL_VC" -Value "$env:VCINSTALLDIR"

# Persist to subsequent workflow steps. `bazelisk build` runs in separate shells,
# so the process-local env vars set above would be lost and Bazel would fall back
# to auto-detecting the newest (unsupported) VS. Writing BAZEL_VC(_FULL_VERSION)
# to $GITHUB_ENV pins Bazel's MSVC toolchain to the toolset selected here.
if ($env:GITHUB_ENV) {
    "BAZEL_VC=$env:VCINSTALLDIR" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
    if ($env:VCToolsVersion) {
        "BAZEL_VC_FULL_VERSION=$env:VCToolsVersion" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
    }
}

Write-Host -ForegroundColor Green "Visual Studio $Version $Arch (MSVC $env:VCToolsVersion) environment set."
