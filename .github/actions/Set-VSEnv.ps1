param (
    [parameter(Mandatory = $false)]
    [ValidateSet(2026, 2022, 2019, 2017)][int]$Version = 2019,

    [parameter(Mandatory = $false)]
    [ValidateSet("all", "x86", "x64")][String]$Arch = "x64",

    # Optional MSVC toolset major.minor (e.g. "14.44") passed to vcvars as -vcvars_ver.
    # Required on VS 2026 images where the default toolset is too new for nvcc.
    [parameter(Mandatory = $false)]
    [string]$ToolsetVersion = ""
)

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

function Write-GithubEnvFile {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    if (-not $env:GITHUB_ENV) {
        return
    }
    $value = [System.Environment]::GetEnvironmentVariable($Name)
    if ($null -eq $value -or $value -eq "") {
        return
    }
    "${Name}=${value}" | Out-File -FilePath $env:GITHUB_ENV -Append -Encoding utf8
}

$vs_where = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'

$version_range = switch ($Version) {
    2026 { '[18,19)' }
    2022 { '[17,18)' }
    2019 { '[16,17)' }
    2017 { '[15,16)' }
}
$info = &$vs_where -version $version_range -format json | ConvertFrom-Json
$vs_env = @{
    install_path = $info ? $info[0].installationPath : $null
    all          = 'Common7\Tools\VsDevCmd.bat'
    x64          = 'VC\Auxiliary\Build\vcvars64.bat'
    x86          = 'VC\Auxiliary\Build\vcvars32.bat'
}

if ( $null -eq $vs_env.install_path) {
    Write-Host -ForegroundColor Red "Visual Studio $Version is not installed."
    exit 1
}

$path = Join-Path $vs_env.install_path $vs_env.$Arch
$vcvars_arg = if ($ToolsetVersion) { " -vcvars_ver=$ToolsetVersion" } else { "" }

C:/Windows/System32/cmd.exe /c "`"$path`"$vcvars_arg & set" | Set-EnvFromCmdSet
Set-Item -Force -Path "Env:\BAZEL_VC" -Value "$env:VCINSTALLDIR"
if ($env:VCToolsVersion) {
    Set-Item -Force -Path "Env:\BAZEL_VC_FULL_VERSION" -Value "$env:VCToolsVersion"
}
Write-GithubEnvFile -Name "BAZEL_VC"
Write-GithubEnvFile -Name "BAZEL_VC_FULL_VERSION"
Write-Host -ForegroundColor Green "Visual Studio $Version $Arch Command Prompt variables set (MSVC $env:VCToolsVersion)."
