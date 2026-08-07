param (
    [parameter(Mandatory = $false)]
    [ValidateSet(2022, 2019, 2017)][int]$Version = 2019,

    [parameter(Mandatory = $false)]
    [ValidateSet("all", "x86", "x64")][String]$Arch = "x64"
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

$vs_where = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'

$version_range = switch ($Version) {
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
    return
}

$path = Join-Path $vs_env.install_path $vs_env.$Arch

C:/Windows/System32/cmd.exe /c "`"$path`" & set" | Set-EnvFromCmdSet
Set-Item -Force -Path "Env:\BAZEL_VC" -Value "$env:VCINSTALLDIR"
Write-Host -ForegroundColor Green "Visual Studio $Version $Arch Command Prompt variables set."
