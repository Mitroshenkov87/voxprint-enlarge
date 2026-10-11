# Create a Start menu shortcut in the shared VOXPrint AI Suite folder.
# The folder is the same one Audiobook Builder and Movie Dubber use.
# This script does not install a runtime. It only points at a checkout.
# The interpreter is the shared runtime, then py -3, then python3.
# python3 on Windows is often the Microsoft Store alias, so it is last.
param(
    [string]$Target = (Join-Path $PSScriptRoot "..\..\main.py")
)
$ErrorActionPreference = "Stop"

function Resolve-VoxprintPython {
    $candidates = @()
    if ($env:VOXPRINT_HOME) {
        $candidates += Join-Path $env:VOXPRINT_HOME "shared\python\pythonw.exe"
        $candidates += Join-Path $env:VOXPRINT_HOME "shared\python\python.exe"
    }
    if ($env:LOCALAPPDATA) {
        $candidates += Join-Path $env:LOCALAPPDATA "Voxprint\shared\python\pythonw.exe"
        $candidates += Join-Path $env:LOCALAPPDATA "Voxprint\shared\python\python.exe"
    }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            Write-Host "python: $candidate"
            return @{ Path = $candidate; Arguments = "" }
        }
    }
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        Write-Host "python: py -3"
        return @{ Path = $py.Source; Arguments = "-3" }
    }
    $python3 = Get-Command python3.exe -ErrorAction SilentlyContinue
    if ($python3 -and $python3.Source -notlike "*\WindowsApps\*") {
        Write-Host "python: python3"
        return @{ Path = $python3.Source; Arguments = "" }
    }
    throw "No shared runtime, py -3, or python3 interpreter was found."
}

$folder = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\VOXPrint AI Suite"
New-Item -ItemType Directory -Force -Path $folder | Out-Null
$shortcut = Join-Path $folder "VOXPrint AI Enlarge.lnk"
$python = Resolve-VoxprintPython
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($shortcut)
$link.TargetPath = $python.Path
$script = (Resolve-Path $Target).Path
if ($python.Arguments) {
    $link.Arguments = "$($python.Arguments) `"$script`""
} else {
    $link.Arguments = "`"$script`""
}
$link.WorkingDirectory = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$icon = Join-Path $link.WorkingDirectory "assets\enlarge.ico"
if (Test-Path $icon) { $link.IconLocation = $icon }
$link.Description = "VOXPrint AI Enlarge"
$link.Save()
Write-Output $shortcut
