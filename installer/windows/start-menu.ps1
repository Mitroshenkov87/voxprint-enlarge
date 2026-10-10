# Create a Start menu shortcut in the shared Voxprint folder.
# The folder is the same one Audiobook Builder and Movie Dubber use.
# This script does not install a runtime. It only points at a checkout.
param(
    [string]$Target = (Join-Path $PSScriptRoot "..\..\main.py")
)
$ErrorActionPreference = "Stop"
$folder = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Voxprint"
New-Item -ItemType Directory -Force -Path $folder | Out-Null
$shortcut = Join-Path $folder "Voxprint AI Enlarge.lnk"
$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($shortcut)
$link.TargetPath = (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Source
if (-not $link.TargetPath) {
    $link.TargetPath = (Get-Command python.exe).Source
}
$link.Arguments = "`"$((Resolve-Path $Target).Path)`""
$link.WorkingDirectory = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$icon = Join-Path $link.WorkingDirectory "assets\enlarge.ico"
if (Test-Path $icon) { $link.IconLocation = $icon }
$link.Description = "Voxprint AI Enlarge"
$link.Save()
Write-Output $shortcut
