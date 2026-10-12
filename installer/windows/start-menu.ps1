# Create a Start menu shortcut in the shared VOXPrint AI Suite folder.
# The folder is the same one Audiobook Builder and Movie Dubber use.
# This script does not install a runtime. It only points at a checkout.
# The interpreter is interpreter.json under the shared python pointer, then
# py -3. The Microsoft Store alias in WindowsApps is not used.
param(
    [string]$Target = (Join-Path $PSScriptRoot "..\..\main.py")
)
$ErrorActionPreference = "Stop"

function Get-InterpreterExecutable([string]$PointerDir) {
    $pointer = Join-Path $PointerDir "interpreter.json"
    if (-not (Test-Path -LiteralPath $pointer)) {
        return ""
    }
    try {
        $payload = Get-Content -LiteralPath $pointer -Raw | ConvertFrom-Json
    } catch {
        return ""
    }
    $version = [string]$payload.version
    $executable = [string]$payload.executable
    if ($version -ne "3.14" -and -not $version.StartsWith("3.14.")) {
        return ""
    }
    if ($executable -and (Test-Path -LiteralPath $executable)) {
        return $executable
    }
    return ""
}

function Add-PointerDirs([System.Collections.Generic.List[string]]$Dirs, [string]$Shared) {
    if (-not (Test-Path -LiteralPath $Shared)) {
        return
    }
    $manifest = Join-Path $Shared "manifest.json"
    if (Test-Path -LiteralPath $manifest) {
        try {
            $document = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
            foreach ($item in $document.components.PSObject.Properties) {
                $path = [string]$item.Value.path
                if ($path -like "python*") {
                    $Dirs.Add((Join-Path $Shared $path))
                }
            }
        } catch {
            # A damaged manifest still leaves the python-* folders below.
        }
    }
    foreach ($dir in Get-ChildItem -LiteralPath $Shared -Directory -ErrorAction SilentlyContinue) {
        if ($dir.Name -eq "python" -or $dir.Name.StartsWith("python-")) {
            $Dirs.Add($dir.FullName)
        }
    }
}

function Resolve-VoxprintPython {
    $dirs = New-Object System.Collections.Generic.List[string]
    if ($env:VOXPRINT_HOME) {
        Add-PointerDirs $dirs (Join-Path $env:VOXPRINT_HOME "shared")
    }
    if ($env:LOCALAPPDATA) {
        Add-PointerDirs $dirs (Join-Path $env:LOCALAPPDATA "Voxprint\shared")
    }
    foreach ($dir in $dirs) {
        $fromJson = Get-InterpreterExecutable $dir
        if ($fromJson) {
            Write-Host "python: $fromJson"
            return @{ Path = $fromJson; Arguments = "" }
        }
        foreach ($name in @("pythonw.exe", "python.exe")) {
            $candidate = Join-Path $dir $name
            if (Test-Path -LiteralPath $candidate) {
                Write-Host "python: $candidate"
                return @{ Path = $candidate; Arguments = "" }
            }
        }
    }
    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($py) {
        Write-Host "python: py -3"
        return @{ Path = $py.Source; Arguments = "-3" }
    }
    throw "No shared runtime or py -3 interpreter was found."
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
