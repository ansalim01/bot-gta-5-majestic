param(
    [string]$ShortcutName = "GameBot Control"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$entryPoint = Join-Path $projectRoot "GameBot.pyw"
$iconPath = Join-Path $projectRoot "assets\gamebot.ico"

$pythonw = Get-Command pythonw.exe -ErrorAction Stop
$desktop = [Environment]::GetFolderPath([Environment+SpecialFolder]::Desktop)
$shortcutPath = Join-Path $desktop "$ShortcutName.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonw.Source
$shortcut.Arguments = "`"$entryPoint`""
$shortcut.WorkingDirectory = $projectRoot
$shortcut.IconLocation = "$iconPath,0"
$shortcut.Description = "GameBot Control"
$shortcut.Save()

Write-Output $shortcutPath
