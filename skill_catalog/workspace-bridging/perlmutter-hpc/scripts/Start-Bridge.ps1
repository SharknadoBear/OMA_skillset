param([Parameter(Mandatory=$true)][string]$Session)
$ErrorActionPreference = 'Stop'
$skillRoot = Split-Path $PSScriptRoot -Parent
$bridgePython = Join-Path $skillRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $bridgePython)) { throw 'Run scripts/Setup.ps1 first.' }
$sessionPath = (Resolve-Path -LiteralPath $Session).Path
$Host.UI.RawUI.WindowTitle = 'Perlmutter | secure local authentication'
& $bridgePython (Join-Path $PSScriptRoot 'bridge.py') serve --session $sessionPath
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Bridge stopped. Review the error and session status before retrying.'
    Read-Host 'Press Enter to close'
}
