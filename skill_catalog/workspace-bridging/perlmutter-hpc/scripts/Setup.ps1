param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$skillRoot = Split-Path $PSScriptRoot -Parent
$bridgePython = Join-Path $skillRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $bridgePython)) {
    & $Python -m venv (Join-Path $skillRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}
& $bridgePython -m pip install -r (Join-Path $skillRoot 'requirements.lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
