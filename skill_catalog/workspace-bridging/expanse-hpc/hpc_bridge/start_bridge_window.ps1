$ErrorActionPreference = "Stop"

$bridgeDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $bridgeDir ".venv\Scripts\python.exe"
$requirements = Join-Path $bridgeDir "requirements.txt"

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating bridge virtual environment..."
    python -m venv (Join-Path $bridgeDir ".venv")
}

Write-Host "Ensuring bridge Python dependencies are installed..."
& $venvPython -m pip install --disable-pip-version-check --quiet -r $requirements

Write-Host ""
Write-Host "Starting Expanse bridge for the runtime target in bridge_identity.json."
Write-Host "Enter password and TOTP only in this window when prompted."
Write-Host ""
Set-Location $bridgeDir
& $venvPython .\bridge_server.py

Write-Host ""
Write-Host "Bridge exited."
Pause
