@echo off
setlocal
cd /d "%~dp0"
echo Starting Kestrel JSON bridge...
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_bridge_window.ps1"
echo.
echo Exit code: %ERRORLEVEL%
echo.
pause
