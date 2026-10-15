@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0baby-monitor-home.ps1" -Action dashboard-start
if errorlevel 1 (
  echo.
  echo Baby Monitor Home Control failed to start.
  pause
)
endlocal
