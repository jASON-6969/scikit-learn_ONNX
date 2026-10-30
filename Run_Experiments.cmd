@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_team_experiments.ps1"
set "run_result=%ERRORLEVEL%"
echo.
if not "%run_result%"=="0" echo The workflow stopped. Keep the output folders and logs. See the recovery instructions.
pause
exit /b %run_result%
