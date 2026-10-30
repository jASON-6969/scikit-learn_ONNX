@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_environment.ps1"
set "setup_result=%ERRORLEVEL%"
echo.
if not "%setup_result%"=="0" echo Setup failed. Keep the log shown above and send it to the group leader.
pause
exit /b %setup_result%
