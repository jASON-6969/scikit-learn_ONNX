@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo Run Setup_Environment.cmd first.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -B -X utf8 "%~dp0scripts\package_team_runner.py" --results-only
set "collect_result=%ERRORLEVEL%"
echo.
pause
exit /b %collect_result%
