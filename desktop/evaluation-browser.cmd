@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH="
set "PYTHONHOME="
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
"%~dp0runtime\python\python.exe" -I -B "%~dp0evaluation-browser.py" %*
if errorlevel 1 pause
endlocal
