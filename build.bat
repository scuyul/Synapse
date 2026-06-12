@echo off
setlocal

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0build.py" %*
    exit /b %ERRORLEVEL%
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py "%~dp0build.py" %*
    exit /b %ERRORLEVEL%
)

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python "%~dp0build.py" %*
    exit /b %ERRORLEVEL%
)

echo Python was not found on PATH. Install Python or activate .venv first.
exit /b 1
