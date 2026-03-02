@echo off
cd /d "%~dp0"
py -3 -V >nul 2>&1
if errorlevel 1 (
  python index.py
) else (
  py -3 index.py
)
if errorlevel 1 (
  echo.
  echo Focus Assist exited with an error.
  echo If this mentions missing Python, reinstall Python and enable PATH during install.
  pause
)
