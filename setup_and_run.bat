@echo off
setlocal
cd /d "%~dp0"
title GazePlay Analyzer

set "PY=python"
where py >nul 2>nul
if %errorlevel%==0 set "PY=py -3"

if exist ".venv\installed.ok" goto run

echo.
echo  First-time setup: installing GazePlay Analyzer. This takes 5-15 minutes
echo  and needs an internet connection. Later starts are instant.
echo.
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" 2>nul
if errorlevel 1 goto nopython

if exist ".venv" rmdir /s /q ".venv"
%PY% -m venv .venv
if errorlevel 1 goto nopython
".venv\Scripts\python.exe" -m pip install --upgrade pip
echo.
echo  Installing PyTorch - CPU version...
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
echo.
echo  Installing the other components...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo.
echo  Downloading the detection models...
".venv\Scripts\python.exe" setup_models.py
if errorlevel 1 goto fail
echo ok> ".venv\installed.ok"

:run
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
goto :eof

:nopython
echo.
echo  Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/
echo  and tick "Add python.exe to PATH" in the installer, then run this file again.
echo.
pause
goto :eof

:fail
echo.
echo  Setup did not finish. Check the internet connection and run this file again.
echo.
pause
