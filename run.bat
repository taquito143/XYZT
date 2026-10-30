@echo off
REM XYZT - desktop app (Windows)
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"

if not exist ".venv\Scripts\python.exe" (
  echo Creating the Python environment...
  %PY% -m venv .venv
  if errorlevel 1 goto :novenv
)

".venv\Scripts\python.exe" -c "import numpy, cv2, flask, scenedetect, onnx, onnxruntime, imageio_ffmpeg" >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies. The first time takes a few minutes...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :noinstall
)

".venv\Scripts\python.exe" app.py %*
pause
exit /b 0

:novenv
echo.
echo Could not create the environment. Install Python 3.12 from python.org,
echo tick "Add python.exe to PATH", then run this file again.
pause
exit /b 1

:noinstall
echo.
echo Installing dependencies failed. Python version in the environment:
".venv\Scripts\python.exe" --version
echo If it is 3.13 or newer and the error mentions onnx or onnxruntime,
echo install Python 3.12, delete the .venv folder and run this file again.
pause
exit /b 1
