#!/usr/bin/env bash
# XYZT - desktop app (macOS / Linux)
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || python3 -m venv .venv || { echo "Could not create the environment (install Python 3.10-3.12)."; exit 1; }
if ! .venv/bin/python -c "import numpy, cv2, flask, scenedetect, onnx, onnxruntime, imageio_ffmpeg" 2>/dev/null; then
  echo "Installing dependencies. The first time takes a few minutes..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt || { echo "Installing dependencies failed. Python in the environment: $(.venv/bin/python --version)"; exit 1; }
fi
exec .venv/bin/python app.py "$@"
