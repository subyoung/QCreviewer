#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Cleaning old build artifacts..."
rm -rf build dist __pycache__

echo "Checking required files..."
if [ ! -f "QCreviewer.py" ]; then
  echo "Error: QCreviewer.py not found in current folder."
  exit 1
fi

if [ ! -f "QCreviewer.spec" ]; then
  echo "Error: QCreviewer.spec not found in current folder."
  exit 1
fi

if [ ! -f "app.icns" ]; then
  echo "Warning: app.icns not found. The app may build without a proper macOS icon."
fi

echo "Running PyInstaller..."
pyinstaller --noconfirm QCreviewer.spec

echo ""
echo "Build finished."
echo "Check: dist/QCreviewer.app"