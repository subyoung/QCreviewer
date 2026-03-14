#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Cleaning old build artifacts..."
rm -rf build dist __pycache__

echo "Checking required files..."
if [ ! -f "QCreviewer_beta.py" ]; then
  echo "Error: QCreviewer_beta.py not found in current folder."
  exit 1
fi

if [ ! -f "QCreviewer_beta.spec" ]; then
  echo "Error: QCreviewer_beta.spec not found in current folder."
  exit 1
fi

if [ ! -f "app.icns" ]; then
  echo "Warning: app.icns not found. The app may build without a proper macOS icon."
fi

echo "Running PyInstaller..."
pyinstaller --noconfirm QCreviewer_beta.spec

echo ""
echo "Build finished."
echo "Check: dist/QCreviewer.app"