@echo off
setlocal
cd /d "%~dp0"

echo Cleaning old build folders...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Running PyInstaller...
pyinstaller --noconfirm QCreviewer.spec || goto :fail

echo.
echo Build complete.
echo Send the entire folder:
echo   dist\QCreviewer\
echo Not just the exe alone.
exit /b 0

:fail
echo.
echo Build failed.
exit /b 1
