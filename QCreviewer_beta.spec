# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
    collect_all,
)

block_cipher = None
is_macos = sys.platform == "darwin"
is_windows = sys.platform.startswith("win")

# -----------------------------------------------------------------------------
# Paths / local resources
# -----------------------------------------------------------------------------
project_dir = os.path.abspath(os.getcwd())
main_script = os.path.join(project_dir, "QCreviewer_beta.py")

icon_ico = os.path.join(project_dir, "app.ico")
icon_icns = os.path.join(project_dir, "app.icns")
logo_png = os.path.join(project_dir, "logo.png")

# -----------------------------------------------------------------------------
# Hidden imports
# -----------------------------------------------------------------------------
hiddenimports = []
for pkg in ["napari", "nibabel", "vispy"]:
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Data files
# -----------------------------------------------------------------------------
datas = []
binaries = []

for pkg in ["napari", "vispy"]:
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

for pkg in [
    "imageio",
    "napari",
    "vispy",
    "magicgui",
    "superqt",
    "app-model",
    "qtpy",
]:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

for pkg in [
    "napari",
    "qtpy",
    "nibabel",
    "scipy",
    "skimage",
    "vispy",
    "magicgui",
    "superqt",
    "app_model",
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# -----------------------------------------------------------------------------
# Local app resources (add only if they exist)
# -----------------------------------------------------------------------------
if os.path.exists(logo_png):
    datas.append((logo_png, "."))

if os.path.exists(icon_ico):
    datas.append((icon_ico, "."))

if os.path.exists(icon_icns):
    datas.append((icon_icns, "."))

# Deduplicate lists a bit
hiddenimports = list(dict.fromkeys(hiddenimports))
datas = list(dict.fromkeys(datas))
binaries = list(dict.fromkeys(binaries))

# -----------------------------------------------------------------------------
# Analysis
# -----------------------------------------------------------------------------
a = Analysis(
    [main_script],
    pathex=[project_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PySide6", "PySide2", "PyQt6"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# -----------------------------------------------------------------------------
# Choose icon depending on platform
# -----------------------------------------------------------------------------
exe_icon = None
if is_windows and os.path.exists(icon_ico):
    exe_icon = icon_ico
elif is_macos and os.path.exists(icon_icns):
    exe_icon = icon_icns
elif os.path.exists(icon_ico):
    exe_icon = icon_ico

# -----------------------------------------------------------------------------
# EXE
# -----------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="QCreviewer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon=exe_icon,
)

# -----------------------------------------------------------------------------
# COLLECT
# -----------------------------------------------------------------------------
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="QCreviewer",
)

# -----------------------------------------------------------------------------
# macOS app bundle
# -----------------------------------------------------------------------------
if is_macos:
    app_icon = icon_icns if os.path.exists(icon_icns) else None

    app = BUNDLE(
        coll,
        name="QCreviewer.app",
        icon=app_icon,
        bundle_identifier="com.qsm.qcreviewer",
        info_plist={
            "CFBundleName": "QCreviewer",
            "CFBundleDisplayName": "QCreviewer",
            "CFBundleShortVersionString": "1.0.0",
            "CFBundleVersion": "1.0.0",
            "NSHighResolutionCapable": True,
        },
    )