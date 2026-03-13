# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None

hiddenimports = []
hiddenimports += collect_submodules("napari")
hiddenimports += collect_submodules("nibabel")
hiddenimports += collect_submodules("vispy")

datas = []
datas += collect_data_files("napari")
datas += collect_data_files("vispy")
datas += [("app.ico", ".")]
datas += copy_metadata("imageio")
datas += copy_metadata("napari")
datas += copy_metadata("vispy")
datas += copy_metadata("magicgui")
datas += copy_metadata("superqt")
datas += copy_metadata("app-model")
datas += copy_metadata("qtpy")
binaries = []

for pkg in [
    'napari',
    'qtpy',
    'nibabel',
    'scipy',
    'skimage',
    'vispy',
    'magicgui',
    'superqt',
    'app_model',
]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# Always bundle local UI/icon resources.
datas += [
    ('app.ico', '.'),
    ('logo.png', '.'),
]


a = Analysis(
    ['QCreviewer.py'],
    pathex=[],
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

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='QCreviewer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    icon='app.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='QCreviewer',
)
