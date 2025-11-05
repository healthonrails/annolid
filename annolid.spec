# -*- mode: python -*-
# vim: ft=python

import sys


sys.setrecursionlimit(5000)  # required on Windows


a = Analysis(
    ['annolid/gui/app.py'],
    pathex=['annolid'],
    binaries=[],
    datas=[
        ('annolid/configs/default_config.yaml', 'annolid/configs'),
        ('annolid/icons/*', 'annolid/icons'),
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    # Prevent PyInstaller from loading alternate Qt bindings; project uses PyQt5.
    excludes=['PySide6', 'PySide2', 'PyQt6'],
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='annolid',
    debug=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    icon='annolid/icons/icon.ico',
)
app = BUNDLE(
    exe,
    name='Annolid.app',
    icon='annolid/icons/icon.icns',
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)
