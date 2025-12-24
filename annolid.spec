# -*- mode: python -*-
# vim: ft=python

import sys


sys.setrecursionlimit(5000)  # required on Windows

# Keep the bundle lean: ship only the GUI and its assets. Heavy ML stacks
# (PyTorch, transformers, YOLO/Detectron extras, ONNX runtimes, etc.) are
# excluded so users install them separately when needed.
EXCLUDED_MODULES = [
    # Alternate Qt bindings
    'PySide6', 'PySide2', 'PyQt6',
    # PyTorch & siblings
    'torch', 'torchvision', 'torchaudio', 'torchtext', 'triton',
    # Other heavy runtimes
    'tensorflow', 'onnx', 'onnxruntime', 'onnxruntime_gpu',
    'mxnet', 'jax', 'diffusers', 'accelerate',
    # Large model/tooling stacks that are optional in the GUI (see pyproject dependencies)
    'transformers', 'huggingface_hub', 'sentencepiece', 'tokenizers',
    'timm', 'ultralytics', 'detectron2', 'pytorch_lightning', 'tensorboard',
]

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
    excludes=EXCLUDED_MODULES,
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
