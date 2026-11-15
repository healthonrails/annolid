# -*- mode: python -*-
# vim: ft=python

from pathlib import Path
import sys


sys.setrecursionlimit(5000)  # required on Windows

ROOT = Path(SPECPATH)

# Desktop bundle policy:
# - include the GUI shell, core annotation/video IO, config, icons, and UI assets;
# - do not bundle ML runtimes, training stacks, model checkpoints, or generated runs;
# - optional features should fail with their existing install hints, then users install
#   the matching pip extra/package in their normal Python environment.

QT_BINDING_EXCLUDES = [
    # Annolid's GUI extra installs PySide6-Essentials; keep alternate bindings
    # and unused heavyweight Qt submodules out of frozen desktop bundles.
    "PyQt5",
    "PyQt6",
    "PySide2",
]

QT_UNUSED_MODULE_EXCLUDES = [
    "PySide6.Qt3DAnimation",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DRender",
    "PySide6.QtBluetooth",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtDesigner",
    "PySide6.QtGraphs",
    "PySide6.QtHelp",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtNfc",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtPositioning",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuickWidgets",
    "PySide6.QtRemoteObjects",
    "PySide6.QtScxml",
    "PySide6.QtSensors",
    "PySide6.QtSerialBus",
    "PySide6.QtSerialPort",
    "PySide6.QtSpatialAudio",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtTextToSpeech",
    "PySide6.QtWebChannel",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebSockets",
]

ML_RUNTIME_EXCLUDES = [
    # PyTorch and siblings
    "torch",
    "torchvision",
    "torchaudio",
    "torchtext",
    "triton",
    # ONNX/OpenVINO runtimes are installed per machine/GPU choice.
    "onnx",
    "onnxruntime",
    "onnxruntime_gpu",
    "openvino",
    # Other large accelerator/runtime stacks
    "tensorflow",
    "keras",
    "jax",
    "jaxlib",
    "mxnet",
    # Scientific/training stacks are installed only for workflows that need them.
    "h5py",
    "matplotlib",
    "numba",
    "pandas",
    "scikit_image",
    "skimage",
    "scikit_learn",
    "sklearn",
    "scipy",
    "tensorboard",
]

OPTIONAL_MODEL_STACK_EXCLUDES = [
    # Transformer/model tooling
    "accelerate",
    "diffusers",
    "huggingface_hub",
    "safetensors",
    "sentencepiece",
    "timm",
    "tokenizers",
    "transformers",
    # Detection/segmentation/tracking extras
    "detectron2",
    "groundingdino",
    "iopath",
    "lap",
    "mediapipe",
    "sam2",
    "sam3",
    "segment_anything",
    "ultralytics",
    # Optional large-image and document backends
    "openslide",
    "pyvips",
    "large_image",
    "large_image_source_gdal",
    "large_image_source_openslide",
    "large_image_source_rasterio",
    "large_image_source_tiff",
]

OPTIONAL_SERVICE_EXCLUDES = [
    # Agent/cloud/channel and dev/download dependencies are installed on demand.
    "IPython",
    "anthropic",
    "bs4",
    "gdown",
    "google",
    "googleapiclient",
    "google_auth_oauthlib",
    "lancedb",
    "mcp",
    "openai",
    "playwright",
    "pyarrow",
    "pytest",
    "qrcode",
    "scrapling",
    "tantivy",
    "tkinter",
]

EXCLUDED_MODULES = sorted(
    set(
        QT_BINDING_EXCLUDES
        + QT_UNUSED_MODULE_EXCLUDES
        + ML_RUNTIME_EXCLUDES
        + OPTIONAL_MODEL_STACK_EXCLUDES
        + OPTIONAL_SERVICE_EXCLUDES
    )
)

# Path fragments that must never be collected as bundle data/binaries. These are
# large local artifacts or generated outputs; runtime code should download/cache
# models or ask users to install the relevant extra when a feature is launched.
EXCLUDED_PATH_PARTS = {
    ".cache",
    "__pycache__",
    "annolid/configs/runs",
    "annolid/depth/checkpoints",
    "annolid/detector/countgd/checkpoints",
    "annolid/detector/countgd/checkpoint_best_regular.pth",
    "annolid/detector/groundingdino_swinb_cogcoor_quant.onnx",
    "annolid/motion/weights",
    "annolid/realtime/models",
    "annolid/realtime/runs",
    "annolid/realtime/yolo11n-seg.mlpackage",
    "annolid/realtime/yolo11n-seg_openvino_model",
    "annolid/realtime/yolo11n.mlpackage",
    "annolid/segmentation/MEDIAR/weights",
    "annolid/segmentation/SAM/edge_sam_3x_decoder.onnx",
    "annolid/segmentation/SAM/edge_sam_3x_encoder.onnx",
    "annolid/segmentation/SAM/segment-anything-2/checkpoints",
    "annolid/segmentation/SAM/vit_b.pth",
    "annolid/segmentation/cutie_vos/weights",
    "annolid/segmentation/sam2.1_b.pt",
    "annolid/tracker/features/checkpoint",
}

EXCLUDED_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".engine",
    ".h5",
    ".mlmodel",
    ".onnx",
    ".pth",
    ".pt",
    ".safetensors",
    ".t7",
    ".tflite",
    ".weights",
}


def _normalized_path(value):
    try:
        return Path(value).as_posix()
    except TypeError:
        return str(value)


def _is_excluded_artifact(value):
    path = _normalized_path(value)
    suffix = Path(path).suffix.lower()
    if suffix in EXCLUDED_SUFFIXES:
        return True
    return any(part in path for part in EXCLUDED_PATH_PARTS)


def _prune_toc(toc):
    return type(toc)(entry for entry in toc if not _is_excluded_artifact(entry[1]))


a = Analysis(
    ["annolid/gui/launcher.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        ("annolid/configs/default_config.yaml", "annolid/configs"),
        ("annolid/icons/*", "annolid/icons"),
        ("annolid/gui/assets/*", "annolid/gui/assets"),
    ],
    # The launcher and stable namespace packages defer imports with importlib.
    hiddenimports=[
        "annolid.gui.app",
        "annolid.domain.project_schema",
        "annolid.infrastructure.persistence",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDED_MODULES,
    noarchive=False,
)

a.binaries = _prune_toc(a.binaries)
a.datas = _prune_toc(a.datas)

pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="annolid",
    debug=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    icon="annolid/icons/icon.ico",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="annolid",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Annolid.app",
        icon="annolid/icons/icon.icns",
        bundle_identifier=None,
        info_plist={"NSHighResolutionCapable": "True"},
    )
