# -*- mode: python -*-
# vim: ft=python

from pathlib import Path
import sys
import tomllib
from PyInstaller.utils.hooks import copy_metadata


sys.setrecursionlimit(5000)  # required on Windows

ROOT = Path(SPECPATH)


def _read_app_version(root: Path) -> str:
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        version = str(data.get("project", {}).get("version") or "").strip()
        if version:
            return version
    except Exception:
        pass
    return "0.0.0"


APP_VERSION = _read_app_version(ROOT)

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
    "project.annolid.json",
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
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in EXCLUDED_SUFFIXES:
        return True
    if any(part.lower().endswith(".mlpackage") for part in p.parts):
        return True
    return any(part in path for part in EXCLUDED_PATH_PARTS)


def _prune_toc(toc):
    return type(toc)(entry for entry in toc if not _is_excluded_artifact(entry[1]))


HIDDEN_IMPORT_SKIP_PREFIXES = (
    "annolid/core/agent/skills",
    "annolid/depth/video_depth_anything",
    "annolid/detector/countgd",
    "annolid/detector/yolov5",
    "annolid/realtime/models",
    "annolid/realtime/runs",
    "annolid/realtime/scripts",
    "annolid/scripts",
    "annolid/segmentation/MEDIAR",
    "annolid/segmentation/SAM",
    "annolid/segmentation/cutie_vos",
    "annolid/segmentation/yolact",
    "annolid/tracker/cowtracker/cowtracker/thirdparty",
    "annolid/utils/tensorboard_sitecustomize",
)

CORE_HIDDEN_IMPORTS = [
    "annolid.gui.app",
    "annolid.domain.project_schema",
    "annolid.infrastructure.persistence",
]


def _collect_annolid_hiddenimports(root: Path) -> list[str]:
    """Collect first-party Annolid modules reachable through lazy __getattr__ loaders."""
    modules = set(CORE_HIDDEN_IMPORTS)
    pkg_root = root / "annolid"
    for py_path in sorted(pkg_root.rglob("*.py")):
        rel = py_path.relative_to(root).as_posix()
        parts = py_path.relative_to(root).parts
        if any(
            part.startswith(".") or part in {"__pycache__", "tests"}
            for part in parts
        ):
            continue
        if any(
            rel == prefix or rel.startswith(prefix + "/")
            for prefix in HIDDEN_IMPORT_SKIP_PREFIXES
        ):
            continue
        if py_path.name == "__init__.py":
            mod_parts = parts[:-1]
        else:
            mod_parts = (*parts[:-1], py_path.stem)
        if mod_parts and all(part.isidentifier() for part in mod_parts):
            modules.add(".".join(mod_parts))
    return sorted(modules)


HIDDEN_IMPORTS = _collect_annolid_hiddenimports(ROOT)


def _collect_tree_datas(
    root: Path, rel_dir: str, suffixes: set[str]
) -> list[tuple[str, str]]:
    base = root / rel_dir
    if not base.is_dir():
        return []
    entries: list[tuple[str, str]] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        rel_parent = path.parent.relative_to(root).as_posix()
        entries.append((path.relative_to(root).as_posix(), rel_parent))
    return entries


a = Analysis(
    ["annolid/gui/launcher.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        ("annolid/configs/*.yaml", "annolid/configs"),
        ("annolid/configs/*.json", "annolid/configs"),
        ("annolid/detector/configs/*", "annolid/detector/configs"),
        ("annolid/tracker/configs/*.yaml", "annolid/tracker/configs"),
        ("annolid/annotation/labels.txt", "annolid/annotation"),
        ("annolid/annotation/labels_interaction", "annolid/annotation"),
        ("annolid/core/output/schema/*.json", "annolid/core/output/schema"),
        *_collect_tree_datas(ROOT, "annolid/core/agent/skills", {".md", ".py", ".sh"}),
        *_collect_tree_datas(ROOT, "annolid/core/agent/workspace", {".md", ".sh"}),
        ("annolid/icons/*", "annolid/icons"),
        ("annolid/gui/icons/*", "annolid/gui/icons"),
        ("annolid/gui/assets/*", "annolid/gui/assets"),
        *copy_metadata("annolid"),
    ],
    # The launcher and stable namespace packages defer imports with importlib.
    hiddenimports=HIDDEN_IMPORTS,
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
        bundle_identifier="com.healthonrails.annolid",
        info_plist={
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION,
            "NSHighResolutionCapable": True,
        },
    )
