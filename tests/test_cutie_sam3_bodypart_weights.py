from __future__ import annotations

from collections import OrderedDict
import json
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

import annolid.segmentation.cutie_vos.predict as cutie_predict
from annolid.gui.mixins.model_identity_mixin import ModelIdentityMixin
from annolid.gui.mixins.prediction_execution_mixin import PredictionExecutionMixin
from annolid.gui.models_registry import MODEL_REGISTRY, validate_model_registry_entries
from annolid.segmentation.cutie_vos.engine import (
    DEFAULT_CUTIE_WEIGHTS_FILENAME,
    ORIG_BACKUP_CUTIE_WEIGHTS_FILENAME,
    SAM3_BODYPARTS_CUTIE_RELEASE_URL,
    SAM3_BODYPARTS_CUTIE_SHA256,
    SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME,
    UPGRADED_WEIGHTS_SHA256_ENV,
    UPGRADED_WEIGHTS_URL_ENV,
    CutieEngine,
    ensure_managed_cutie_weights_file,
    maybe_upgrade_and_cache_compact_cutie_weights,
    resolve_cutie_weights_path,
    unwrap_cutie_state_dict,
)
from annolid.segmentation.cutie_vos.inference.inference_core import InferenceCore
from annolid.segmentation.cutie_vos.interactive_utils import (
    _allow_non_blocking_host_transfer,
    image_to_torch,
)
from annolid.segmentation.cutie_vos.model.cutie import CUTIE
from annolid.segmentation.cutie_vos.model.weight_compression import (
    COMPACT_CHECKPOINT_FORMAT,
    TuckerConv2d,
    TuckerGConv2d,
    compress_and_factorize_cutie_in_place,
    fold_conv_bn_pair,
    hosvd_decompose_conv,
    is_compact_cutie_checkpoint,
    pack_compact_cutie_checkpoint,
)
from annolid.segmentation.cutie_vos.predict import CutieCoreVideoProcessor
from annolid.segmentation.cutie_vos.processor import SegmentedCutieExecutor
from annolid.segmentation.cutie_vos.runtime import (
    is_cutie_model_name,
    resolve_tracking_video_processor_class,
)
from annolid.segmentation.cutie_vos.video_processor import CutieVideoProcessor
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.cutie_mask_manager import CutieMaskManager
from annolid.utils.model_assets import md5sum, sha256sum

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = REPO_ROOT / "annolid" / "segmentation" / "cutie_vos" / "weights"
BASE_WEIGHTS_PATH = WEIGHTS_DIR / DEFAULT_CUTIE_WEIGHTS_FILENAME
ORIG_BACKUP_WEIGHTS_PATH = WEIGHTS_DIR / "cutie-base-mega.orig.pth"
UPGRADED_WEIGHTS_PATH = WEIGHTS_DIR / SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME
TINY_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "two_mice_body_part_tiny"


def test_resolve_cutie_weights_path_defaults_and_custom_lookup(tmp_path: Path) -> None:
    assert resolve_cutie_weights_path(None, weights_dir=tmp_path) is None
    assert resolve_cutie_weights_path("", weights_dir=tmp_path) is None
    assert resolve_cutie_weights_path("Cutie.pt", weights_dir=tmp_path) is None

    custom_file = tmp_path / "cutie-sam3-bodyparts.pth"
    custom_file.write_bytes(b"stub")
    assert (
        resolve_cutie_weights_path("cutie-sam3-bodyparts.pth", weights_dir=tmp_path)
        == custom_file
    )
    assert resolve_cutie_weights_path(custom_file, weights_dir=tmp_path) == custom_file

    missing = resolve_cutie_weights_path("nonexistent_custom.pth", weights_dir=tmp_path)
    assert missing == tmp_path / "nonexistent_custom.pth"


def test_unwrap_cutie_state_dict_supports_flat_and_nested_payloads() -> None:
    inner = {"pix_feat_proj.weight": torch.zeros((2, 2))}
    assert unwrap_cutie_state_dict(inner) == inner
    assert unwrap_cutie_state_dict({"model": inner}) == inner
    assert unwrap_cutie_state_dict({"weights": inner}) == inner
    assert unwrap_cutie_state_dict({"state_dict": inner}) == inner

    compact_envelope = {
        "__annolid_cutie_format__": COMPACT_CHECKPOINT_FORMAT,
        "tucker_config": {"fold_bn": True},
        "weights": inner,
    }
    assert unwrap_cutie_state_dict(compact_envelope) is compact_envelope


def test_conv_bn_folding_and_tucker_hosvd_compression_primitives() -> None:
    torch.manual_seed(42)
    conv = torch.nn.Conv2d(16, 24, kernel_size=3, padding=1, bias=False)
    bn = torch.nn.BatchNorm2d(24).eval()
    bn.running_mean.uniform_(-0.5, 0.5)
    bn.running_var.uniform_(0.2, 2.0)
    bn.weight.data.uniform_(0.5, 1.5)
    bn.bias.data.uniform_(-0.3, 0.3)

    x = torch.randn(2, 16, 12, 12)
    with torch.no_grad():
        expected = bn(conv(x))
        fused_conv, identity = fold_conv_bn_pair(conv, bn)
        actual = identity(fused_conv(x))
    assert torch.allclose(actual, expected, atol=1e-5)

    # Full-rank HOSVD Tucker-2 decomposition must reconstruct 3x3 conv up to FP32 precision
    tucker4d = hosvd_decompose_conv(
        fused_conv, rank_in=16, rank_out=24, is_group_5d=False
    )
    assert isinstance(tucker4d, TuckerConv2d)
    with torch.no_grad():
        assert torch.allclose(tucker4d(x), expected, atol=1e-4)

    # 5D group wrapper shape & parity check
    tucker5d = hosvd_decompose_conv(
        fused_conv, rank_in=12, rank_out=16, is_group_5d=True
    )
    assert isinstance(tucker5d, TuckerGConv2d)
    g = torch.randn(1, 3, 16, 12, 12)
    out5d = tucker5d(g)
    assert out5d.shape == (1, 3, 24, 12, 12)

    # End-to-end CUTIE compact pack + load_weights round-trip (multi-object & single-object)
    cfg_dir = REPO_ROOT / "annolid" / "segmentation" / "cutie_vos" / "config"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3.2", config_dir=str(cfg_dir)):
        cfg = compose(config_name="eval_config")
    src = CUTIE(cfg, pretrained_backbone=False).eval()
    t_cfg = compress_and_factorize_cutie_in_place(src)
    payload = pack_compact_cutie_checkpoint(src, tucker_cfg=t_cfg)
    assert is_compact_cutie_checkpoint(payload)

    dst_multi = CUTIE(cfg, pretrained_backbone=False).eval()
    dst_multi.load_weights(unwrap_cutie_state_dict(payload))
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in dst_multi.modules())

    dst_single = CUTIE(cfg, single_object=True, pretrained_backbone=False).eval()
    dst_single.load_weights(unwrap_cutie_state_dict(payload))
    assert dst_single.mask_encoder.conv1.weight.shape[1] == 4


def test_cutie_core_video_processor_loads_custom_weights_without_remote_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True)
    custom_ckpt = weights_dir / "cutie-sam3-bodyparts.pth"
    torch.save({"weights": {"ok": torch.tensor([1.0])}}, custom_ckpt)

    class _Cfg(dict):
        def __getattr__(self, item: str):
            return self[item]

    class _OpenDict:
        def __init__(self, cfg):
            self.cfg = cfg

        def __enter__(self):
            return self.cfg

        def __exit__(self, *args):
            return False

    class _DummyCutie:
        def __init__(self, cfg):
            self.cfg = cfg
            self.loaded = None

        def to(self, _device):
            return self

        def eval(self):
            return self

        def load_weights(self, sd):
            self.loaded = sd

    download_calls = []
    monkeypatch.setattr(
        cutie_predict,
        "ensure_cached_model_asset",
        lambda **kwargs: download_calls.append(kwargs),
    )
    monkeypatch.setattr(cutie_predict, "initialize", lambda **kwargs: None)
    monkeypatch.setattr(cutie_predict, "compose", lambda **kwargs: _Cfg())
    monkeypatch.setattr(cutie_predict, "open_dict", lambda cfg: _OpenDict(cfg))
    monkeypatch.setattr(cutie_predict, "CUTIE", _DummyCutie)
    monkeypatch.setattr(
        cutie_predict.GlobalHydra,
        "instance",
        lambda: types.SimpleNamespace(
            is_initialized=lambda: False,
            clear=lambda: None,
        ),
    )

    proc = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    proc.current_folder = str(tmp_path)
    proc.model_weights_path = "cutie-sam3-bodyparts.pth"
    proc.max_mem_frames = 5
    proc.max_internal_size = 480
    proc.use_amp = False
    proc.mem_every = 3
    proc.device = "cpu"

    model, cfg = proc._initialize_model()
    assert cfg.weights == str(custom_ckpt)
    assert torch.equal(model.loaded["ok"], torch.tensor([1.0]))
    assert download_calls == []

    # Missing explicit weights must fail loudly rather than silently downloading base weights
    proc.model_weights_path = str(tmp_path / "missing_custom.pth")
    with pytest.raises(FileNotFoundError):
        proc._initialize_model()


def test_executor_and_mask_manager_forward_custom_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    class _FakeEngine:
        def __init__(self, **kwargs):
            captured["engine_kwargs"] = kwargs

    monkeypatch.setattr(
        "annolid.segmentation.cutie_vos.processor.CutieEngine", _FakeEngine
    )
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    executor = SegmentedCutieExecutor(
        str(video_path),
        segment_annotated_frame=0,
        segment_start_frame=0,
        segment_end_frame=4,
        processing_config={
            "mem_every": 4,
            "t_max_value": 5,
            "model_weights_path": "cutie-sam3-bodyparts.pth",
        },
        device="cpu",
    )
    assert executor.cutie_engine is not None
    assert captured["engine_kwargs"]["model_weights_path"] == "cutie-sam3-bodyparts.pth"

    proc_captured = {}

    class _FakeCoreProc:
        def __init__(self, video_name, **kwargs):
            proc_captured["video_name"] = video_name
            proc_captured["kwargs"] = kwargs
            self.device = "cpu"
            self.cutie = object()
            self.cfg = types.SimpleNamespace(
                mem_every=4,
                stagger_updates=3,
                chunk_size=-1,
                save_aux=False,
                max_internal_size=-1,
                flip_aug=False,
                amp=False,
            )

    monkeypatch.setattr(
        "annolid.tracking.cutie_mask_manager.CutieCoreVideoProcessor", _FakeCoreProc
    )
    monkeypatch.setattr(
        "annolid.tracking.cutie_mask_manager.InferenceCore",
        lambda net, cfg: types.SimpleNamespace(network=net, cfg=cfg),
    )

    cfg = CutieDinoTrackerConfig(
        cutie_device="cpu",
        cutie_weights_path="cutie-sam3-bodyparts.pth",
    )
    adapter = types.SimpleNamespace(image_height=64, image_width=64)
    mgr = CutieMaskManager(video_path, adapter, cfg)
    mgr._ensure_core()
    assert proc_captured["kwargs"]["model_weights_path"] == "cutie-sam3-bodyparts.pth"


def test_registry_and_gui_wiring_for_cutie_sam3_bodyparts() -> None:
    valid, errors, warnings = validate_model_registry_entries(MODEL_REGISTRY)
    assert valid, f"Registry invalid: {errors}"

    matches = [m for m in MODEL_REGISTRY if m.identifier == "cutie_sam3_bodyparts"]
    assert len(matches) == 1
    entry = matches[0]
    assert entry.display_name == "Cutie (SAM3 Body Parts)"
    assert entry.weight_file == SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME
    assert is_cutie_model_name(entry.identifier)
    assert (
        resolve_tracking_video_processor_class(entry.identifier) is CutieVideoProcessor
    )

    class _Host(PredictionExecutionMixin, ModelIdentityMixin):
        video_file = "/tmp/demo.mp4"

        @staticmethod
        def get_video_brightness_contrast_values(_video_path):
            return (0, 0)

    host = _Host()
    assert host._is_cutie_tracking_model("cutie_sam3_bodyparts")
    assert host._is_cutie_tracking_model("Cutie")


@pytest.mark.skipif(
    not (
        ORIG_BACKUP_WEIGHTS_PATH.is_file()
        and BASE_WEIGHTS_PATH.is_file()
        and UPGRADED_WEIGHTS_PATH.is_file()
    ),
    reason="Local upstream backup and directly upgraded Cutie checkpoints required for live comparison.",
)
def test_cutie_sam3_bodyparts_checkpoint_integrity_and_tiny_fixture_improvement() -> (
    None
):
    # 1. Verify immutable upstream backup matches published Cutie MD5
    assert md5sum(ORIG_BACKUP_WEIGHTS_PATH) == CutieEngine._MD5

    # 2. Verify both default cutie-base-mega.pth and cutie-sam3-bodyparts.pth carry the
    # directly upgraded, compressed Tucker-2 + BN-folded checkpoint and match the published SHA-256
    assert sha256sum(BASE_WEIGHTS_PATH) == sha256sum(UPGRADED_WEIGHTS_PATH)
    assert sha256sum(UPGRADED_WEIGHTS_PATH) == SAM3_BODYPARTS_CUTIE_SHA256
    orig_size_mb = ORIG_BACKUP_WEIGHTS_PATH.stat().st_size / (1024.0 * 1024.0)
    up_size_mb = BASE_WEIGHTS_PATH.stat().st_size / (1024.0 * 1024.0)
    assert up_size_mb < 45.0
    assert orig_size_mb / up_size_mb > 3.5

    orig_sd = unwrap_cutie_state_dict(
        torch.load(ORIG_BACKUP_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    )
    up_payload = unwrap_cutie_state_dict(
        torch.load(BASE_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    )
    assert is_compact_cutie_checkpoint(up_payload)
    assert isinstance(up_payload["weights"], (dict, OrderedDict))
    for k, u_val in up_payload["weights"].items():
        if u_val.is_floating_point():
            assert torch.isfinite(u_val).all(), f"Non-finite tensor in {k}"

    # 3. Live VOS evaluation on tests/fixtures/two_mice_body_part_tiny (social_huddle 4 frames)
    manifest = json.loads(
        (TINY_FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    huddle_frames = [f for f in manifest["frames"] if f["scenario"] == "social_huddle"]
    assert len(huddle_frames) == 4

    frames_rgb = [
        cv2.cvtColor(
            cv2.imread(str(TINY_FIXTURE_DIR / f["image_file"])), cv2.COLOR_BGR2RGB
        )
        for f in huddle_frames
    ]
    masks_bgr = [
        cv2.imread(str(TINY_FIXTURE_DIR / f["id_mask_file"]), cv2.IMREAD_UNCHANGED)
        for f in huddle_frames
    ]

    cfg_dir = REPO_ROOT / "annolid" / "segmentation" / "cutie_vos" / "config"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3.2", config_dir=str(cfg_dir)):
        cfg = compose(config_name="eval_config")
    with open_dict(cfg):
        cfg.mem_every = 2
        cfg.max_mem_frames = 5
        cfg.max_internal_size = -1
        cfg.use_long_term = False
        cfg.amp = False
        cfg.chunk_size = -1
        cfg.top_k = 30
        cfg.stagger_updates = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = CUTIE(cfg).to(device).eval()
    base_model.load_weights(orig_sd)
    up_model = CUTIE(cfg).to(device).eval()
    up_model.load_weights(up_payload)

    orig_params = sum(p.numel() for p in base_model.parameters())
    up_params = sum(p.numel() for p in up_model.parameters())
    assert up_params < 20_000_000
    assert up_params < orig_params * 0.55
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in up_model.modules())

    def _eval_mode(model: CUTIE, mode: str) -> float:
        gts = []
        for bgr in masks_bgr:
            coarse, fine, track = (
                bgr[:, :, 0].astype(np.int32),
                bgr[:, :, 1].astype(np.int32),
                bgr[:, :, 2].astype(np.int32),
            )
            out = np.zeros_like(track)
            if mode == "instance":
                for tid in (1, 2):
                    out[track == tid] = tid
            elif mode == "coarse":
                for tid in (1, 2):
                    for pid in (1, 2, 3, 4):
                        out[(track == tid) & (coarse == pid)] = (tid - 1) * 4 + pid
            else:
                for tid in (1, 2):
                    for pid in range(1, 12):
                        out[(track == tid) & (fine == pid)] = (tid - 1) * 11 + pid
            gts.append(out)

        oids = [int(v) for v in np.unique(gts[0]) if int(v) > 0]
        core = InferenceCore(model, cfg=cfg)
        ious = []
        with torch.inference_mode():
            for fi, rgb in enumerate(frames_rgb):
                ft = image_to_torch(rgb, device=device)
                if fi == 0:
                    pred = core.step(
                        ft,
                        torch.from_numpy(gts[0]),
                        objects=oids,
                        idx_mask=True,
                        force_permanent=True,
                        return_index_mask=True,
                        output_size=rgb.shape[:2],
                    )
                else:
                    pred = core.step(
                        ft, return_index_mask=True, output_size=rgb.shape[:2]
                    )
                pred_np = core.output_prob_to_mask(pred).cpu().numpy()
                if fi > 0:
                    for oid in oids:
                        g = gts[fi] == oid
                        p = pred_np == oid
                        if not g.any() and not p.any():
                            continue
                        union = int(np.logical_or(p, g).sum())
                        inter = int(np.logical_and(p, g).sum())
                        ious.append(1.0 if union == 0 else inter / union)
        return float(np.mean(ious))

    base_inst = _eval_mode(base_model, "instance")
    up_inst = _eval_mode(up_model, "instance")
    base_coarse = _eval_mode(base_model, "coarse")
    up_coarse = _eval_mode(up_model, "coarse")
    base_fine = _eval_mode(base_model, "fine")
    up_fine = _eval_mode(up_model, "fine")

    # Upgraded weights must not regress whole-instance tracking and must measurably
    # improve overall instance + coarse + fine body-part segmentation accuracy.
    assert up_inst >= base_inst - 0.01, (
        f"Instance mIoU regressed: base={base_inst:.4f}, upgraded={up_inst:.4f}"
    )
    assert (up_inst + up_coarse + up_fine) > (base_inst + base_coarse + base_fine), (
        f"Expected improved aggregate mIoU: base={(base_inst + base_coarse + base_fine) / 3:.4f}, "
        f"upgraded={(up_inst + up_coarse + up_fine) / 3:.4f}"
    )


def test_fresh_install_auto_bootstraps_and_caches_compact_cutie_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 1. Hosted release-asset override (URL + SHA-256) takes precedence when configured
    rel_dir = tmp_path / "release_weights"
    rel_calls = []

    def _fake_release_downloader(**kwargs):
        rel_calls.append(kwargs)
        p = Path(kwargs["cache_dir"]) / kwargs["file_name"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"compact")
        return p

    monkeypatch.setenv(
        UPGRADED_WEIGHTS_URL_ENV, "https://example.org/cutie-compact.pth"
    )
    monkeypatch.setenv(UPGRADED_WEIGHTS_SHA256_ENV, "abc123")
    got = ensure_managed_cutie_weights_file(
        SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME,
        weights_dir=rel_dir,
        remote_url="https://example.org/upstream.pth",
        expected_md5="deadbeef",
        asset_downloader=_fake_release_downloader,
    )
    assert got == rel_dir / SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME
    assert rel_calls[0]["expected_sha256"] == "abc123"

    # 2. Built-in canonical GitHub Release asset (zero env vars required) downloads the
    # Dual-Teacher checkpoint verified by SAM3_BODYPARTS_CUTIE_SHA256 and mirrors the sibling file
    monkeypatch.delenv(UPGRADED_WEIGHTS_URL_ENV, raising=False)
    monkeypatch.delenv(UPGRADED_WEIGHTS_SHA256_ENV, raising=False)

    builtin_dir = tmp_path / "builtin_release_weights"
    builtin_calls = []

    def _fake_builtin_release_downloader(**kwargs):
        builtin_calls.append(kwargs)
        p = Path(kwargs["cache_dir"]) / kwargs["file_name"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"distilled_compact")
        return p

    got_builtin = ensure_managed_cutie_weights_file(
        None,
        weights_dir=builtin_dir,
        remote_url="https://example.org/upstream.pth",
        expected_md5="a6071de6136982e396851903ab4c083a",
        asset_downloader=_fake_builtin_release_downloader,
        try_release_asset=True,
    )
    assert got_builtin == builtin_dir / DEFAULT_CUTIE_WEIGHTS_FILENAME
    assert builtin_calls[0]["url"] == SAM3_BODYPARTS_CUTIE_RELEASE_URL
    assert builtin_calls[0]["expected_sha256"] == SAM3_BODYPARTS_CUTIE_SHA256
    assert (builtin_dir / SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME).is_file()

    # 3. Frozen desktop bundle (Annolid.app) never mutates the app bundle, routing to model_downloads_dir()
    frozen_bundle_weights = (
        tmp_path / "Annolid.app" / "Contents" / "Frameworks" / "weights"
    )
    user_dl_dir = tmp_path / "user_downloads"
    monkeypatch.setattr(
        "annolid.segmentation.cutie_vos.engine.model_downloads_dir",
        lambda: user_dl_dir,
    )
    got_frozen = ensure_managed_cutie_weights_file(
        SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME,
        weights_dir=frozen_bundle_weights,
        remote_url="https://example.org/upstream.pth",
        expected_md5="a6071de6136982e396851903ab4c083a",
        asset_downloader=_fake_builtin_release_downloader,
        try_release_asset=True,
    )
    assert got_frozen == user_dl_dir / SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME
    assert not frozen_bundle_weights.exists()

    # 4. If the GitHub Release asset is not yet published (404), seamlessly fall back to
    # upstream dense weights, back up cutie-base-mega.orig.pth, and cache compact checkpoints
    cfg_dir = REPO_ROOT / "annolid" / "segmentation" / "cutie_vos" / "config"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3.2", config_dir=str(cfg_dir)):
        cfg = compose(config_name="eval_config")

    dense_model = CUTIE(cfg, pretrained_backbone=False).eval()
    dense_sd = OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in dense_model.state_dict().items()
    )

    fresh_dir = tmp_path / "fresh_weights"

    def _fake_fallback_downloader(**kwargs):
        if kwargs.get("url") == SAM3_BODYPARTS_CUTIE_RELEASE_URL:
            raise RuntimeError("HTTP 404 Not Found")
        p = Path(kwargs["cache_dir"]) / kwargs["file_name"]
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weights": dense_sd}, p)
        return p

    target = ensure_managed_cutie_weights_file(
        SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME,
        weights_dir=fresh_dir,
        remote_url="https://example.org/upstream.pth",
        expected_md5="a6071de6136982e396851903ab4c083a",
        asset_downloader=_fake_fallback_downloader,
        try_release_asset=True,
    )
    assert target == fresh_dir / SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME
    assert target.is_file()

    runtime_model = CUTIE(cfg, pretrained_backbone=False).eval()
    unwrapped = unwrap_cutie_state_dict(
        torch.load(target, map_location="cpu", weights_only=True)
    )
    runtime_model.load_weights(unwrapped)
    maybe_upgrade_and_cache_compact_cutie_weights(
        runtime_model,
        unwrapped,
        model_path=target,
        weights_dir=fresh_dir,
    )

    assert (fresh_dir / ORIG_BACKUP_CUTIE_WEIGHTS_FILENAME).is_file()
    assert sum(p.numel() for p in runtime_model.parameters()) < 20_000_000
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in runtime_model.modules())
    for fname in (
        DEFAULT_CUTIE_WEIGHTS_FILENAME,
        SAM3_BODYPARTS_CUTIE_WEIGHTS_FILENAME,
    ):
        saved = torch.load(fresh_dir / fname, map_location="cpu", weights_only=True)
        assert is_compact_cutie_checkpoint(saved)


def test_mps_host_to_device_transfer_determinism_in_inference_core() -> None:
    unpinned_cpu = torch.zeros((3, 8, 8), dtype=torch.float32)
    assert not _allow_non_blocking_host_transfer(unpinned_cpu, "cpu")
    assert not _allow_non_blocking_host_transfer(unpinned_cpu, "cuda")
    if torch.backends.mps.is_available():
        assert not _allow_non_blocking_host_transfer(unpinned_cpu, "mps")

    if not (torch.backends.mps.is_available() and BASE_WEIGHTS_PATH.is_file()):
        return

    manifest = json.loads(
        (TINY_FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    huddle_frames = [f for f in manifest["frames"] if f["scenario"] == "social_huddle"]
    frames_rgb = [
        cv2.cvtColor(
            cv2.imread(str(TINY_FIXTURE_DIR / f["image_file"])), cv2.COLOR_BGR2RGB
        )
        for f in huddle_frames
    ]
    masks_bgr = [
        cv2.imread(str(TINY_FIXTURE_DIR / f["id_mask_file"]), cv2.IMREAD_UNCHANGED)
        for f in huddle_frames
    ]
    gts = []
    for bgr in masks_bgr:
        fine, track = bgr[:, :, 1].astype(np.int32), bgr[:, :, 2].astype(np.int32)
        out = np.zeros_like(track)
        for tid in (1, 2):
            for pid in range(1, 12):
                out[(track == tid) & (fine == pid)] = (tid - 1) * 11 + pid
        gts.append(out)
    oids = [int(v) for v in np.unique(gts[0]) if int(v) > 0]

    cfg_dir = REPO_ROOT / "annolid" / "segmentation" / "cutie_vos" / "config"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3.2", config_dir=str(cfg_dir)):
        cfg = compose(config_name="eval_config")
    with open_dict(cfg):
        cfg.mem_every = 2
        cfg.max_mem_frames = 5
        cfg.max_internal_size = -1
        cfg.use_long_term = False
        cfg.amp = False
        cfg.chunk_size = -1
        cfg.top_k = 30
        cfg.stagger_updates = 2

    payload = unwrap_cutie_state_dict(
        torch.load(BASE_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    )

    def _run_sequence(dev: str) -> list[np.ndarray]:
        model = CUTIE(cfg).to(dev).eval()
        model.load_weights(payload)
        core = InferenceCore(model, cfg=cfg)
        preds = []
        with torch.inference_mode():
            for fi, rgb in enumerate(frames_rgb):
                ft = image_to_torch(rgb, device=dev)
                if fi == 0:
                    # Pass an unpinned temporary CPU mask tensor directly to step()
                    pred = core.step(
                        ft,
                        torch.from_numpy(gts[0].copy()),
                        objects=oids,
                        idx_mask=True,
                        force_permanent=True,
                        return_index_mask=True,
                        output_size=rgb.shape[:2],
                    )
                else:
                    pred = core.step(
                        ft, return_index_mask=True, output_size=rgb.shape[:2]
                    )
                preds.append(core.output_prob_to_mask(pred).cpu().numpy())
        return preds

    cpu_preds = _run_sequence("cpu")
    mps_runs = [_run_sequence("mps") for _ in range(3)]
    for run_idx in (1, 2):
        for fi in range(len(frames_rgb)):
            assert np.array_equal(mps_runs[0][fi], mps_runs[run_idx][fi]), (
                f"MPS run {run_idx} frame {fi} diverged from run 0"
            )
    for fi in range(1, len(frames_rgb)):
        agree = float((cpu_preds[fi] == mps_runs[0][fi]).mean())
        assert agree > 0.999, f"MPS vs CPU pixel agreement {agree:.6f} on frame {fi}"
