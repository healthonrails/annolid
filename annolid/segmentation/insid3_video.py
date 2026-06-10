"""INSID3-style in-context DINOv3 segmentation for Annolid videos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from annolid.annotation.masks import mask_to_polygons
from annolid.data.videos import CV2Video
from annolid.features import Dinov3Config, Dinov3FeatureExtractor
from annolid.features.dinov3_feature_grid import (
    apply_channel_debias_basis,
    extract_feature_grid,
    normalize_feature_grid,
    svd_positional_basis,
)
from annolid.segmentation.crf_refinement import (
    CrfMaskRefiner,
    CrfRefinementConfig,
)
from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.domain import InstanceRegistry
from annolid.utils.logger import logger


@dataclass(frozen=True)
class Insid3VideoConfig:
    """Runtime parameters for INSID3-style video segmentation."""

    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    short_side: int = 768
    svd_components: int = 20
    tau: float = 0.6
    merge_threshold: float = 0.2
    max_cluster_area_growth: float = 8.0
    max_seed_area_growth: float = 3.0
    min_seed_pixels: int = 1
    label_competition_margin: float = 0.02
    search_bbox_padding: float = 0.75
    spatial_prior_weight: float = 0.25
    crf_refine: bool = False
    crf_backend: str = "auto"
    crf_band_px: int = 10
    crf_p_core: float = 0.95
    crf_iterations: int = 10
    device: Optional[str] = None


@dataclass
class _ReferenceObject:
    label: str
    mask: torch.Tensor
    prototype: torch.Tensor
    features: torch.Tensor
    seed_area: int
    seed_shape: Tuple[int, int]
    seed_bbox: Tuple[int, int, int, int]


class Insid3VideoSegmenter:
    """Training-free in-context segmentation on DINO feature grids."""

    def __init__(
        self,
        *,
        config: Optional[Insid3VideoConfig] = None,
        extractor: Optional[Dinov3FeatureExtractor] = None,
    ) -> None:
        self.config = config or Insid3VideoConfig()
        self.extractor = extractor or Dinov3FeatureExtractor(
            Dinov3Config(
                model_name=self.config.model_name,
                short_side=int(self.config.short_side),
                device=self.config.device,
                layers=(-2, -1),
            )
        )
        self.patch_size = int(getattr(self.extractor, "patch_size", 16))
        self._basis_cache: Dict[Tuple[str, int, int, int, int], torch.Tensor] = {}
        self.mask_refiner = CrfMaskRefiner(
            CrfRefinementConfig(
                enabled=bool(self.config.crf_refine),
                backend=str(self.config.crf_backend),
                band_px=int(self.config.crf_band_px),
                p_core=float(self.config.crf_p_core),
                iterations=int(self.config.crf_iterations),
            ),
            device=self.config.device,
        )

    def build_references(
        self,
        image: Image.Image,
        masks: Dict[str, np.ndarray],
    ) -> list[_ReferenceObject]:
        feats = self._extract_features(image)
        debiased = self._debias_feature_grid(feats)
        refs: list[_ReferenceObject] = []
        for label, mask_np in masks.items():
            mask = self._downsample_mask(mask_np, debiased.shape[-2:])
            if int(mask.sum().item()) < int(self.config.min_seed_pixels):
                logger.warning(
                    "INSID3 seed mask for '%s' is empty after downsampling.", label
                )
                continue
            fg = debiased[:, mask]
            prototype = F.normalize(fg.mean(dim=1), p=2, dim=0)
            seed_bbox = _mask_bbox(np.asarray(mask_np, dtype=bool))
            if seed_bbox is None:
                continue
            refs.append(
                _ReferenceObject(
                    label=label,
                    mask=mask,
                    prototype=prototype,
                    features=debiased.detach(),
                    seed_area=int(np.count_nonzero(mask_np)),
                    seed_shape=(
                        int(np.asarray(mask_np).shape[0]),
                        int(np.asarray(mask_np).shape[1]),
                    ),
                    seed_bbox=seed_bbox,
                )
            )
        return refs

    @torch.no_grad()
    def segment(
        self,
        image: Image.Image,
        references: Sequence[_ReferenceObject],
        *,
        output_size: Tuple[int, int],
        priors: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        feats = self._extract_features(image)
        debiased = self._debias_feature_grid(feats)
        cluster_labels = self._cluster_features(feats, tau=float(self.config.tau))
        num_clusters = (
            int(cluster_labels.max().item()) + 1 if cluster_labels.numel() else 0
        )
        if num_clusters <= 0:
            return {}

        original_prototypes = self._cluster_prototypes(
            feats.reshape(feats.shape[0], -1).transpose(0, 1),
            cluster_labels.reshape(-1),
            num_clusters,
        )
        debiased_prototypes = self._cluster_prototypes(
            debiased.reshape(debiased.shape[0], -1).transpose(0, 1),
            cluster_labels.reshape(-1),
            num_clusters,
        )

        predictions: Dict[str, np.ndarray] = {}
        image_arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        sim_maps = torch.stack(
            [torch.einsum("chw,c->hw", debiased, ref.prototype) for ref in references],
            dim=0,
        )
        best_label_sim = sim_maps.max(dim=0).values
        for ref_idx, ref in enumerate(references):
            sim = sim_maps[ref_idx]
            competition_mask = None
            if len(references) > 1:
                competition_mask = sim >= (
                    best_label_sim - float(self.config.label_competition_margin)
                )
            patch_mask = self._predict_patch_mask(
                feats=feats,
                debiased=debiased,
                cluster_labels=cluster_labels,
                original_prototypes=original_prototypes,
                debiased_prototypes=debiased_prototypes,
                reference=ref,
                prototype_similarity=sim,
                label_competition_mask=competition_mask,
                num_clusters=num_clusters,
            )
            mask = self._upsample_mask(patch_mask, output_size)
            score = self._upsample_score_map(sim, output_size)
            prior = priors.get(ref.label) if priors else None
            mask = self._regularize_pixel_mask(
                mask=mask,
                score=score,
                reference=ref,
                output_size=output_size,
                prior_mask=prior,
            )
            predictions[ref.label] = self.mask_refiner.refine(image_arr, mask)
        return predictions

    def _extract_features(self, image: Image.Image) -> torch.Tensor:
        return extract_feature_grid(self.extractor, image)

    @staticmethod
    def _normalize_feature_grid(feats: torch.Tensor) -> torch.Tensor:
        return normalize_feature_grid(feats)

    def _debias_feature_grid(self, feats: torch.Tensor) -> torch.Tensor:
        basis = self._positional_basis(feats)
        if basis is None or basis.numel() == 0:
            return feats
        return apply_channel_debias_basis(feats, basis, strength=1.0)

    def _positional_basis(self, feats: torch.Tensor) -> Optional[torch.Tensor]:
        channels, grid_h, grid_w = feats.shape
        model_key = str(getattr(getattr(self.extractor, "cfg", None), "model_name", ""))
        return svd_positional_basis(
            extractor=self.extractor,
            feature_shape=(int(channels), int(grid_h), int(grid_w)),
            patch_size=int(self.patch_size),
            components=max(1, int(self.config.svd_components)),
            cache=self._basis_cache,
            model_key=model_key,
        )

    @staticmethod
    def _downsample_mask(mask: np.ndarray, grid_hw: Tuple[int, int]) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
        down = (
            F.interpolate(tensor, size=grid_hw, mode="bilinear", align_corners=False)[
                0, 0
            ]
            > 0.5
        )
        if bool(down.any()):
            return down
        nearest = F.interpolate(tensor, size=grid_hw, mode="nearest")[0, 0] > 0.5
        if bool(nearest.any()):
            return nearest
        ys, xs = np.nonzero(mask)
        if ys.size:
            rr = int(
                np.clip(
                    round(float(ys.mean()) * grid_hw[0] / mask.shape[0]),
                    0,
                    grid_hw[0] - 1,
                )
            )
            cc = int(
                np.clip(
                    round(float(xs.mean()) * grid_hw[1] / mask.shape[1]),
                    0,
                    grid_hw[1] - 1,
                )
            )
            nearest[rr, cc] = True
        return nearest

    @staticmethod
    def _upsample_mask(mask: torch.Tensor, output_size: Tuple[int, int]) -> np.ndarray:
        height, width = int(output_size[0]), int(output_size[1])
        up = F.interpolate(
            mask[None, None].float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return (up > 0.5).cpu().numpy().astype(bool)

    @staticmethod
    def _upsample_score_map(
        score: torch.Tensor, output_size: Tuple[int, int]
    ) -> np.ndarray:
        height, width = int(output_size[0]), int(output_size[1])
        up = F.interpolate(
            score[None, None].float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        out = up.detach().cpu().numpy().astype(np.float32)
        finite = np.isfinite(out)
        if not bool(finite.any()):
            return np.zeros((height, width), dtype=np.float32)
        low = float(np.nanmin(out[finite]))
        high = float(np.nanmax(out[finite]))
        if high <= low:
            return np.zeros((height, width), dtype=np.float32)
        return ((out - low) / (high - low)).clip(0.0, 1.0)

    def _regularize_pixel_mask(
        self,
        *,
        mask: np.ndarray,
        score: np.ndarray,
        reference: _ReferenceObject,
        output_size: Tuple[int, int],
        prior_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        mask_bool = np.asarray(mask, dtype=bool)
        if not bool(mask_bool.any()):
            return mask_bool

        height, width = int(output_size[0]), int(output_size[1])
        scale_y = height / max(1, int(reference.seed_shape[0]))
        scale_x = width / max(1, int(reference.seed_shape[1]))
        max_area = int(
            max(
                reference.seed_area + 1,
                reference.seed_area
                * scale_x
                * scale_y
                * max(1.0, float(self.config.max_seed_area_growth)),
            )
        )

        anchor_mask = _scaled_seed_mask(reference, (height, width))
        if prior_mask is not None and np.asarray(prior_mask).shape == mask_bool.shape:
            prior_bool = np.asarray(prior_mask, dtype=bool)
            if bool(prior_bool.any()):
                anchor_mask = prior_bool
                roi_mask = _expanded_bbox_mask(
                    prior_bool,
                    padding=float(self.config.search_bbox_padding),
                )
                restricted = mask_bool & roi_mask
                if bool(restricted.any()):
                    mask_bool = restricted

        if int(np.count_nonzero(mask_bool)) <= max_area:
            return _keep_components_near_anchor(mask_bool, anchor_mask)

        capped = _cap_mask_area(
            mask=mask_bool,
            score=score,
            anchor_mask=anchor_mask,
            max_area=max_area,
            spatial_prior_weight=float(self.config.spatial_prior_weight),
        )
        if bool(capped.any()):
            return _keep_components_near_anchor(capped, anchor_mask)
        return _keep_components_near_anchor(mask_bool, anchor_mask)

    def _predict_patch_mask(
        self,
        *,
        feats: torch.Tensor,
        debiased: torch.Tensor,
        cluster_labels: torch.Tensor,
        original_prototypes: torch.Tensor,
        debiased_prototypes: torch.Tensor,
        reference: _ReferenceObject,
        prototype_similarity: torch.Tensor,
        label_competition_mask: Optional[torch.Tensor],
        num_clusters: int,
    ) -> torch.Tensor:
        candidate_mask = self._locate_candidate_mask(
            debiased=debiased,
            reference=reference,
            prototype_similarity=prototype_similarity,
        )
        if label_competition_mask is not None:
            candidate_mask = candidate_mask & label_competition_mask
        matched_mask = candidate_mask & (cluster_labels >= 0)
        if not bool(matched_mask.any()):
            return candidate_mask

        matched_ids, matched_counts = cluster_labels[matched_mask].unique(
            return_counts=True
        )
        all_ids, all_counts = cluster_labels[cluster_labels >= 0].unique(
            return_counts=True
        )
        areas = torch.zeros(num_clusters, dtype=torch.float32, device=feats.device)
        areas[all_ids] = all_counts.float()
        area_weights = torch.zeros(
            num_clusters, dtype=torch.float32, device=feats.device
        )
        area_weights[matched_ids] = matched_counts.float() / areas[
            matched_ids
        ].clamp_min(1)

        cross_matched = debiased_prototypes[matched_ids] @ reference.prototype
        seed_id = int(matched_ids[int(torch.argmax(cross_matched).item())].item())

        intra_sim = original_prototypes @ original_prototypes[seed_id]
        cross_sim = torch.zeros(num_clusters, dtype=torch.float32, device=feats.device)
        for cluster_id in range(num_clusters):
            idx = cluster_labels == cluster_id
            if bool(idx.any()):
                cross_sim[cluster_id] = prototype_similarity[idx].mean()

        combined = cross_sim * intra_sim
        area_weights[seed_id] = 1.0
        combined = combined * area_weights

        final_mask = torch.zeros_like(cluster_labels, dtype=torch.bool)
        valid = cluster_labels >= 0
        final_mask[valid] = combined[cluster_labels[valid]] > float(
            self.config.merge_threshold
        )
        if label_competition_mask is not None:
            final_mask = final_mask & label_competition_mask
        if not bool(final_mask.any()):
            return candidate_mask
        candidate_area = int(candidate_mask.sum().item())
        final_area = int(final_mask.sum().item())
        reference_area = max(1, int(reference.mask.sum().item()))
        max_growth = max(1.0, float(self.config.max_cluster_area_growth))
        max_seed_growth = max(1.0, float(self.config.max_seed_area_growth))
        if (
            candidate_area > 0
            and final_area > max(candidate_area + 1, candidate_area * max_growth)
        ) or final_area > max(reference_area + 1, reference_area * max_seed_growth):
            logger.debug(
                "INSID3 cluster aggregation rejected runaway mask for '%s': "
                "candidate_area=%s reference_area=%s final_area=%s "
                "max_growth=%.2f max_seed_growth=%.2f",
                reference.label,
                candidate_area,
                reference_area,
                final_area,
                max_growth,
                max_seed_growth,
            )
            return candidate_mask
        return final_mask

    @staticmethod
    def _locate_candidate_mask(
        *,
        debiased: torch.Tensor,
        reference: _ReferenceObject,
        prototype_similarity: torch.Tensor,
    ) -> torch.Tensor:
        forward_mask = prototype_similarity > 0
        if not bool(forward_mask.any()):
            forward_mask = prototype_similarity >= torch.quantile(
                prototype_similarity.reshape(-1),
                0.9,
            )

        channels, grid_h, grid_w = debiased.shape
        target_flat = debiased.reshape(channels, -1).transpose(0, 1)
        ref_flat = reference.features.reshape(channels, -1)
        best_ref = torch.argmax(target_flat @ ref_flat, dim=1)
        ref_mask_flat = reference.mask.reshape(-1)
        backward_mask = ref_mask_flat[best_ref].reshape(grid_h, grid_w)
        return forward_mask & backward_mask

    @staticmethod
    def _cluster_features(feats: torch.Tensor, *, tau: float) -> torch.Tensor:
        flat = feats.reshape(feats.shape[0], -1).transpose(0, 1)
        labels = _sklearn_agglomerative_labels(flat, tau)
        if labels is not None:
            return labels.reshape(feats.shape[1], feats.shape[2]).to(feats.device)
        return _neighbor_connected_component_labels(feats, tau=tau)

    @staticmethod
    def _cluster_prototypes(
        features: torch.Tensor,
        labels: torch.Tensor,
        num_clusters: int,
    ) -> torch.Tensor:
        protos: list[torch.Tensor] = []
        for cluster_id in range(num_clusters):
            idx = labels == cluster_id
            if bool(idx.any()):
                proto = features[idx].mean(dim=0)
            else:
                proto = torch.zeros_like(features[0])
            protos.append(F.normalize(proto, p=2, dim=0).unsqueeze(0))
        return torch.cat(protos, dim=0)


def _sklearn_agglomerative_labels(
    features: torch.Tensor,
    tau: float,
) -> Optional[torch.Tensor]:
    try:
        from sklearn.cluster import AgglomerativeClustering
    except Exception:
        return None
    sim = (features @ features.transpose(0, 1)).clamp(-1, 1)
    dist = (1.0 - sim).cpu().numpy()
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=float(1.0 - tau),
        )
    except TypeError:  # scikit-learn < 1.2
        clustering = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage="average",
            distance_threshold=float(1.0 - tau),
        )
    labels = clustering.fit_predict(dist)
    return torch.from_numpy(labels.astype(np.int64))


def _neighbor_connected_component_labels(
    feats: torch.Tensor, *, tau: float
) -> torch.Tensor:
    _, grid_h, grid_w = feats.shape
    labels = torch.full((grid_h, grid_w), -1, dtype=torch.long, device=feats.device)
    current = 0
    for r_idx in range(grid_h):
        for c_idx in range(grid_w):
            if int(labels[r_idx, c_idx].item()) >= 0:
                continue
            seed = feats[:, r_idx, c_idx]
            stack = [(r_idx, c_idx)]
            labels[r_idx, c_idx] = current
            while stack:
                rr, cc = stack.pop()
                center = feats[:, rr, cc]
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if nr < 0 or nc < 0 or nr >= grid_h or nc >= grid_w:
                        continue
                    if int(labels[nr, nc].item()) >= 0:
                        continue
                    sim = max(
                        float(torch.dot(center, feats[:, nr, nc]).item()),
                        float(torch.dot(seed, feats[:, nr, nc]).item()),
                    )
                    if sim >= float(tau):
                        labels[nr, nc] = current
                        stack.append((nr, nc))
            current += 1
    return labels


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if xs.size == 0 or ys.size == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _scaled_seed_mask(
    reference: _ReferenceObject,
    output_size: Tuple[int, int],
) -> np.ndarray:
    height, width = int(output_size[0]), int(output_size[1])
    x1, y1, x2, y2 = reference.seed_bbox
    src_h, src_w = reference.seed_shape
    scale_x = width / max(1, int(src_w))
    scale_y = height / max(1, int(src_h))
    sx1 = int(np.clip(np.floor(x1 * scale_x), 0, width - 1))
    sx2 = int(np.clip(np.ceil((x2 + 1) * scale_x) - 1, 0, width - 1))
    sy1 = int(np.clip(np.floor(y1 * scale_y), 0, height - 1))
    sy2 = int(np.clip(np.ceil((y2 + 1) * scale_y) - 1, 0, height - 1))
    out = np.zeros((height, width), dtype=bool)
    out[sy1 : sy2 + 1, sx1 : sx2 + 1] = True
    return out


def _expanded_bbox_mask(mask: np.ndarray, *, padding: float) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    bbox = _mask_bbox(mask_bool)
    out = np.zeros_like(mask_bool, dtype=bool)
    if bbox is None:
        return out
    x1, y1, x2, y2 = bbox
    height, width = mask_bool.shape
    box_w = max(1, x2 - x1 + 1)
    box_h = max(1, y2 - y1 + 1)
    pad_x = int(round(box_w * max(0.0, float(padding))))
    pad_y = int(round(box_h * max(0.0, float(padding))))
    rx1 = max(0, x1 - pad_x)
    rx2 = min(width - 1, x2 + pad_x)
    ry1 = max(0, y1 - pad_y)
    ry2 = min(height - 1, y2 + pad_y)
    out[ry1 : ry2 + 1, rx1 : rx2 + 1] = True
    return out


def _keep_components_near_anchor(
    mask: np.ndarray, anchor_mask: np.ndarray
) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    if not bool(mask_bool.any()):
        return mask_bool
    anchor_bool = np.asarray(anchor_mask, dtype=bool)
    num_labels, labels = _connected_components(mask_bool)
    if num_labels <= 2:
        return mask_bool

    keep = np.zeros_like(mask_bool, dtype=bool)
    best_label = 0
    best_score = -1.0
    anchor_yx = np.column_stack(np.nonzero(anchor_bool))
    anchor_center = (
        anchor_yx.mean(axis=0)
        if anchor_yx.size
        else np.array(mask_bool.shape, dtype=np.float32) / 2.0
    )
    max_dist = max(float(max(mask_bool.shape)), 1.0)
    for label_idx in range(1, num_labels):
        component = labels == label_idx
        overlap = int(np.logical_and(component, anchor_bool).sum())
        yx = np.column_stack(np.nonzero(component))
        center = yx.mean(axis=0)
        distance = float(np.linalg.norm(center - anchor_center)) / max_dist
        score = float(overlap) - distance
        if score > best_score:
            best_score = score
            best_label = label_idx
    if best_label > 0:
        keep[labels == best_label] = True
        return keep
    return mask_bool


def _cap_mask_area(
    *,
    mask: np.ndarray,
    score: np.ndarray,
    anchor_mask: np.ndarray,
    max_area: int,
    spatial_prior_weight: float,
) -> np.ndarray:
    mask_bool = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(mask_bool))
    if count <= max_area:
        return mask_bool.copy()

    score_arr = np.asarray(score, dtype=np.float32)
    if score_arr.shape != mask_bool.shape:
        score_arr = np.zeros_like(mask_bool, dtype=np.float32)
    yx = np.column_stack(np.nonzero(mask_bool))
    if yx.size == 0:
        return mask_bool.copy()

    anchor_yx = np.column_stack(np.nonzero(np.asarray(anchor_mask, dtype=bool)))
    anchor_center = (
        anchor_yx.mean(axis=0)
        if anchor_yx.size
        else np.array(mask_bool.shape, dtype=np.float32) / 2.0
    )
    distance = np.linalg.norm(yx.astype(np.float32) - anchor_center, axis=1)
    distance = distance / max(float(max(mask_bool.shape)), 1.0)
    ranking = score_arr[mask_bool] - float(spatial_prior_weight) * distance
    keep_count = max(1, min(int(max_area), count))
    keep_indices = np.argpartition(ranking, -keep_count)[-keep_count:]
    capped = np.zeros_like(mask_bool, dtype=bool)
    capped[yx[keep_indices, 0], yx[keep_indices, 1]] = True
    return capped


def _connected_components(mask: np.ndarray) -> tuple[int, np.ndarray]:
    try:
        import cv2

        return cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=8)
    except Exception:
        from scipy import ndimage

        labels, count = ndimage.label(np.asarray(mask, dtype=bool))
        return int(count) + 1, labels.astype(np.int32, copy=False)


class Insid3VideoProcessor:
    """Annolid video processor wrapper for INSID3-style segmentation."""

    _RUNTIME_KWARG_EXCLUDE = {
        "video_path",
        "model_name",
        "results_folder",
        "start_frame",
        "end_frame",
        "step",
        "mem_every",
        "point_tracking",
        "has_occlusion",
        "save_video_with_color_mask",
        "is_cutie",
        "is_new_segment",
    }

    def __init__(self, video_path: str, *args: Any, **kwargs: Any) -> None:
        _ = args
        self.video_path = str(video_path)
        self.video_loader = CV2Video(self.video_path)
        self.num_frames = int(self.video_loader.total_frames())
        self.results_folder = Path(
            kwargs.get("results_folder") or Path(video_path).with_suffix("")
        )
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.pred_worker: Optional[Any] = None
        runtime = {
            key: value
            for key, value in kwargs.items()
            if key not in self._RUNTIME_KWARG_EXCLUDE
        }
        model_name = str(
            runtime.pop(
                "patch_model_name",
                runtime.pop(
                    "dinov3_model_name",
                    "facebook/dinov3-vits16-pretrain-lvd1689m",
                ),
            )
        )
        self.config = Insid3VideoConfig(
            model_name=model_name,
            short_side=int(
                runtime.pop("insid3_short_side", runtime.pop("short_side", 768))
            ),
            svd_components=int(
                runtime.pop("insid3_svd_components", runtime.pop("svd_components", 20))
            ),
            tau=float(runtime.pop("insid3_tau", runtime.pop("tau", 0.6))),
            merge_threshold=float(
                runtime.pop(
                    "insid3_merge_threshold", runtime.pop("merge_threshold", 0.2)
                )
            ),
            max_cluster_area_growth=float(
                runtime.pop(
                    "insid3_max_cluster_area_growth",
                    runtime.pop("max_cluster_area_growth", 8.0),
                )
            ),
            max_seed_area_growth=float(
                runtime.pop(
                    "insid3_max_seed_area_growth",
                    runtime.pop("max_seed_area_growth", 3.0),
                )
            ),
            label_competition_margin=float(
                runtime.pop(
                    "insid3_label_competition_margin",
                    runtime.pop("label_competition_margin", 0.02),
                )
            ),
            search_bbox_padding=float(
                runtime.pop(
                    "insid3_search_bbox_padding",
                    runtime.pop("search_bbox_padding", 0.75),
                )
            ),
            spatial_prior_weight=float(
                runtime.pop(
                    "insid3_spatial_prior_weight",
                    runtime.pop("spatial_prior_weight", 0.25),
                )
            ),
            crf_refine=_runtime_bool(
                runtime.pop("insid3_crf_refine", runtime.pop("crf_refine", False))
            ),
            crf_backend=str(
                runtime.pop("insid3_crf_backend", runtime.pop("crf_backend", "auto"))
            ),
            crf_band_px=int(
                runtime.pop("insid3_crf_band_px", runtime.pop("crf_band_px", 10))
            ),
            crf_p_core=float(
                runtime.pop("insid3_crf_p_core", runtime.pop("crf_p_core", 0.95))
            ),
            crf_iterations=int(
                runtime.pop("insid3_crf_iterations", runtime.pop("crf_iterations", 10))
            ),
            device=runtime.pop("device", None),
        )
        self.segmenter = Insid3VideoSegmenter(config=self.config)

    def set_pred_worker(self, pred_worker: Any) -> None:
        self.pred_worker = pred_worker

    def get_total_frames(self) -> int:
        return int(self.num_frames)

    def cleanup(self) -> None:
        try:
            self.video_loader.release()
        except Exception:
            pass

    def process_video_frames(self, *args: Any, **kwargs: Any) -> str:
        _ = args
        if self._should_stop():
            return "INSID3 video segmentation stopped by user."

        adapter = AnnotationAdapter(
            image_height=int(self.video_loader.get_height()),
            image_width=int(self.video_loader.get_width()),
            description="INSID3",
            persist_json=False,
            mask_description="INSID3",
        )
        try:
            seed_frame, registry = adapter.load_initial_state(self.results_folder)
        except RuntimeError as exc:
            return str(exc)

        masks = self._mask_lookup_from_registry(adapter, registry)
        if not masks:
            return "No polygon seed masks found for INSID3 video segmentation."

        seed_image = self._frame_image(seed_frame)
        references = self.segmenter.build_references(seed_image, masks)
        if not references:
            return "No valid INSID3 reference masks after patch-grid downsampling."

        start_frame = max(0, int(kwargs.get("start_frame", seed_frame)))
        end_arg = kwargs.get("end_frame", self.num_frames - 1)
        end_frame = min(self.num_frames - 1, int(end_arg))
        if end_frame < start_frame:
            return "No frames selected for INSID3 video segmentation."

        total = max(1, end_frame - start_frame + 1)
        output_size = (
            int(self.video_loader.get_height()),
            int(self.video_loader.get_width()),
        )
        written = 0
        prior_masks: Dict[str, np.ndarray] = {
            label: np.asarray(mask, dtype=bool).copy() for label, mask in masks.items()
        }
        for offset, frame_number in enumerate(
            range(start_frame, end_frame + 1), start=1
        ):
            if self._should_stop():
                return "INSID3 video segmentation stopped by user."
            frame_image = self._frame_image(frame_number)
            masks_by_label = self.segmenter.segment(
                frame_image,
                references,
                output_size=output_size,
                priors=prior_masks,
            )
            frame_registry = InstanceRegistry()
            for label, mask in masks_by_label.items():
                polygon = self._mask_to_polygon(mask)
                if polygon:
                    frame_registry.ensure_instance(label).set_mask(
                        bitmap=mask,
                        polygon=polygon,
                    )
            adapter.write_annotation(
                frame_number=frame_number,
                registry=frame_registry,
                output_dir=self.results_folder,
            )
            for label, mask in masks_by_label.items():
                if np.asarray(mask).any():
                    prior_masks[label] = np.asarray(mask, dtype=bool).copy()
            written += 1
            self._report_progress(offset, total)

        return f"INSID3 video segmentation completed for {written} frames."

    def _frame_image(self, frame_number: int) -> Image.Image:
        frame = self.video_loader.load_frame(int(frame_number))
        return Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")

    @staticmethod
    def _mask_lookup_from_registry(
        adapter: AnnotationAdapter,
        registry: InstanceRegistry,
    ) -> Dict[str, np.ndarray]:
        masks: Dict[str, np.ndarray] = {}
        for instance in registry:
            mask = instance.mask_bitmap
            if mask is None and instance.polygon is not None:
                mask = adapter.mask_bitmap_from_polygon(instance.polygon)
            if mask is not None and np.asarray(mask).any():
                masks[instance.label] = np.asarray(mask, dtype=bool)
        return masks

    @staticmethod
    def _mask_to_polygon(mask: np.ndarray) -> list[tuple[float, float]]:
        polygons, _has_holes = mask_to_polygons(
            mask.astype(np.uint8),
            simplify=True,
            epsilon_ratio=0.0025,
            min_epsilon=0.5,
            max_epsilon=4.0,
        )
        if not polygons:
            return []
        largest = max(polygons, key=lambda arr: len(arr))
        points = np.asarray(largest, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < 3:
            return []
        out = [(float(x), float(y)) for x, y in points]
        if out[0] != out[-1]:
            out.append(out[0])
        return out

    def _report_progress(self, processed: int, total: int) -> None:
        if total <= 0 or self.pred_worker is None:
            return
        progress = int(min(100, max(0, (processed / total) * 100)))
        if hasattr(self.pred_worker, "report_progress"):
            self.pred_worker.report_progress(progress)
        elif hasattr(self.pred_worker, "progress_signal"):
            self.pred_worker.progress_signal.emit(progress)

    def _should_stop(self) -> bool:
        return bool(
            self.pred_worker is not None
            and hasattr(self.pred_worker, "is_stopped")
            and self.pred_worker.is_stopped()
        )


def _runtime_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "on", "enabled"}
