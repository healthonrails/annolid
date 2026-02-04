from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.cm as cm

from annolid.features import Dinov3FeatureExtractor, Dinov3Config

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    heat01: np.ndarray  # (gh, gw) in [0,1]
    box_xyxy: Tuple[int, int, int, int]
    overlay: Optional[Image.Image] = None


class DinoPatchSimilarity:
    """Compute intra-image patch similarity heatmaps using DINOv3 features."""

    def __init__(self, extractor: Optional[Dinov3FeatureExtractor] = None) -> None:
        self.extractor = extractor or Dinov3FeatureExtractor(Dinov3Config())

    @staticmethod
    def _overlay(
        base: Image.Image,
        heat01: np.ndarray,
        *,
        alpha: float = 0.55,
        box: Optional[Tuple[int, int, int, int]] = None,
        colormap: str = "inferno",
    ) -> Image.Image:
        H, W = base.height, base.width
        heat_img = Image.fromarray((heat01 * 255).astype(np.uint8)).resize(
            (W, H), resample=Image.NEAREST
        )
        rgba = (cm.get_cmap(colormap)(np.asarray(heat_img) / 255.0) * 255).astype(
            np.uint8
        )
        ov = Image.fromarray(rgba, "RGBA")
        ov.putalpha(int(alpha * 255))
        out = Image.alpha_composite(base.convert("RGBA"), ov)
        if box is not None:
            from PIL import ImageDraw

            ImageDraw.Draw(out, "RGBA").rectangle(
                box, outline=(255, 255, 255, 220), width=2
            )
        return out

    @torch.inference_mode()
    def similarity(
        self,
        image: Image.Image,
        click_xy: Tuple[int, int],
        *,
        alpha: float = 0.55,
        return_overlay: bool = True,
    ) -> SimilarityResult:
        """Compute cosine similarity of every patch vs the clicked patch.

        Returns a normalized heatmap and (optionally) a rendered overlay.
        """
        # Extract features [D,h,w]
        feats = self.extractor.extract(
            image, return_layer="last", normalize=True
        )  # [D,h,w] CPU
        if isinstance(feats, np.ndarray):
            feats_t = torch.from_numpy(feats)
        else:
            feats_t = feats

        D, h, w = feats_t.shape
        base = image.convert("RGB")

        # Map click to patch index using original image size → resized grid
        # We reconstruct the resized size via extractor utilities
        new_h, new_w = self.extractor._compute_resized_hw(*base.size)
        scale_x = new_w / base.width
        scale_y = new_h / base.height
        px_x, px_y = new_w / w, new_h / h

        click_x_resized = click_xy[0] * scale_x
        click_y_resized = click_xy[1] * scale_y

        i = min(max(int(click_x_resized / px_x), 0), w - 1)
        j = min(max(int(click_y_resized / px_y), 0), h - 1)

        # Cosine similarity (prefer device matmul if available)
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        feats_dev = feats_t.to(device)
        flat = F.normalize(feats_dev.permute(1, 2, 0).reshape(-1, D), dim=1)
        v = F.normalize(feats_dev[:, j, i].reshape(1, D), dim=1)
        sims = (flat @ v.T).reshape(h, w)

        # Normalize to [0,1]
        smin, smax = float(sims.min()), float(sims.max())
        heat01 = ((sims - smin) / (smax - smin + 1e-12)).detach().to("cpu").numpy()

        # Box in original image coordinates
        bx0 = int(i * (base.width / w))
        by0 = int(j * (base.height / h))
        bx1 = int((i + 1) * (base.width / w))
        by1 = int((j + 1) * (base.height / h))
        box = (bx0, by0, bx1, by1)

        overlay = None
        if return_overlay:
            overlay = self._overlay(base, heat01, alpha=alpha, box=box)

        return SimilarityResult(heat01=heat01, box_xyxy=box, overlay=overlay)


# -------------------------
#           CLI
# -------------------------


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="DINOv3 patch similarity heatmap")
    ap.add_argument("--image", required=True, help="Path to image")
    ap.add_argument(
        "--model",
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="Hugging Face model id or legacy alias (e.g., facebook/dinov3-vits16-pretrain-lvd1689m)",
    )
    ap.add_argument(
        "--short-side",
        type=int,
        default=768,
        help="Target short side before snapping to patch multiple",
    )
    ap.add_argument(
        "--click",
        required=False,
        help="Click 'x,y' in original image pixels (defaults to center)",
    )
    ap.add_argument(
        "--opacity", type=float, default=0.55, help="Overlay alpha in [0,1]"
    )
    ap.add_argument("--out", default="overlay.png", help="Output image path")
    ap.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Force compute device",
    )
    ap.add_argument(
        "--no-overlay",
        action="store_true",
        help="Only write heatmap .npy; skip PNG overlay",
    )
    return ap


def _parse_click(val: Optional[str], w: int, h: int) -> Tuple[int, int]:
    if not val:
        return (w // 2, h // 2)
    try:
        xs, ys = val.split(",")
        return int(xs), int(ys)
    except Exception as e:
        raise SystemExit(f"Invalid --click value '{val}': {e}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ap = _build_argparser()
    args = ap.parse_args()

    # Build extractor per user choices
    cfg = Dinov3Config(
        model_name=args.model, short_side=args.short_side, device=args.device
    )
    engine = DinoPatchSimilarity(Dinov3FeatureExtractor(cfg))

    img = Image.open(args.image).convert("RGB")
    click = _parse_click(args.click, img.width, img.height)

    res = engine.similarity(
        img, click_xy=click, alpha=args.opacity, return_overlay=(not args.no_overlay)
    )

    # Save outputs
    np.save(args.out.replace(".png", "_heat.npy"), res.heat01)
    logger.info("Saved heatmap: %s", args.out.replace(".png", "_heat.npy"))

    if not args.no_overlay and res.overlay is not None:
        res.overlay.save(args.out)
        logger.info("Saved overlay: %s", args.out)


if __name__ == "__main__":  # pragma: no cover
    main()
