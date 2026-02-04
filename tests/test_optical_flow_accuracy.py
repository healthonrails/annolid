import numpy as np
import cv2
import pytest

try:
    from annolid.motion.raft_wrapper import RAFTOpticalFlow
except Exception:
    RAFTOpticalFlow = None


def _make_shifted_frames(shift_x: float = 1.0, shift_y: float = 0.0, size: int = 64):
    """Create a simple synthetic pair of frames with a translated blob."""
    prev = np.zeros((size, size, 3), dtype=np.uint8)
    curr = np.zeros_like(prev)
    center_prev = (size // 2, size // 2)
    center_curr = (int(center_prev[0] + shift_x), int(center_prev[1] + shift_y))
    cv2.circle(prev, center_prev, 6, (255, 255, 255), -1)
    cv2.circle(curr, center_curr, 6, (255, 255, 255), -1)
    mask = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) > 0
    return prev, curr, mask


@pytest.mark.skipif(
    RAFTOpticalFlow is None, reason="torchvision optical_flow not available"
)
def test_raft_flow_detects_shift():
    import torch  # Local import to allow skipif when torch is absent

    prev, curr, mask = _make_shifted_frames(shift_x=1.0, shift_y=0.5, size=128)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    prev_t = torch.from_numpy(prev).permute(2, 0, 1).unsqueeze(0).float().to(device)
    curr_t = torch.from_numpy(curr).permute(2, 0, 1).unsqueeze(0).float().to(device)

    raft = RAFTOpticalFlow(model_type="small", device=device)
    with torch.inference_mode():
        flow_t = raft(prev_t, curr_t).squeeze(0).permute(1, 2, 0).cpu().numpy()

    mask_f = mask.astype(np.float32)
    mean_flow = (flow_t * mask_f[..., None]).sum(axis=(0, 1)) / mask_f.sum()

    # Expect measurable displacement (direction may differ by convention)
    assert np.abs(mean_flow[0]) > 0.2
    assert np.abs(mean_flow[1]) > 0.1
    assert np.isfinite(flow_t).all()
