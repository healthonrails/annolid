import numpy as np
import cv2
import pytest

from annolid.motion.optical_flow import compute_optical_flow

try:
    from annolid.motion.raft_wrapper import (
        RAFTOpticalFlow,
        clear_raft_model_cache,
        get_raft_model,
    )
except Exception:
    RAFTOpticalFlow = None
    clear_raft_model_cache = None
    get_raft_model = None


def _make_shifted_frames(shift_x: float = 1.0, shift_y: float = 0.0, size: int = 64):
    """Create a simple synthetic pair of frames with a translated blob."""
    prev = np.zeros((size, size, 3), dtype=np.uint8)
    curr = np.zeros_like(prev)
    center_prev = (size // 2, size // 2)
    center_curr = (
        int(round(center_prev[0] + shift_x)),
        int(round(center_prev[1] + shift_y)),
    )
    cv2.circle(prev, center_prev, 10, (50, 140, 240), -1)
    cv2.circle(curr, center_curr, 10, (50, 140, 240), -1)
    mask = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) > 0
    return prev, curr, mask


@pytest.mark.skipif(
    RAFTOpticalFlow is None, reason="torchvision optical_flow not available"
)
def test_raft_flow_detects_shift():
    import torch  # Local import to allow skipif when torch is absent

    prev, curr, mask = _make_shifted_frames(shift_x=4.0, shift_y=-2.0, size=128)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    prev_rgb = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
    curr_rgb = cv2.cvtColor(curr, cv2.COLOR_BGR2RGB)
    prev_u8 = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    curr_u8 = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    prev_f255 = prev_u8.float()
    curr_f255 = curr_u8.float()
    prev_f1 = prev_f255 / 255.0
    curr_f1 = curr_f255 / 255.0

    raft = get_raft_model(model_type="small", device=device)
    assert get_raft_model(model_type="small", device=device) is raft

    with torch.inference_mode():
        flow_u8 = raft(prev_u8, curr_u8).squeeze(0).permute(1, 2, 0).cpu().numpy()
        flow_f255 = raft(prev_f255, curr_f255).squeeze(0).permute(1, 2, 0).cpu().numpy()
        flow_f1 = raft(prev_f1, curr_f1).squeeze(0).permute(1, 2, 0).cpu().numpy()

    np.testing.assert_allclose(flow_f255, flow_u8, atol=1e-4)
    np.testing.assert_allclose(flow_f1, flow_u8, atol=1e-4)

    mean_flow = flow_u8[mask].mean(axis=0)
    assert np.isfinite(flow_u8).all()
    assert 1.5 < mean_flow[0] < 7.0
    assert -4.5 < mean_flow[1] < -0.5


@pytest.mark.skipif(
    RAFTOpticalFlow is None, reason="torchvision optical_flow not available"
)
def test_compute_optical_flow_raft_handles_grayscale_and_odd_dims():
    prev, curr, mask = _make_shifted_frames(shift_x=3.0, shift_y=-2.0, size=100)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    flow_hsv, flow = compute_optical_flow(
        prev_gray,
        curr_gray,
        use_raft=True,
        raft_model="small",
        torch_device="cpu",
    )

    assert flow.shape == (100, 100, 2)
    assert flow_hsv.shape == (100, 100, 3)
    assert np.isfinite(flow).all()
    mean_flow = flow[mask].mean(axis=0)
    assert mean_flow[0] > 1.0
    assert mean_flow[1] < -0.5
