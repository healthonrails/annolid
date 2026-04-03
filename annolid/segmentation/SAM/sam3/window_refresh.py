from __future__ import annotations

from typing import Callable, Optional, Tuple


def compute_mid_window_refresh_index(
    frame_count: int,
    propagation_direction: Optional[str],
    *,
    min_window_size: int = 4,
) -> Optional[int]:
    """
    Return the midpoint frame index for a second-pass refresh, or None.

    The shared policy is intentionally conservative:
    - only enable for forward-style propagation
    - only enable for windows large enough to justify a second pass
    - always keep the refresh inside the current window
    """
    direction = (propagation_direction or "").strip().lower()
    if direction != "forward":
        return None

    frame_count = int(frame_count or 0)
    if frame_count < max(1, int(min_window_size or 1)):
        return None

    refresh_local_idx = max(1, frame_count // 2)
    if refresh_local_idx >= frame_count:
        return None
    return int(refresh_local_idx)


def run_mid_window_refresh(
    frame_count: int,
    propagation_direction: Optional[str],
    *,
    seed_first_frame: Callable[[], None],
    propagate_segment: Callable[[int, int], Tuple[int, int]],
    refresh_mid_frame: Optional[Callable[[int], Tuple[int, int]]] = None,
    min_window_size: int = 4,
) -> Tuple[int, int, Optional[int]]:
    """
    Execute a window pass with an optional midpoint refresh.

    The callback contract stays deliberately small:
    - `seed_first_frame()` must seed the current window at local frame 0
    - `propagate_segment(start_idx, length)` must execute propagation over that
      local segment and return `(frames_processed, masks_written)`
    - `refresh_mid_frame(idx)` is called only when a midpoint refresh is active
    """
    refresh_local_idx = compute_mid_window_refresh_index(
        frame_count,
        propagation_direction,
        min_window_size=min_window_size,
    )

    seed_first_frame()

    if refresh_local_idx is None or refresh_mid_frame is None:
        frames_processed, masks_written = propagate_segment(0, int(frame_count))
        return int(frames_processed), int(masks_written), None

    first_frames, first_masks = propagate_segment(0, int(refresh_local_idx))
    refresh_frames, refresh_masks = refresh_mid_frame(int(refresh_local_idx))
    second_frames, second_masks = propagate_segment(
        int(refresh_local_idx) + 1,
        max(0, int(frame_count) - int(refresh_local_idx) - 1),
    )
    return (
        int(first_frames) + int(refresh_frames) + int(second_frames),
        int(first_masks) + int(refresh_masks) + int(second_masks),
        int(refresh_local_idx),
    )
