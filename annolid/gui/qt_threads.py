from __future__ import annotations

from typing import TypeVar

_ThreadT = TypeVar("_ThreadT")

# macOS secondary pthreads default to a 512 KB stack unless overridden before
# start(), which is too small for deep native C++/OpenMP/ONNX/PyTorch/OpenCV
# call chains executed inside background QThread workers.
DEFAULT_WORKER_THREAD_STACK_SIZE = 8 * 1024 * 1024


def configure_worker_thread_stack(
    thread: _ThreadT,
    stack_size: int = DEFAULT_WORKER_THREAD_STACK_SIZE,
) -> _ThreadT:
    """Set a QThread's stack size before start() when supported by the thread object."""
    set_stack_size = getattr(thread, "setStackSize", None)
    if callable(set_stack_size):
        set_stack_size(int(stack_size))
    return thread
