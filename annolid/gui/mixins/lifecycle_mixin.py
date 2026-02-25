from __future__ import annotations

import logging
import time
from qtpy import QtCore

from annolid.utils.logger import logger


def _iter_effective_handlers(log: logging.Logger):
    cur: logging.Logger | None = log
    while cur is not None:
        for handler in cur.handlers:
            yield handler
        if not cur.propagate:
            break
        cur = cur.parent  # type: ignore[assignment]


def _has_closed_stream_handler(log: logging.Logger) -> bool:
    for handler in _iter_effective_handlers(log):
        stream = getattr(handler, "stream", None)
        if stream is not None and bool(getattr(stream, "closed", False)):
            return True
    return False


def _safe_info(message: str, *args) -> None:
    """Best-effort logging that tolerates interpreter/shutdown stream teardown."""
    if _has_closed_stream_handler(logger):
        return
    try:
        logger.info(message, *args)
    except (ValueError, RuntimeError):
        # Logging handlers/streams may already be closed during late shutdown.
        return


class LifecycleMixin:
    """App lifecycle and teardown helpers."""

    def _stop_frame_loader(self):
        """Tear down the frame loader safely from its owning thread."""
        loader = getattr(self, "frame_loader", None)
        if loader is None:
            return
        stop_start = time.perf_counter()

        old_loader = loader
        try:
            target_thread = old_loader.thread()
            current_thread = QtCore.QThread.currentThread()
            if target_thread is None or not target_thread.isRunning():
                if target_thread is not current_thread:
                    try:
                        old_loader.moveToThread(current_thread)
                    except RuntimeError:
                        logger.debug(
                            "Unable to move frame loader to current thread during shutdown.",
                            exc_info=True,
                        )
                old_loader.shutdown()
            elif target_thread is current_thread:
                old_loader.shutdown()
            else:
                QtCore.QMetaObject.invokeMethod(
                    old_loader,
                    "shutdown",
                    QtCore.Qt.BlockingQueuedConnection,
                )
        except RuntimeError:
            logger.debug("Frame loader already cleaned up.", exc_info=True)
        finally:
            if self.frame_loader is old_loader:
                self.frame_loader = None
            elapsed_ms = (time.perf_counter() - stop_start) * 1000.0
            _safe_info("Frame loader stop completed in %.1fms.", elapsed_ms)

    def clean_up(self):
        def quit_and_wait(thread, message):
            if thread is not None:
                try:
                    thread.quit()
                    thread.wait()
                except RuntimeError:
                    logger.info(message)

        self._stop_frame_loader()
        self._stop_csv_worker()
        if hasattr(self, "realtime_manager") and self.realtime_manager:
            try:
                self.realtime_manager.stop_realtime_inference()
            except (AttributeError, RuntimeError, TypeError) as exc:
                logger.debug(
                    "Failed stopping realtime inference during cleanup: %s", exc
                )
        quit_and_wait(self.frame_worker, "Thank you!")
        quit_and_wait(self.seg_train_thread, "See you next time!")
        quit_and_wait(self.seg_pred_thread, "Bye!")
        if hasattr(self, "yolo_training_manager") and self.yolo_training_manager:
            self.yolo_training_manager.cleanup()
        if (
            hasattr(self, "dino_kpseg_training_manager")
            and self.dino_kpseg_training_manager
        ):
            self.dino_kpseg_training_manager.cleanup()
        if hasattr(self, "ai_chat_manager") and self.ai_chat_manager:
            try:
                self.ai_chat_manager.cleanup()
            except (AttributeError, RuntimeError, TypeError) as exc:
                logger.debug("Failed cleaning up AI chat manager: %s", exc)
        try:
            dialog = getattr(self, "_training_dashboard_dialog", None)
            if dialog is not None:
                dialog.close()
        except (AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Failed closing training dashboard dialog: %s", exc)
        try:
            from annolid.gui.tensorboard import stop_tensorboard

            stop_tensorboard()
        except (ImportError, AttributeError, RuntimeError, TypeError) as exc:
            logger.debug("Failed stopping tensorboard during cleanup: %s", exc)
