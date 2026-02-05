from __future__ import annotations

from qtpy import QtCore

from annolid.utils.logger import logger


class LifecycleMixin:
    """App lifecycle and teardown helpers."""

    def _stop_frame_loader(self):
        """Tear down the frame loader safely from its owning thread."""
        loader = getattr(self, "frame_loader", None)
        if loader is None:
            return

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
        try:
            dialog = getattr(self, "_training_dashboard_dialog", None)
            if dialog is not None:
                dialog.close()
        except Exception:
            pass
        try:
            from annolid.gui.tensorboard import stop_tensorboard

            stop_tensorboard()
        except Exception:
            pass
