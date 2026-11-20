"""BORIS-style synchronized secondary camera grid and external time-series plotter."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from annolid.behavior.external_data import (
    ExternalDataError,
    ExternalDataStreamSpec,
    LoadedExternalStream,
    inspect_external_data_columns,
    load_external_data_stream,
    write_external_data_svg,
)
from annolid.behavior.time_budget import BehaviorInterval
from annolid.core.media.audio import (
    AudioBuffer,
    AudioTrackMixSpec,
    mix_synchronized_audio_buffers,
)
from annolid.data import videos

_STREAM_PALETTE: Tuple[str, ...] = (
    "#2563eb",
    "#16a34a",
    "#ea580c",
    "#9333ea",
    "#0891b2",
    "#dc2626",
)


def _open_media_reader(video_path: Path):
    lower_name = video_path.name.lower()
    if lower_name.endswith((".tif", ".tiff", ".ome.tif", ".ome.tiff")):
        return videos.TiffStackVideo(str(video_path))
    return videos.CV2Video(str(video_path))


class SecondaryVideoTile(QtWidgets.QFrame):
    """Single synchronized secondary camera viewport with millisecond offset & audio mix controls."""

    removeRequested = QtCore.Signal(object)
    offsetChanged = QtCore.Signal(object, float)
    audioConfigChanged = QtCore.Signal(object)

    def __init__(
        self,
        video_path: Path | str,
        *,
        player_id: str = "2",
        offset_s: float = 0.0,
        audio_buffer: Optional[AudioBuffer] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.video_path = Path(video_path).expanduser().resolve()
        self.player_id = str(player_id)
        self._reader = _open_media_reader(self.video_path)
        raw_fps = float(self._reader.get_fps() or 0.0)
        self._fps: float = raw_fps if raw_fps > 0 else 29.97
        self._total_frames: int = max(0, int(self._reader.total_frames() or 0))
        self._displayed_frame: Optional[int] = None
        self._pending_time_s: Optional[float] = None
        self._current_qimage: Optional[QtGui.QImage] = None
        self._audio_buffer: Optional[AudioBuffer] = audio_buffer
        self._audio_load_attempted: bool = audio_buffer is not None

        self._coalesce_timer = QtCore.QTimer(self)
        self._coalesce_timer.setSingleShot(True)
        self._coalesce_timer.setInterval(0)
        self._coalesce_timer.timeout.connect(self._flush_pending_seek)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        hdr = QtWidgets.QHBoxLayout()
        hdr.setSpacing(4)
        title_lbl = QtWidgets.QLabel(
            f"Player #{self.player_id}: {self.video_path.name}", self
        )
        title_lbl.setStyleSheet("font-weight: 600;")
        hdr.addWidget(title_lbl, 1)

        self.audio_mute_btn = QtWidgets.QToolButton(self)
        self.audio_mute_btn.setCheckable(True)
        self.audio_mute_btn.setChecked(True)  # muted by default
        self.audio_mute_btn.setText("🔇")
        self.audio_mute_btn.setToolTip(
            "Toggle secondary camera audio in synchronized playback mix"
        )
        self.audio_mute_btn.toggled.connect(self._on_audio_mute_toggled)
        hdr.addWidget(self.audio_mute_btn)

        self.audio_gain_spin = QtWidgets.QSpinBox(self)
        self.audio_gain_spin.setRange(0, 200)
        self.audio_gain_spin.setSuffix("%")
        self.audio_gain_spin.setValue(100)
        self.audio_gain_spin.setToolTip("Secondary camera audio mix volume (0–200%)")
        self.audio_gain_spin.valueChanged.connect(
            lambda _v: self.audioConfigChanged.emit(self)
        )
        hdr.addWidget(self.audio_gain_spin)

        hdr.addWidget(QtWidgets.QLabel("Offset (s):", self))
        self.offset_spin = QtWidgets.QDoubleSpinBox(self)
        self.offset_spin.setRange(-36000.0, 36000.0)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setSingleStep(0.05)
        self.offset_spin.setValue(float(offset_s))
        self.offset_spin.setToolTip(
            "Time offset added to primary video time: t_cam = t_primary + offset"
        )
        self.offset_spin.valueChanged.connect(self._on_offset_changed)
        hdr.addWidget(self.offset_spin)

        self.remove_btn = QtWidgets.QToolButton(self)
        self.remove_btn.setText("✕")
        self.remove_btn.setToolTip("Close this secondary camera")
        self.remove_btn.clicked.connect(lambda: self.removeRequested.emit(self))
        hdr.addWidget(self.remove_btn)
        lay.addLayout(hdr)

        self.viewport_label = QtWidgets.QLabel(self)
        self.viewport_label.setAlignment(QtCore.Qt.AlignCenter)
        self.viewport_label.setMinimumSize(220, 140)
        self.viewport_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.viewport_label.setStyleSheet(
            "background-color: #0f172a; color: #94a3b8; border-radius: 4px;"
        )
        lay.addWidget(self.viewport_label, 1)

        self.status_label = QtWidgets.QLabel(
            f"{self._fps:.2f} fps · {self._total_frames} frames", self
        )
        self.status_label.setStyleSheet("color: #64748b; font-size: 11px;")
        lay.addWidget(self.status_label)

        self.sync_to_time(0.0, immediate=True)

    @property
    def offset_s(self) -> float:
        return float(self.offset_spin.value())

    @property
    def displayed_frame(self) -> Optional[int]:
        return self._displayed_frame

    @property
    def is_audio_muted(self) -> bool:
        return bool(self.audio_mute_btn.isChecked())

    @property
    def audio_gain(self) -> float:
        return float(self.audio_gain_spin.value()) / 100.0

    def set_audio_muted(self, muted: bool) -> None:
        self.audio_mute_btn.setChecked(bool(muted))

    def set_audio_gain_percent(self, percent: int) -> None:
        self.audio_gain_spin.setValue(max(0, min(200, int(percent))))

    def ensure_audio_buffer(self) -> Optional[AudioBuffer]:
        if not self._audio_load_attempted:
            self._audio_load_attempted = True
            try:
                self._audio_buffer = AudioBuffer.from_file(
                    str(self.video_path), fps=self._fps
                )
            except Exception:
                self._audio_buffer = None
        return self._audio_buffer

    def set_offset_s(self, value: float) -> None:
        self.offset_spin.setValue(float(value))

    def _on_audio_mute_toggled(self, muted: bool) -> None:
        self.audio_mute_btn.setText("🔇" if muted else "🔊")
        if not muted:
            self.ensure_audio_buffer()
        if self._pending_time_s is not None:
            self._update_status_text(
                self._pending_time_s + self.offset_s, self._displayed_frame or 0
            )
        self.audioConfigChanged.emit(self)

    def _on_offset_changed(self, val: float) -> None:
        self.offsetChanged.emit(self, float(val))
        if not self.is_audio_muted:
            self.audioConfigChanged.emit(self)
        if self._pending_time_s is not None:
            self.sync_to_time(self._pending_time_s, immediate=True)

    def sync_to_time(self, primary_time_s: float, *, immediate: bool = False) -> None:
        self._pending_time_s = float(primary_time_s)
        if immediate:
            self._flush_pending_seek()
        elif not self._coalesce_timer.isActive():
            self._coalesce_timer.start()

    def _update_status_text(
        self, cam_time: float, target_frame: int, clamped_note: str = ""
    ) -> None:
        audio_note = ""
        if not self.is_audio_muted:
            buf = self.ensure_audio_buffer()
            if buf is None:
                audio_note = " · no audio"
            else:
                dbfs = buf.rms_dbfs_at_time(max(0.0, cam_time))
                if dbfs is not None:
                    audio_note = f" · 🔊 {dbfs:.1f} dBFS"
        self.status_label.setText(
            f"Frame {target_frame}/{max(0, self._total_frames - 1)} · "
            f"t={max(0.0, cam_time):.3f}s{clamped_note}{audio_note}"
        )

    def _flush_pending_seek(self) -> None:
        if self._pending_time_s is None or self._reader is None:
            return
        if self._total_frames <= 0:
            self.viewport_label.setText("Empty video")
            return

        cam_time = self._pending_time_s + self.offset_s
        if cam_time < 0.0:
            target_frame = 0
            clamped_note = f" (pre-start {cam_time:.2f}s)"
        else:
            raw_idx = int(round(cam_time * self._fps))
            if raw_idx >= self._total_frames:
                target_frame = self._total_frames - 1
                clamped_note = " (end)"
            else:
                target_frame = raw_idx
                clamped_note = ""

        self._update_status_text(cam_time, target_frame, clamped_note)
        if self._displayed_frame == target_frame and self._current_qimage is not None:
            return

        try:
            rgb = self._reader.load_frame(target_frame)
        except Exception as exc:
            self.viewport_label.setText(f"Decode error at frame {target_frame}: {exc}")
            return

        rgb_contig = np.ascontiguousarray(rgb)
        if rgb_contig.ndim == 2:
            h, w = rgb_contig.shape
            qimg = QtGui.QImage(
                rgb_contig.data, w, h, w, QtGui.QImage.Format_Grayscale8
            ).copy()
        else:
            h, w, ch = rgb_contig.shape
            fmt = (
                QtGui.QImage.Format_RGBA8888 if ch == 4 else QtGui.QImage.Format_RGB888
            )
            qimg = QtGui.QImage(rgb_contig.data, w, h, w * ch, fmt).copy()

        self._displayed_frame = target_frame
        self._current_qimage = qimg
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._current_qimage is None:
            return
        avail = self.viewport_label.size()
        tw = max(64, avail.width() - 4)
        th = max(48, avail.height() - 4)
        pix = QtGui.QPixmap.fromImage(self._current_qimage).scaled(
            tw,
            th,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.viewport_label.setPixmap(pix)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()

    def close_reader(self) -> None:
        self._coalesce_timer.stop()
        reader = self._reader
        self._reader = None
        if reader is not None and hasattr(reader, "release"):
            try:
                reader.release()
            except Exception:
                pass


class ExternalStreamConfigDialog(QtWidgets.QDialog):
    """Modal configuration dialog for adding a synchronized external CSV/TSV trace."""

    def __init__(
        self,
        file_path: Path,
        *,
        default_color: str = "#2563eb",
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure External Data Stream")
        self._file_path = Path(file_path).expanduser().resolve()
        form = QtWidgets.QFormLayout(self)

        path_lbl = QtWidgets.QLabel(self._file_path.name, self)
        path_lbl.setToolTip(str(self._file_path))
        form.addRow("File:", path_lbl)

        col_choices = inspect_external_data_columns(self._file_path)
        self.time_col_combo = QtWidgets.QComboBox(self)
        self.time_col_combo.addItem("(Uniform sampling rate)", userData="")
        for item in col_choices:
            self.time_col_combo.addItem(item, userData=item.split(":", 1)[0].strip())
        if len(col_choices) >= 2:
            self.time_col_combo.setCurrentIndex(1)

        self.val_col_combo = QtWidgets.QComboBox(self)
        for item in col_choices:
            self.val_col_combo.addItem(item, userData=item.split(":", 1)[0].strip())
        if len(col_choices) >= 2:
            self.val_col_combo.setCurrentIndex(1)

        default_var = ""
        if len(col_choices) >= 2 and ":" in col_choices[1]:
            default_var = col_choices[1].split(":", 1)[1].strip()
        self.var_edit = QtWidgets.QLineEdit(default_var or self._file_path.stem, self)
        self.color_edit = QtWidgets.QLineEdit(default_color, self)
        self.offset_spin = QtWidgets.QDoubleSpinBox(self)
        self.offset_spin.setRange(-36000.0, 36000.0)
        self.offset_spin.setDecimals(3)
        self.offset_spin.setSingleStep(0.05)
        self.window_spin = QtWidgets.QDoubleSpinBox(self)
        self.window_spin.setRange(1.0, 86400.0)
        self.window_spin.setDecimals(1)
        self.window_spin.setValue(60.0)
        self.sr_spin = QtWidgets.QDoubleSpinBox(self)
        self.sr_spin.setRange(0.1, 100000.0)
        self.sr_spin.setDecimals(2)
        self.sr_spin.setValue(30.0)
        self.sub_first_chk = QtWidgets.QCheckBox(
            "Subtract first sample (baseline = 0)", self
        )

        form.addRow("Time column:", self.time_col_combo)
        form.addRow("Value column:", self.val_col_combo)
        form.addRow("Variable label:", self.var_edit)
        form.addRow("Trace color:", self.color_edit)
        form.addRow("Time offset (s):", self.offset_spin)
        form.addRow("Display window (s):", self.window_spin)
        form.addRow("Sampling rate (Hz, if no time col):", self.sr_spin)
        form.addRow("", self.sub_first_chk)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

    def build_spec(self) -> ExternalDataStreamSpec:
        t_col = str(self.time_col_combo.currentData() or "").strip()
        v_col = str(self.val_col_combo.currentData() or "1").strip()
        cols = f"{t_col},{v_col}" if t_col else v_col
        sr = float(self.sr_spin.value()) if not t_col else None
        lbl = self.var_edit.text().strip() or self._file_path.stem
        return ExternalDataStreamSpec(
            file_path=str(self._file_path),
            columns=cols,
            title=lbl,
            variable_name=lbl,
            color=self.color_edit.text().strip() or "#2563eb",
            time_offset_s=float(self.offset_spin.value()),
            time_interval_s=float(self.window_spin.value()),
            subtract_first_value=bool(self.sub_first_chk.isChecked()),
            sampling_rate_hz=sr,
        )


class ExternalDataPlotCanvas(QtWidgets.QWidget):
    """Fast QPainter waveform canvas with live playhead, bout shading, and click-to-seek."""

    timeScrubbed = QtCore.Signal(float)
    windowChanged = QtCore.Signal(float)

    _LEFT_PAD = 58.0
    _RIGHT_PAD = 14.0
    _TOP_PAD = 28.0
    _BOTTOM_PAD = 24.0

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(175)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self._streams: List[LoadedExternalStream] = []
        self._current_time_s: float = 0.0
        self._window_width_s: float = 60.0
        self._intervals_provider: Optional[Callable[[], Sequence[BehaviorInterval]]] = (
            None
        )
        self._scrubbing: bool = False

    def set_streams(self, streams: Sequence[LoadedExternalStream]) -> None:
        self._streams = list(streams)
        self.update()

    def set_intervals_provider(
        self, provider: Optional[Callable[[], Sequence[BehaviorInterval]]]
    ) -> None:
        self._intervals_provider = provider
        self.update()

    def set_current_time(self, time_s: float) -> None:
        self._current_time_s = max(0.0, float(time_s))
        self.update()

    def set_window_width(self, window_s: float) -> None:
        self._window_width_s = max(0.0, float(window_s))
        self.update()

    def current_time(self) -> float:
        return self._current_time_s

    def _time_range(self) -> Tuple[float, float]:
        if self._window_width_s <= 0 and self._streams:
            bounds = [s.time_bounds for s in self._streams if s.sample_count > 0]
            if bounds:
                t0 = min(b[0] for b in bounds)
                t1 = max(b[1] for b in bounds)
                return (t0, max(t0 + 1.0, t1))
        half = 0.5 * max(1.0, self._window_width_s)
        return (self._current_time_s - half, self._current_time_s + half)

    def _x_to_time(self, x_px: float) -> float:
        plot_w = max(1.0, float(self.width()) - self._LEFT_PAD - self._RIGHT_PAD)
        t0, t1 = self._time_range()
        frac = max(0.0, min(1.0, (float(x_px) - self._LEFT_PAD) / plot_w))
        return max(0.0, t0 + frac * (t1 - t0))

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._streams:
            self._scrubbing = True
            self.timeScrubbed.emit(self._x_to_time(event.pos().x()))
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._scrubbing and self._streams:
            self.timeScrubbed.emit(self._x_to_time(event.pos().x()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._scrubbing:
            self._scrubbing = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta != 0 and self._window_width_s > 0:
            factor = 0.8 if delta > 0 else 1.25
            new_w = max(2.0, min(36000.0, self._window_width_s * factor))
            self._window_width_s = new_w
            self.windowChanged.emit(new_w)
            self.update()
            event.accept()
            return
        super().wheelEvent(event)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        w = float(self.width())
        h = float(self.height())
        painter.fillRect(self.rect(), QtGui.QColor("#ffffff"))

        plot_rect = QtCore.QRectF(
            self._LEFT_PAD,
            self._TOP_PAD,
            max(40.0, w - self._LEFT_PAD - self._RIGHT_PAD),
            max(40.0, h - self._TOP_PAD - self._BOTTOM_PAD),
        )
        painter.fillRect(plot_rect, QtGui.QColor("#f8fafc"))
        painter.setPen(QtGui.QPen(QtGui.QColor("#cbd5e1"), 1.0))
        painter.drawRect(plot_rect)

        if not self._streams:
            painter.setPen(QtGui.QColor("#64748b"))
            painter.drawText(
                plot_rect,
                QtCore.Qt.AlignCenter,
                "No external data streams loaded.\nClick '+ Add Data Stream…' or import a .boris file with plot_data.",
            )
            painter.end()
            return

        t0, t1 = self._time_range()
        span = max(1e-6, t1 - t0)
        max_pts = max(300, int(plot_rect.width() * 2.0))

        sliced: List[Tuple[LoadedExternalStream, np.ndarray, np.ndarray]] = []
        y_min = float("inf")
        y_max = float("-inf")
        center = 0.5 * (t0 + t1)
        for st in self._streams:
            ts, vs = st.window_slice(center, span, max_points=max_pts)
            mask = (ts >= t0) & (ts <= t1)
            ts = ts[mask]
            vs = vs[mask]
            if vs.size > 0:
                y_min = min(y_min, float(np.min(vs)))
                y_max = max(y_max, float(np.max(vs)))
            sliced.append((st, ts, vs))

        if not math.isfinite(y_min) or not math.isfinite(y_max):
            y_min, y_max = -1.0, 1.0
        elif abs(y_max - y_min) <= 1e-9:
            y_min -= 1.0
            y_max += 1.0
        else:
            pad = 0.08 * (y_max - y_min)
            y_min -= pad
            y_max += pad

        def _x_of(tv: float) -> float:
            return plot_rect.left() + ((tv - t0) / span) * plot_rect.width()

        def _y_of(vv: float) -> float:
            return (
                plot_rect.top()
                + (1.0 - (vv - y_min) / (y_max - y_min)) * plot_rect.height()
            )

        # Draw active behavioral bouts behind traces
        if self._intervals_provider is not None:
            try:
                intervals = self._intervals_provider() or ()
            except Exception:
                intervals = ()
            bout_brush = QtGui.QColor(253, 230, 138, 90)
            for iv in intervals:
                if iv.end <= t0 or iv.start >= t1 or iv.end <= iv.start:
                    continue
                bx0 = _x_of(max(t0, iv.start))
                bx1 = _x_of(min(t1, iv.end))
                painter.fillRect(
                    QtCore.QRectF(
                        bx0,
                        plot_rect.top(),
                        max(1.5, bx1 - bx0),
                        plot_rect.height(),
                    ),
                    bout_brush,
                )

        # Horizontal grid lines & Y ticks
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        grid_pen = QtGui.QPen(QtGui.QColor("#e2e8f0"), 1.0, QtCore.Qt.DotLine)
        text_pen = QtGui.QPen(QtGui.QColor("#475569"))
        for frac in (0.0, 0.5, 1.0):
            gy = plot_rect.top() + (1.0 - frac) * plot_rect.height()
            painter.setPen(grid_pen)
            painter.drawLine(
                QtCore.QPointF(plot_rect.left(), gy),
                QtCore.QPointF(plot_rect.right(), gy),
            )
            painter.setPen(text_pen)
            val = y_min + frac * (y_max - y_min)
            painter.drawText(
                QtCore.QRectF(2.0, gy - 8.0, self._LEFT_PAD - 6.0, 16.0),
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                f"{val:.2f}",
            )

        # Clip waveforms to plot rect
        painter.save()
        painter.setClipRect(plot_rect)
        for st, ts, vs in sliced:
            qcol = QtGui.QColor(st.spec.color or "#2563eb")
            if not qcol.isValid():
                qcol = QtGui.QColor("#2563eb")
            painter.setPen(QtGui.QPen(qcol, 1.6))
            if ts.size >= 2:
                path = QtGui.QPainterPath()
                path.moveTo(_x_of(float(ts[0])), _y_of(float(vs[0])))
                for idx in range(1, int(ts.size)):
                    path.lineTo(_x_of(float(ts[idx])), _y_of(float(vs[idx])))
                painter.drawPath(path)
            elif ts.size == 1:
                painter.drawEllipse(
                    QtCore.QPointF(_x_of(float(ts[0])), _y_of(float(vs[0]))), 2.5, 2.5
                )

        # Playhead vertical cursor
        if t0 <= self._current_time_s <= t1:
            px = _x_of(self._current_time_s)
            play_pen = QtGui.QPen(QtGui.QColor("#dc2626"), 1.6, QtCore.Qt.DashLine)
            painter.setPen(play_pen)
            painter.drawLine(
                QtCore.QPointF(px, plot_rect.top()),
                QtCore.QPointF(px, plot_rect.bottom()),
            )
        painter.restore()

        # Live readout header bar
        hud_x = plot_rect.left()
        bold_font = QtGui.QFont(font)
        bold_font.setBold(True)
        painter.setFont(bold_font)
        for st in self._streams:
            qcol = QtGui.QColor(st.spec.color or "#2563eb")
            if not qcol.isValid():
                qcol = QtGui.QColor("#2563eb")
            cur_v = st.value_at_time(self._current_time_s)
            v_txt = f"{cur_v:.3f}" if cur_v is not None else "—"
            badge = f"● {st.label}: {v_txt}   "
            painter.setPen(qcol)
            painter.drawText(
                QtCore.QRectF(hud_x, 4.0, max(60.0, w - hud_x - 8.0), 20.0),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                badge,
            )
            hud_x += painter.fontMetrics().horizontalAdvance(badge) + 4.0
            if hud_x > w - 80.0:
                break

        # Time axis ticks
        painter.setFont(font)
        painter.setPen(text_pen)
        for ti in range(5):
            frac = ti / 4.0
            tx = plot_rect.left() + frac * plot_rect.width()
            tv = t0 + frac * span
            painter.drawText(
                QtCore.QRectF(tx - 34.0, plot_rect.bottom() + 3.0, 68.0, 18.0),
                QtCore.Qt.AlignCenter,
                f"{tv:.1f}s",
            )
        painter.end()


class BorisSyncWidget(QtWidgets.QWidget):
    """Unified BORIS Multi-Camera Video Grid & Synchronized External Data Plotter."""

    timeJumpRequested = QtCore.Signal(float)
    audioMixChanged = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._tiles: List[SecondaryVideoTile] = []
        self._streams: List[LoadedExternalStream] = []
        self._current_time_s: float = 0.0
        self._intervals_provider: Optional[Callable[[], Sequence[BehaviorInterval]]] = (
            None
        )

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)

        # 1. Secondary Synchronized Cameras Group
        cam_group = QtWidgets.QGroupBox("Synchronized Secondary Cameras", splitter)
        cam_lay = QtWidgets.QVBoxLayout(cam_group)
        cam_lay.setContentsMargins(6, 6, 6, 6)
        cam_lay.setSpacing(5)

        cam_bar = QtWidgets.QHBoxLayout()
        self.add_camera_btn = QtWidgets.QPushButton("+ Add Camera Video…", cam_group)
        self.add_camera_btn.clicked.connect(self._browse_and_add_camera)
        self.clear_cameras_btn = QtWidgets.QPushButton("Clear Cameras", cam_group)
        self.clear_cameras_btn.clicked.connect(self.clear_secondary_videos)

        self.primary_audio_chk = QtWidgets.QCheckBox("Primary Audio", cam_group)
        self.primary_audio_chk.setChecked(True)
        self.primary_audio_chk.setToolTip(
            "Include primary video audio in synchronized playback mix"
        )
        self.primary_audio_chk.toggled.connect(lambda _v: self.audioMixChanged.emit())
        self.primary_gain_spin = QtWidgets.QSpinBox(cam_group)
        self.primary_gain_spin.setRange(0, 200)
        self.primary_gain_spin.setSuffix("%")
        self.primary_gain_spin.setValue(100)
        self.primary_gain_spin.setToolTip("Primary video audio volume (0–200%)")
        self.primary_gain_spin.valueChanged.connect(
            lambda _v: self.audioMixChanged.emit()
        )

        cam_bar.addWidget(self.add_camera_btn)
        cam_bar.addWidget(self.clear_cameras_btn)
        cam_bar.addSpacing(8)
        cam_bar.addWidget(self.primary_audio_chk)
        cam_bar.addWidget(self.primary_gain_spin)
        cam_bar.addStretch(1)
        cam_lay.addLayout(cam_bar)

        self._cam_scroll = QtWidgets.QScrollArea(cam_group)
        self._cam_scroll.setWidgetResizable(True)
        self._cam_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._cam_container = QtWidgets.QWidget(self._cam_scroll)
        self._cam_grid = QtWidgets.QGridLayout(self._cam_container)
        self._cam_grid.setContentsMargins(0, 0, 0, 0)
        self._cam_grid.setSpacing(6)
        self._cam_empty_lbl = QtWidgets.QLabel(
            "No secondary cameras attached. Add side/bottom/thermal camera videos to scrub in lockstep.",
            self._cam_container,
        )
        self._cam_empty_lbl.setWordWrap(True)
        self._cam_empty_lbl.setStyleSheet("color: #64748b; padding: 8px;")
        self._cam_grid.addWidget(self._cam_empty_lbl, 0, 0)
        self._cam_scroll.setWidget(self._cam_container)
        cam_lay.addWidget(self._cam_scroll, 1)

        # 2. Synchronized External Physiological / Hardware Data Streams Group
        ext_group = QtWidgets.QGroupBox(
            "External Physiological / Hardware Data Streams", splitter
        )
        ext_lay = QtWidgets.QVBoxLayout(ext_group)
        ext_lay.setContentsMargins(6, 6, 6, 6)
        ext_lay.setSpacing(5)

        ext_bar = QtWidgets.QHBoxLayout()
        self.add_stream_btn = QtWidgets.QPushButton("+ Add Data Stream…", ext_group)
        self.add_stream_btn.clicked.connect(self._browse_and_add_stream)
        self.window_spin = QtWidgets.QDoubleSpinBox(ext_group)
        self.window_spin.setRange(0.0, 36000.0)
        self.window_spin.setDecimals(1)
        self.window_spin.setSpecialValueText("Full Session")
        self.window_spin.setValue(60.0)
        self.window_spin.valueChanged.connect(self._on_window_spin_changed)
        self.export_svg_btn = QtWidgets.QPushButton("Export SVG…", ext_group)
        self.export_svg_btn.setEnabled(False)
        self.export_svg_btn.clicked.connect(self._export_streams_svg)
        self.clear_streams_btn = QtWidgets.QPushButton("Clear", ext_group)
        self.clear_streams_btn.clicked.connect(self.clear_external_streams)

        ext_bar.addWidget(self.add_stream_btn)
        ext_bar.addWidget(QtWidgets.QLabel("Window (s):", ext_group))
        ext_bar.addWidget(self.window_spin)
        ext_bar.addWidget(self.export_svg_btn)
        ext_bar.addWidget(self.clear_streams_btn)
        ext_bar.addStretch(1)
        ext_lay.addLayout(ext_bar)

        self.plot_canvas = ExternalDataPlotCanvas(ext_group)
        self.plot_canvas.timeScrubbed.connect(self.timeJumpRequested.emit)
        self.plot_canvas.windowChanged.connect(self._sync_window_spin_from_canvas)
        ext_lay.addWidget(self.plot_canvas, 1)

        self._stream_list_layout = QtWidgets.QVBoxLayout()
        self._stream_list_layout.setContentsMargins(0, 0, 0, 0)
        self._stream_list_layout.setSpacing(3)
        ext_lay.addLayout(self._stream_list_layout)

        splitter.addWidget(cam_group)
        splitter.addWidget(ext_group)
        root.addWidget(splitter, 1)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def secondary_tiles(self) -> List[SecondaryVideoTile]:
        return list(self._tiles)

    @property
    def external_streams(self) -> List[LoadedExternalStream]:
        return list(self._streams)

    def set_intervals_provider(
        self, provider: Optional[Callable[[], Sequence[BehaviorInterval]]]
    ) -> None:
        self._intervals_provider = provider
        self.plot_canvas.set_intervals_provider(provider)

    def sync_to_time(self, primary_time_s: float, *, immediate: bool = False) -> None:
        self._current_time_s = max(0.0, float(primary_time_s))
        for tile in self._tiles:
            tile.sync_to_time(self._current_time_s, immediate=immediate)
        self.plot_canvas.set_current_time(self._current_time_s)

    def add_secondary_video(
        self,
        video_path: Path | str,
        *,
        player_id: Optional[str] = None,
        offset_s: float = 0.0,
        audio_buffer: Optional[AudioBuffer] = None,
    ) -> SecondaryVideoTile:
        pid = player_id or str(len(self._tiles) + 2)
        tile = SecondaryVideoTile(
            video_path,
            player_id=pid,
            offset_s=offset_s,
            audio_buffer=audio_buffer,
            parent=self._cam_container,
        )
        tile.removeRequested.connect(self.remove_secondary_video)
        tile.audioConfigChanged.connect(lambda _t: self.audioMixChanged.emit())
        tile.sync_to_time(self._current_time_s, immediate=True)
        self._tiles.append(tile)
        self._relayout_camera_tiles()
        return tile

    def custom_audio_mix_active(self) -> bool:
        """Return True when secondary camera audio is unmuted or primary gain/mute is customized."""
        if not self.primary_audio_chk.isChecked():
            return True
        if int(self.primary_gain_spin.value()) != 100:
            return True
        return any(not t.is_audio_muted and t.audio_gain > 0.0 for t in self._tiles)

    def build_mixed_audio(
        self,
        primary_buffer: Optional[AudioBuffer],
        primary_start_sec: float,
        *,
        playback_speed: float = 1.0,
        duration_sec: Optional[float] = None,
        target_sample_rate: int = 48_000,
    ) -> Tuple[np.ndarray, int]:
        """Mix primary audio and unmuted secondary camera audio tracks in exact phase."""
        specs: List[AudioTrackMixSpec] = []
        if primary_buffer is not None:
            specs.append(
                AudioTrackMixSpec(
                    buffer=primary_buffer,
                    offset_s=0.0,
                    gain=float(self.primary_gain_spin.value()) / 100.0,
                    muted=not bool(self.primary_audio_chk.isChecked()),
                )
            )
        for tile in self._tiles:
            if tile.is_audio_muted or tile.audio_gain <= 0.0:
                continue
            buf = tile.ensure_audio_buffer()
            if buf is not None:
                specs.append(
                    AudioTrackMixSpec(
                        buffer=buf,
                        offset_s=tile.offset_s,
                        gain=tile.audio_gain,
                        muted=False,
                    )
                )
        return mix_synchronized_audio_buffers(
            specs,
            primary_start_sec=float(primary_start_sec),
            duration_sec=duration_sec,
            target_sample_rate=target_sample_rate,
            playback_speed=playback_speed,
        )

    def remove_secondary_video(self, tile: SecondaryVideoTile) -> None:
        if tile in self._tiles:
            self._tiles.remove(tile)
        tile.close_reader()
        tile.setParent(None)
        tile.deleteLater()
        self._relayout_camera_tiles()

    def clear_secondary_videos(self) -> None:
        for tile in list(self._tiles):
            tile.close_reader()
            tile.setParent(None)
            tile.deleteLater()
        self._tiles.clear()
        self._relayout_camera_tiles()

    def add_external_stream(
        self,
        spec_or_stream: ExternalDataStreamSpec | LoadedExternalStream | Path | str,
        *,
        base_dir: Optional[Path | str] = None,
    ) -> LoadedExternalStream:
        if isinstance(spec_or_stream, LoadedExternalStream):
            loaded = spec_or_stream
        elif isinstance(spec_or_stream, ExternalDataStreamSpec):
            loaded = load_external_data_stream(spec_or_stream, base_dir=base_dir)
        else:
            default_col = _STREAM_PALETTE[len(self._streams) % len(_STREAM_PALETTE)]
            loaded = load_external_data_stream(
                ExternalDataStreamSpec(
                    file_path=str(spec_or_stream), color=default_col
                ),
                base_dir=base_dir,
            )
        self._streams.append(loaded)
        if len(self._streams) == 1 and loaded.spec.time_interval_s > 0:
            self.window_spin.setValue(float(loaded.spec.time_interval_s))
        self._rebuild_stream_controls()
        return loaded

    def clear_external_streams(self) -> None:
        self._streams.clear()
        self._rebuild_stream_controls()

    def export_boris_media_and_plots(
        self, primary_video: Optional[str] = None
    ) -> Tuple[
        Dict[str, Tuple[str, ...]], Dict[str, float], Tuple[ExternalDataStreamSpec, ...]
    ]:
        """Return ``(media_by_player, media_offsets, external_data)`` for ``.boris`` export."""
        by_player: Dict[str, Tuple[str, ...]] = {}
        offsets: Dict[str, float] = {}
        if primary_video:
            by_player["1"] = (str(primary_video),)
            offsets["1"] = 0.0
        for idx, tile in enumerate(self._tiles, start=2):
            pk = tile.player_id if tile.player_id != "1" else str(idx)
            by_player[pk] = (str(tile.video_path),)
            offsets[pk] = float(tile.offset_s)
        ext_specs = tuple(st.spec for st in self._streams)
        return by_player, offsets, ext_specs

    def shutdown(self) -> None:
        self.clear_secondary_videos()

    # ------------------------------------------------------------------ #
    # Internal UI Helpers
    # ------------------------------------------------------------------ #
    def _relayout_camera_tiles(self) -> None:
        while self._cam_grid.count():
            item = self._cam_grid.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._cam_empty_lbl:
                w.setParent(self._cam_container)
        if not self._tiles:
            self._cam_empty_lbl.setVisible(True)
            self._cam_grid.addWidget(self._cam_empty_lbl, 0, 0)
            return
        self._cam_empty_lbl.setVisible(False)
        cols = 2 if len(self._tiles) >= 2 else 1
        for idx, tile in enumerate(self._tiles):
            self._cam_grid.addWidget(tile, idx // cols, idx % cols)

    def _rebuild_stream_controls(self) -> None:
        while self._stream_list_layout.count():
            item = self._stream_list_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        for idx, st in enumerate(self._streams):
            row_frame = QtWidgets.QFrame(self)
            rlay = QtWidgets.QHBoxLayout(row_frame)
            rlay.setContentsMargins(2, 1, 2, 1)
            rlay.setSpacing(6)
            lbl = QtWidgets.QLabel(
                f"● {st.label} ({st.sample_count} samples)", row_frame
            )
            lbl.setStyleSheet(f"color: {st.spec.color or '#2563eb'}; font-weight: 600;")
            rlay.addWidget(lbl, 1)
            rlay.addWidget(QtWidgets.QLabel("Offset (s):", row_frame))
            off_spin = QtWidgets.QDoubleSpinBox(row_frame)
            off_spin.setRange(-36000.0, 36000.0)
            off_spin.setDecimals(3)
            off_spin.setSingleStep(0.05)
            off_spin.setValue(float(st.spec.time_offset_s))
            off_spin.valueChanged.connect(
                lambda val, i=idx: self._update_stream_offset(i, float(val))
            )
            rlay.addWidget(off_spin)
            rm_btn = QtWidgets.QToolButton(row_frame)
            rm_btn.setText("✕")
            rm_btn.clicked.connect(lambda _c=False, i=idx: self._remove_stream_at(i))
            rlay.addWidget(rm_btn)
            self._stream_list_layout.addWidget(row_frame)

        self.export_svg_btn.setEnabled(bool(self._streams))
        self.plot_canvas.set_streams(self._streams)

    def _update_stream_offset(self, index: int, new_offset_s: float) -> None:
        if 0 <= index < len(self._streams):
            self._streams[index] = self._streams[index].with_offset(new_offset_s)
            self.plot_canvas.set_streams(self._streams)

    def _remove_stream_at(self, index: int) -> None:
        if 0 <= index < len(self._streams):
            self._streams.pop(index)
            self._rebuild_stream_controls()

    def _on_window_spin_changed(self, val: float) -> None:
        self.plot_canvas.set_window_width(float(val))

    def _sync_window_spin_from_canvas(self, val: float) -> None:
        with QtCore.QSignalBlocker(self.window_spin):
            self.window_spin.setValue(float(val))

    def _browse_and_add_camera(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Secondary Synchronized Video",
            str(Path.cwd()),
            "Videos (*.mp4 *.avi *.mov *.mkv *.tif *.tiff);;All Files (*)",
        )
        if not chosen:
            return
        try:
            self.add_secondary_video(Path(chosen))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Secondary Camera Error", f"Failed to open video:\n{exc}"
            )

    def _browse_and_add_stream(self) -> None:
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select External Physiological / Hardware CSV or TSV",
            str(Path.cwd()),
            "Time-Series Tables (*.csv *.tsv *.txt);;All Files (*)",
        )
        if not chosen:
            return
        try:
            default_col = _STREAM_PALETTE[len(self._streams) % len(_STREAM_PALETTE)]
            dlg = ExternalStreamConfigDialog(
                Path(chosen), default_color=default_col, parent=self
            )
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                return
            self.add_external_stream(dlg.build_spec())
        except (OSError, ExternalDataError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(
                self, "External Data Stream Error", f"Failed to load stream:\n{exc}"
            )

    def _export_streams_svg(self) -> None:
        if not self._streams:
            return
        chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Synchronized External Data SVG",
            "external_streams.svg",
            "SVG Vector Graphics (*.svg)",
        )
        if not chosen:
            return
        intervals: Iterable[BehaviorInterval] = ()
        if self._intervals_provider is not None:
            try:
                intervals = self._intervals_provider() or ()
            except Exception:
                intervals = ()
        write_external_data_svg(
            self._streams,
            Path(chosen),
            intervals=intervals,
            playhead_time=self._current_time_s,
        )
