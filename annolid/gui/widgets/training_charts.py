"""Native pure-Qt interactive training curves and artifact gallery widgets.

Renders live and historical training metrics (loss curves, validation metrics,
learning rates, multi-run overlays, EMA smoothing, log-Y scale, hover crosshair
readout) without requiring matplotlib, QtWebEngine, or a running TensorBoard
server.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from qtpy import QtCore, QtGui, QtWidgets

from annolid.services.training_monitor import (
    DiscoveredRun,
    MetricSeries,
    compute_hparam_diff,
    smooth_ema,
)
from annolid.services.training_telemetry import (
    TelemetrySample,
    format_telemetry_summary,
)


_SERIES_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (37, 99, 235),  # blue
    (234, 88, 12),  # orange
    (22, 163, 74),  # green
    (147, 51, 234),  # purple
    (13, 148, 136),  # teal
    (220, 38, 38),  # red
    (202, 138, 4),  # amber
    (219, 39, 119),  # pink
)


def series_color(index: int) -> QtGui.QColor:
    r, g, b = _SERIES_PALETTE[int(index) % len(_SERIES_PALETTE)]
    return QtGui.QColor(r, g, b)


@dataclass(frozen=True)
class ChartSeries:
    """Single plotted curve with raw steps & values."""

    label: str
    steps: Tuple[int, ...]
    values: Tuple[float, ...]
    color: QtGui.QColor
    dashed: bool = False


def _format_axis_value(val: float) -> str:
    av = abs(val)
    if av == 0:
        return "0"
    if av < 1e-3 or av >= 1e5:
        return f"{val:.1e}"
    if av < 1.0:
        return f"{val:.3f}"
    if av < 100.0:
        return f"{val:.2f}"
    return f"{val:.1f}"


def _format_tooltip_value(val: float) -> str:
    av = abs(val)
    if av == 0:
        return "0"
    if av < 1e-4 or av >= 1e6:
        return f"{val:.3e}"
    if av < 10.0:
        return f"{val:.4f}"
    return f"{val:.2f}"


class TrainingCurvePlotWidget(QtWidgets.QWidget):
    """Interactive multi-series line chart with EMA smoothing and hover crosshair."""

    def __init__(
        self,
        *,
        title: str = "",
        empty_text: str = "No metrics yet.",
        fixed_unit_range: bool = False,
        fixed_y_range: Optional[Tuple[float, float]] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title = str(title)
        self._empty_text = str(empty_text)
        self._fixed_unit_range = bool(fixed_unit_range)
        self._fixed_y_range = fixed_y_range
        self._step_unit = "epoch"
        self._smoothing: float = 0.0
        self._log_scale: bool = False
        self._show_markers: bool = False
        self._series: List[ChartSeries] = []
        self._hover_pos: Optional[QtCore.QPoint] = None

        self.setMouseTracking(True)
        self.setMinimumHeight(210)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

    def set_chart_data(
        self,
        *,
        title: Optional[str] = None,
        series: Sequence[ChartSeries],
        empty_text: Optional[str] = None,
        step_unit: str = "epoch",
        fixed_unit_range: Optional[bool] = None,
        fixed_y_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        if title is not None:
            self._title = str(title)
        if empty_text is not None:
            self._empty_text = str(empty_text)
        if fixed_unit_range is not None:
            self._fixed_unit_range = bool(fixed_unit_range)
        if fixed_y_range is not None:
            self._fixed_y_range = fixed_y_range
        self._step_unit = str(step_unit or "epoch")
        self._series = list(series)
        self.update()

    def set_options(
        self,
        *,
        smoothing: Optional[float] = None,
        log_scale: Optional[bool] = None,
        show_markers: Optional[bool] = None,
    ) -> None:
        if smoothing is not None:
            self._smoothing = max(0.0, min(0.95, float(smoothing)))
        if log_scale is not None:
            self._log_scale = bool(log_scale)
        if show_markers is not None:
            self._show_markers = bool(show_markers)
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self._hover_pos = event.pos()
        self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self._hover_pos = None
        self.update()
        super().leaveEvent(event)

    def _transform_y(self, y: float) -> Optional[float]:
        if not math.isfinite(y):
            return None
        if self._log_scale:
            if y <= 0.0:
                return None
            return math.log10(y)
        return float(y)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        panel = self.rect().adjusted(2, 2, -2, -2)
        pal = self.palette()
        bg = pal.color(QtGui.QPalette.Base)
        fg = pal.color(QtGui.QPalette.Text)
        muted = QtGui.QColor(fg)
        muted.setAlpha(150)
        border = QtGui.QColor(fg)
        border.setAlpha(45)

        painter.setPen(QtGui.QPen(border, 1))
        painter.setBrush(bg)
        painter.drawRoundedRect(panel, 8, 8)

        # Header title + compact latest readout
        header_rect = panel.adjusted(12, 6, -12, -(panel.height() - 28))
        header_font = painter.font()
        header_font.setBold(True)
        header_font.setPointSize(max(9, header_font.pointSize()))
        painter.setFont(header_font)
        painter.setPen(fg)
        painter.drawText(
            header_rect,
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            self._title,
        )

        # Prepare smoothed & transformed series
        prepared: List[
            Tuple[ChartSeries, Tuple[float, ...], List[Tuple[int, float, float]]]
        ] = []
        all_x: List[int] = []
        all_ty: List[float] = []
        for curve in self._series:
            if not curve.steps or not curve.values:
                continue
            sm = smooth_ema(curve.values, self._smoothing)
            pts: List[Tuple[int, float, float]] = []
            for x, raw_y, sm_y in zip(curve.steps, curve.values, sm):
                ty_sm = self._transform_y(sm_y)
                ty_raw = self._transform_y(raw_y)
                if ty_sm is None:
                    continue
                all_x.append(int(x))
                all_ty.append(ty_sm)
                if ty_raw is not None:
                    all_ty.append(ty_raw)
                pts.append((int(x), float(raw_y), float(sm_y)))
            if pts:
                prepared.append((curve, sm, pts))

        # Legend row under header (compact colored swatches + latest values)
        legend_font = painter.font()
        legend_font.setBold(False)
        legend_font.setPointSize(max(8, legend_font.pointSize() - 1))
        painter.setFont(legend_font)
        fm = QtGui.QFontMetrics(legend_font)

        legend_y = panel.top() + 30
        cur_x = panel.left() + 12
        max_legend_x = panel.right() - 12
        legend_rows = 1 if prepared else 0
        for curve, _sm, pts in prepared[:8]:
            latest_val = pts[-1][2]
            text = f"{curve.label}: {_format_tooltip_value(latest_val)}"
            item_w = 14 + fm.horizontalAdvance(text) + 12
            if cur_x + item_w > max_legend_x and cur_x > panel.left() + 12:
                cur_x = panel.left() + 12
                legend_y += 16
                legend_rows += 1
                if legend_rows > 2:
                    break
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(curve.color)
            painter.drawRoundedRect(QtCore.QRectF(cur_x, legend_y + 3, 9, 9), 2, 2)
            painter.setPen(muted)
            painter.drawText(
                QtCore.QRect(cur_x + 13, legend_y, item_w - 13, 15),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                text,
            )
            cur_x += item_w

        plot_top = panel.top() + 32 + legend_rows * 16 + 4
        plot = QtCore.QRect(
            panel.left() + 58,
            plot_top,
            max(10, panel.width() - 74),
            max(10, panel.bottom() - plot_top - 26),
        )
        if plot.width() < 24 or plot.height() < 24:
            painter.end()
            return

        if not prepared or not all_x or not all_ty:
            painter.setPen(muted)
            msg = (
                "No positive values available for log scale."
                if (self._log_scale and self._series)
                else self._empty_text
            )
            painter.drawText(plot, QtCore.Qt.AlignCenter, msg)
            painter.end()
            return

        min_x = min(all_x)
        max_x = max(all_x)
        if max_x == min_x:
            max_x = min_x + 1

        if self._fixed_y_range is not None and not self._log_scale:
            min_ty = float(self._fixed_y_range[0])
            max_ty = max(min_ty + 1e-6, float(self._fixed_y_range[1]))
        elif self._fixed_unit_range and not self._log_scale:
            min_ty = 0.0
            max_ty = max(1.0, max(all_ty) * 1.05)
        else:
            min_ty = min(all_ty)
            max_ty = max(all_ty)
            span = max_ty - min_ty
            if span <= 1e-12:
                pad = max(0.1, abs(max_ty) * 0.1)
                min_ty = min_ty - pad if self._log_scale else max(0.0, min_ty - pad)
                max_ty = max_ty + pad
            else:
                if self._log_scale:
                    min_ty -= span * 0.08
                    max_ty += span * 0.08
                else:
                    min_ty = (
                        max(0.0, min_ty - span * 0.08)
                        if min_ty >= 0
                        else min_ty - span * 0.08
                    )
                    max_ty += span * 0.08

        def map_pt(xv: int, tyv: float) -> QtCore.QPointF:
            nx = (xv - min_x) / float(max_x - min_x)
            ny = (tyv - min_ty) / float(max_ty - min_ty)
            px = plot.left() + nx * plot.width()
            py = plot.bottom() - max(0.0, min(1.0, ny)) * plot.height()
            return QtCore.QPointF(px, py)

        # Horizontal gridlines + Y ticks
        grid_pen = QtGui.QPen(border, 1, QtCore.Qt.DotLine)
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y_px = int(round(plot.bottom() - frac * plot.height()))
            painter.setPen(grid_pen)
            painter.drawLine(plot.left(), y_px, plot.right(), y_px)
            ty_val = min_ty + frac * (max_ty - min_ty)
            disp_val = (10.0**ty_val) if self._log_scale else ty_val
            painter.setPen(muted)
            painter.drawText(
                QtCore.QRect(panel.left() + 4, y_px - 8, 50, 16),
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                _format_axis_value(disp_val),
            )

        # Vertical gridlines + X ticks
        unit_prefix = (
            "It"
            if self._step_unit == "iteration"
            else (
                "St"
                if self._step_unit == "step"
                else ("#" if self._step_unit == "sample" else "Ep")
            )
        )
        for frac in (0.0, 0.5, 1.0):
            x_px = int(round(plot.left() + frac * plot.width()))
            if 0.0 < frac < 1.0:
                painter.setPen(grid_pen)
                painter.drawLine(x_px, plot.top(), x_px, plot.bottom())
            x_val = int(round(min_x + frac * (max(all_x) - min_x)))
            align = (
                QtCore.Qt.AlignLeft
                if frac == 0.0
                else (QtCore.Qt.AlignRight if frac == 1.0 else QtCore.Qt.AlignHCenter)
            )
            rect_x = (
                plot.left()
                if frac == 0.0
                else (plot.right() - 72 if frac == 1.0 else x_px - 36)
            )
            painter.setPen(muted)
            painter.drawText(
                QtCore.QRect(rect_x, plot.bottom() + 4, 72, 16),
                align | QtCore.Qt.AlignVCenter,
                f"{unit_prefix} {x_val}",
            )

        # Draw faint raw curves first when EMA smoothing is active
        if self._smoothing > 1e-4:
            for curve, _sm, pts in prepared:
                raw_poly: List[QtCore.QPointF] = []
                for x, raw_y, _sm_y in pts:
                    ty = self._transform_y(raw_y)
                    if ty is not None:
                        raw_poly.append(map_pt(x, ty))
                if len(raw_poly) >= 2:
                    faint = QtGui.QColor(curve.color)
                    faint.setAlpha(65)
                    painter.setPen(QtGui.QPen(faint, 1.25))
                    painter.drawPolyline(QtGui.QPolygonF(raw_poly))

        # Draw main smoothed curves
        for curve, _sm, pts in prepared:
            mapped: List[QtCore.QPointF] = []
            for x, _raw_y, sm_y in pts:
                ty = self._transform_y(sm_y)
                if ty is not None:
                    mapped.append(map_pt(x, ty))
            if not mapped:
                continue
            style = QtCore.Qt.DashLine if curve.dashed else QtCore.Qt.SolidLine
            painter.setPen(QtGui.QPen(curve.color, 2.0, style))
            if len(mapped) >= 2:
                painter.drawPolyline(QtGui.QPolygonF(mapped))
            if self._show_markers or len(mapped) <= 18:
                painter.setBrush(curve.color)
                for pt in mapped:
                    painter.drawEllipse(pt, 2.8, 2.8)

        # Interactive hover crosshair & readout box
        if self._hover_pos is not None and plot.contains(self._hover_pos):
            nx_hover = (self._hover_pos.x() - plot.left()) / float(max(1, plot.width()))
            target_x = min_x + nx_hover * (max_x - min_x)

            nearest_step: Optional[int] = None
            best_dist = float("inf")
            for _curve, _sm, pts in prepared:
                for x, _r, _s in pts:
                    d = abs(x - target_x)
                    if d < best_dist:
                        best_dist = d
                        nearest_step = x

            if nearest_step is not None:
                hx = int(round(map_pt(nearest_step, min_ty).x()))
                cross_pen = QtGui.QPen(muted, 1, QtCore.Qt.DashLine)
                painter.setPen(cross_pen)
                painter.drawLine(hx, plot.top(), hx, plot.bottom())

                tip_lines: List[Tuple[Optional[QtGui.QColor], str]] = [
                    (None, f"{self._step_unit.title()} {nearest_step}")
                ]
                for curve, _sm, pts in prepared:
                    hit = min(pts, key=lambda item: abs(item[0] - nearest_step))
                    if abs(hit[0] - nearest_step) <= max(
                        1, int((max_x - min_x) * 0.05)
                    ):
                        ty = self._transform_y(hit[2])
                        if ty is not None:
                            pt = map_pt(hit[0], ty)
                            painter.setPen(QtGui.QPen(bg, 1.5))
                            painter.setBrush(curve.color)
                            painter.drawEllipse(pt, 4.2, 4.2)
                        if self._smoothing > 1e-4:
                            txt = (
                                f"{curve.label}: {_format_tooltip_value(hit[2])} "
                                f"(raw {_format_tooltip_value(hit[1])})"
                            )
                        else:
                            txt = f"{curve.label}: {_format_tooltip_value(hit[1])}"
                        tip_lines.append((curve.color, txt))

                if len(tip_lines) > 1:
                    line_h = 16
                    box_w = max(fm.horizontalAdvance(t) for _c, t in tip_lines) + 28
                    box_h = len(tip_lines) * line_h + 10
                    tx = (
                        hx + 10
                        if (hx + 10 + box_w) <= plot.right()
                        else max(plot.left(), hx - box_w - 10)
                    )
                    ty_box = plot.top() + 6
                    tip_rect = QtCore.QRect(tx, ty_box, box_w, box_h)

                    tip_bg = QtGui.QColor(bg)
                    tip_bg.setAlpha(235)
                    painter.setPen(QtGui.QPen(border, 1))
                    painter.setBrush(tip_bg)
                    painter.drawRoundedRect(tip_rect, 6, 6)

                    for i, (col, text) in enumerate(tip_lines):
                        row_y = tip_rect.top() + 5 + i * line_h
                        if col is not None:
                            painter.setPen(QtCore.Qt.NoPen)
                            painter.setBrush(col)
                            painter.drawEllipse(
                                QtCore.QPointF(tip_rect.left() + 10, row_y + 8),
                                3.5,
                                3.5,
                            )
                            painter.setPen(fg)
                            painter.drawText(
                                QtCore.QRect(
                                    tip_rect.left() + 18, row_y, box_w - 22, line_h
                                ),
                                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                                text,
                            )
                        else:
                            bold_f = QtGui.QFont(legend_font)
                            bold_f.setBold(True)
                            painter.setFont(bold_f)
                            painter.setPen(fg)
                            painter.drawText(
                                QtCore.QRect(
                                    tip_rect.left() + 8, row_y, box_w - 14, line_h
                                ),
                                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                                text,
                            )
                            painter.setFont(legend_font)

        painter.end()


class TrainingMetricsCanvas(QtWidgets.QWidget):
    """Multi-mode training curves canvas supporting single-run & multi-run comparison."""

    _VIEW_MODES: Tuple[Tuple[str, str], ...] = (
        ("overview", "Overview (Loss + Primary Metric)"),
        ("losses", "Loss Breakdown"),
        ("validation", "Validation Metrics"),
        ("lr", "Learning Rate"),
        ("custom", "Single Metric / Compare Tag"),
    )

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._primary_run: Optional[DiscoveredRun] = None
        self._compared_runs: List[DiscoveredRun] = []
        self._available_tags: List[str] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QtWidgets.QLabel(self.tr("View:")))
        self.view_mode_combo = QtWidgets.QComboBox(self)
        for key, label in self._VIEW_MODES:
            self.view_mode_combo.addItem(label, key)
        self.view_mode_combo.currentIndexChanged.connect(self._on_controls_changed)
        controls.addWidget(self.view_mode_combo)

        self.metric_tag_label = QtWidgets.QLabel(self.tr("Metric:"))
        self.metric_tag_combo = QtWidgets.QComboBox(self)
        self.metric_tag_combo.setMinimumWidth(180)
        self.metric_tag_combo.currentIndexChanged.connect(self._render)
        controls.addWidget(self.metric_tag_label)
        controls.addWidget(self.metric_tag_combo)

        controls.addWidget(QtWidgets.QLabel(self.tr("Smoothing:")))
        self.smoothing_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
        self.smoothing_slider.setRange(0, 95)
        self.smoothing_slider.setValue(0)
        self.smoothing_slider.setMaximumWidth(110)
        self.smoothing_slider.valueChanged.connect(self._on_smoothing_changed)
        self.smoothing_value_label = QtWidgets.QLabel("0.00")
        self.smoothing_value_label.setMinimumWidth(34)
        controls.addWidget(self.smoothing_slider)
        controls.addWidget(self.smoothing_value_label)

        self.log_scale_check = QtWidgets.QCheckBox(self.tr("Log Y"))
        self.log_scale_check.toggled.connect(self._on_plot_options_changed)
        controls.addWidget(self.log_scale_check)

        self.markers_check = QtWidgets.QCheckBox(self.tr("Points"))
        self.markers_check.toggled.connect(self._on_plot_options_changed)
        controls.addWidget(self.markers_check)

        controls.addStretch(1)

        self.export_png_btn = QtWidgets.QPushButton(self.tr("Export PNG…"))
        self.export_png_btn.clicked.connect(self._export_png)
        controls.addWidget(self.export_png_btn)

        layout.addLayout(controls)

        self.compare_banner = QtWidgets.QLabel("")
        self.compare_banner.setWordWrap(True)
        self.compare_banner.setProperty("muted", True)
        self.compare_banner.setVisible(False)
        layout.addWidget(self.compare_banner)

        self._plots_container = QtWidgets.QWidget(self)
        self._plots_layout = QtWidgets.QHBoxLayout(self._plots_container)
        self._plots_layout.setContentsMargins(0, 0, 0, 0)
        self._plots_layout.setSpacing(10)

        self.left_plot = TrainingCurvePlotWidget(
            title="Training / Validation Loss",
            empty_text="Select a training run to view loss curves.",
            parent=self._plots_container,
        )
        self.right_plot = TrainingCurvePlotWidget(
            title="Validation Metrics",
            empty_text="Waiting for validation metrics…",
            fixed_unit_range=True,
            parent=self._plots_container,
        )
        self._plots_layout.addWidget(self.left_plot, 1)
        self._plots_layout.addWidget(self.right_plot, 1)
        layout.addWidget(self._plots_container, 1)

        self._sync_metric_picker_visibility()

    def set_runs(
        self,
        primary_run: Optional[DiscoveredRun],
        compared_runs: Sequence[DiscoveredRun] = (),
    ) -> None:
        self._primary_run = primary_run
        seen_keys: set[str] = set()
        ordered: List[DiscoveredRun] = []
        for run in list(compared_runs):
            k = str(run.run_dir)
            if k not in seen_keys:
                seen_keys.add(k)
                ordered.append(run)
        if primary_run is not None and str(primary_run.run_dir) not in seen_keys:
            ordered.insert(0, primary_run)
        self._compared_runs = ordered
        self._refresh_available_tags()
        self._render()

    def _effective_runs(self) -> List[DiscoveredRun]:
        if len(self._compared_runs) > 1:
            return list(self._compared_runs)
        if self._primary_run is not None:
            return [self._primary_run]
        return list(self._compared_runs)

    def _refresh_available_tags(self) -> None:
        tags: List[str] = []
        seen: set[str] = set()
        for run in self._effective_runs():
            for tag in sorted(run.metrics.keys()):
                if tag not in seen:
                    seen.add(tag)
                    tags.append(tag)
        if tags == self._available_tags:
            return
        prev = self.metric_tag_combo.currentText()
        self._available_tags = tags
        self.metric_tag_combo.blockSignals(True)
        self.metric_tag_combo.clear()
        for tag in tags:
            self.metric_tag_combo.addItem(tag)
        if prev and prev in tags:
            self.metric_tag_combo.setCurrentText(prev)
        self.metric_tag_combo.blockSignals(False)

    def _sync_metric_picker_visibility(self) -> None:
        mode = str(self.view_mode_combo.currentData() or "overview")
        is_custom = mode == "custom"
        self.metric_tag_label.setVisible(is_custom)
        self.metric_tag_combo.setVisible(is_custom)
        self.right_plot.setVisible(mode == "overview")

    def _on_controls_changed(self) -> None:
        self._sync_metric_picker_visibility()
        self._render()

    def _on_smoothing_changed(self, value: int) -> None:
        w = float(value) / 100.0
        self.smoothing_value_label.setText(f"{w:.2f}")
        self._on_plot_options_changed()

    def _on_plot_options_changed(self) -> None:
        w = float(self.smoothing_slider.value()) / 100.0
        log_y = bool(self.log_scale_check.isChecked())
        pts = bool(self.markers_check.isChecked())
        self.left_plot.set_options(smoothing=w, log_scale=log_y, show_markers=pts)
        self.right_plot.set_options(smoothing=w, log_scale=log_y, show_markers=pts)

    @staticmethod
    def _short_run_label(run: DiscoveredRun) -> str:
        if run.metadata.alias:
            return run.metadata.alias
        name = run.run_dir.name
        if run.model and run.model != name:
            return f"{run.model}/{name}"
        return name

    @staticmethod
    def _find_series(
        run: DiscoveredRun, candidates: Sequence[str]
    ) -> Optional[MetricSeries]:
        lowered = {k.lower(): s for k, s in run.metrics.items()}
        for cand in candidates:
            hit = lowered.get(cand.lower())
            if hit is not None:
                return hit
        return None

    def _render(self) -> None:
        runs = self._effective_runs()
        mode = str(self.view_mode_combo.currentData() or "overview")
        self._sync_metric_picker_visibility()

        if len(runs) > 1:
            names = ", ".join(self._short_run_label(r) for r in runs[:6])
            extra = f" (+{len(runs) - 6} more)" if len(runs) > 6 else ""
            self.compare_banner.setText(
                self.tr("Comparing %d runs: %s%s") % (len(runs), names, extra)
            )
            self.compare_banner.setVisible(True)
        else:
            self.compare_banner.setVisible(False)

        if not runs:
            self.left_plot.set_chart_data(
                title="Training / Validation Loss",
                series=[],
                empty_text="Select a training run to view loss curves.",
                fixed_unit_range=False,
            )
            self.right_plot.set_chart_data(
                title="Validation Metrics",
                series=[],
                empty_text="Select a training run to view validation metrics.",
                fixed_unit_range=True,
            )
            return

        step_unit = runs[0].step_unit

        if mode == "overview":
            loss_series: List[ChartSeries] = []
            metric_series: List[ChartSeries] = []
            comparing = len(runs) > 1

            for idx, run in enumerate(runs):
                rlabel = self._short_run_label(run)
                tr = self._find_series(
                    run,
                    (
                        "train_loss",
                        "loss/total",
                        "train_loss_total",
                        "total_loss",
                        "loss",
                        "train/box_loss",
                    ),
                )
                va = self._find_series(
                    run,
                    ("val_loss", "val/loss", "val_loss_total", "val/box_loss"),
                )
                if tr is not None:
                    loss_series.append(
                        ChartSeries(
                            label=f"{rlabel} (train)" if comparing else "Train Loss",
                            steps=tr.steps,
                            values=tr.values,
                            color=series_color(idx if comparing else 0),
                            dashed=False,
                        )
                    )
                if va is not None:
                    loss_series.append(
                        ChartSeries(
                            label=f"{rlabel} (val)" if comparing else "Val Loss",
                            steps=va.steps,
                            values=va.values,
                            color=series_color(idx if comparing else 1),
                            dashed=comparing,
                        )
                    )

                if comparing:
                    if (
                        run.primary_metric_name
                        and run.primary_metric_name in run.metrics
                    ):
                        ms = run.metrics[run.primary_metric_name]
                        metric_series.append(
                            ChartSeries(
                                label=f"{rlabel} ({ms.tag})",
                                steps=ms.steps,
                                values=ms.values,
                                color=series_color(idx),
                            )
                        )
                else:
                    val_candidates = (
                        ("metrics/mAP50(B)", "Box mAP50"),
                        ("metrics/mAP50(M)", "Mask mAP50"),
                        ("metrics/mAP50(P)", "Pose mAP50"),
                        ("metrics/mAP50-95(B)", "Box mAP50-95"),
                        ("metrics/mAP50-95(M)", "Mask mAP50-95"),
                        ("metrics/mAP50-95(P)", "Pose mAP50-95"),
                        ("metrics/pck@8px", "PCK@8px"),
                        ("metrics/pck@4px", "PCK@4px"),
                        ("metrics/pck@16px", "PCK@16px"),
                        ("metrics/bbox_AP", "BBox AP"),
                        ("metrics/segm_AP", "Segm AP"),
                        ("metrics/keypoint_AP", "Keypoint AP"),
                        ("val/accuracy", "Val Acc"),
                        ("val_acc", "Val Acc"),
                    )
                    c_idx = 2
                    for tag_key, nice in val_candidates:
                        ms = self._find_series(run, (tag_key,))
                        if ms is not None and len(metric_series) < 5:
                            metric_series.append(
                                ChartSeries(
                                    label=nice,
                                    steps=ms.steps,
                                    values=ms.values,
                                    color=series_color(c_idx),
                                )
                            )
                            c_idx += 1
                    if (
                        not metric_series
                        and run.primary_metric_name
                        and run.primary_metric_name in run.metrics
                    ):
                        ms = run.metrics[run.primary_metric_name]
                        metric_series.append(
                            ChartSeries(
                                label=ms.tag,
                                steps=ms.steps,
                                values=ms.values,
                                color=series_color(2),
                            )
                        )

            self.left_plot.set_chart_data(
                title="Training / Validation Loss",
                series=loss_series,
                empty_text="Waiting for loss metrics…",
                step_unit=step_unit,
                fixed_unit_range=False,
            )
            fixed_unit = bool(
                metric_series
                and all(0.0 <= v <= 1.05 for s in metric_series for v in s.values)
            )
            self.right_plot.set_chart_data(
                title="Validation / Accuracy Metrics",
                series=metric_series,
                empty_text="Waiting for validation metrics…",
                step_unit=step_unit,
                fixed_unit_range=fixed_unit,
            )
            return

        if mode == "losses":
            series: List[ChartSeries] = []
            comparing = len(runs) > 1
            color_idx = 0
            for r_idx, run in enumerate(runs):
                rlabel = self._short_run_label(run)
                for tag, ms in sorted(run.metrics.items()):
                    if "loss" not in tag.lower():
                        continue
                    series.append(
                        ChartSeries(
                            label=f"{rlabel}: {tag}" if comparing else tag,
                            steps=ms.steps,
                            values=ms.values,
                            color=series_color(r_idx if comparing else color_idx),
                            dashed=("val" in tag.lower()),
                        )
                    )
                    color_idx += 1
                    if len(series) >= 10:
                        break
            self.left_plot.set_chart_data(
                title="Loss Breakdown",
                series=series,
                empty_text="No loss series found for this run.",
                step_unit=step_unit,
                fixed_unit_range=False,
            )
            return

        if mode == "validation":
            series = []
            comparing = len(runs) > 1
            color_idx = 0
            for r_idx, run in enumerate(runs):
                rlabel = self._short_run_label(run)
                for tag, ms in sorted(run.metrics.items()):
                    tl = tag.lower()
                    if not (
                        tl.startswith("metrics/")
                        or any(
                            k in tl
                            for k in (
                                "map",
                                "pck",
                                "precision",
                                "recall",
                                "dice",
                                "acc",
                                "f1",
                                "_ap",
                            )
                        )
                    ):
                        continue
                    series.append(
                        ChartSeries(
                            label=f"{rlabel}: {tag}" if comparing else tag,
                            steps=ms.steps,
                            values=ms.values,
                            color=series_color(r_idx if comparing else color_idx),
                        )
                    )
                    color_idx += 1
                    if len(series) >= 10:
                        break
            fixed_unit = bool(
                series and all(0.0 <= v <= 1.05 for s in series for v in s.values)
            )
            self.left_plot.set_chart_data(
                title="Validation Metrics",
                series=series,
                empty_text="No validation metrics recorded yet.",
                step_unit=step_unit,
                fixed_unit_range=fixed_unit,
            )
            return

        if mode == "lr":
            series = []
            comparing = len(runs) > 1
            color_idx = 0
            for r_idx, run in enumerate(runs):
                rlabel = self._short_run_label(run)
                for tag, ms in sorted(run.metrics.items()):
                    tl = tag.lower()
                    if tl == "lr" or tl.startswith("lr/") or tl.endswith("/lr"):
                        series.append(
                            ChartSeries(
                                label=f"{rlabel}: {tag}" if comparing else tag,
                                steps=ms.steps,
                                values=ms.values,
                                color=series_color(r_idx if comparing else color_idx),
                            )
                        )
                        color_idx += 1
            self.left_plot.set_chart_data(
                title="Learning Rate Schedule",
                series=series,
                empty_text="No learning rate metrics found.",
                step_unit=step_unit,
                fixed_unit_range=False,
            )
            return

        # Custom tag comparison
        selected_tag = self.metric_tag_combo.currentText().strip()
        series = []
        if selected_tag:
            for r_idx, run in enumerate(runs):
                ms = run.metrics.get(selected_tag)
                if ms is not None:
                    series.append(
                        ChartSeries(
                            label=self._short_run_label(run)
                            if len(runs) > 1
                            else selected_tag,
                            steps=ms.steps,
                            values=ms.values,
                            color=series_color(r_idx),
                        )
                    )
        self.left_plot.set_chart_data(
            title=selected_tag or "Metric Comparison",
            series=series,
            empty_text="Select a metric tag to visualize.",
            step_unit=step_unit,
            fixed_unit_range=False,
        )

    def _export_png(self) -> None:
        default_name = "training_curves.png"
        if self._primary_run is not None:
            default_name = str(
                self._primary_run.run_dir / "training_curves_snapshot.png"
            )
        out_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Export Training Curves PNG"),
            default_name,
            "PNG Image (*.png)",
        )
        if not out_path:
            return
        pixmap = self._plots_container.grab()
        if not pixmap.save(out_path, "PNG"):
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Export Failed"),
                self.tr("Could not save curves image to:\n%s") % out_path,
            )


class _ScaledImagePreview(QtWidgets.QLabel):
    """Aspect-preserving image label that resizes cleanly without expanding loops."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._source_pixmap: Optional[QtGui.QPixmap] = None
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(220, 160)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

    def set_image_path(self, path: Optional[Path]) -> None:
        if path is None or not path.is_file():
            self._source_pixmap = None
            self.setPixmap(QtGui.QPixmap())
            self.setText(self.tr("No preview available."))
            return
        pm = QtGui.QPixmap(str(path))
        if pm.isNull():
            self._source_pixmap = None
            self.setPixmap(QtGui.QPixmap())
            self.setText(self.tr("Unable to load image:\n%s") % path.name)
            return
        self._source_pixmap = pm
        self.setText("")
        self._update_scaled()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_scaled()

    def _update_scaled(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        target = self.contentsRect().size()
        if target.width() <= 2 or target.height() <= 2:
            return
        scaled = self._source_pixmap.scaled(
            target,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)


class TrainingArtifactsGalleryWidget(QtWidgets.QWidget):
    """Gallery displaying training plots, confusion matrices, and batch previews."""

    open_tensorboard_images_requested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._run_dir: Optional[Path] = None
        self._paths: List[Path] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        bar = QtWidgets.QHBoxLayout()
        self.info_label = QtWidgets.QLabel(
            self.tr("Select a run to browse training plots and visual previews.")
        )
        self.info_label.setProperty("muted", True)
        self.open_img_btn = QtWidgets.QPushButton(self.tr("Open Image"))
        self.open_img_btn.setEnabled(False)
        self.open_img_btn.clicked.connect(self._open_selected_image)

        self.open_tb_images_btn = QtWidgets.QPushButton(
            self.tr("View TensorBoard Images")
        )
        self.open_tb_images_btn.setEnabled(False)
        self.open_tb_images_btn.clicked.connect(
            self.open_tensorboard_images_requested.emit
        )

        bar.addWidget(self.info_label, 1)
        bar.addWidget(self.open_img_btn)
        bar.addWidget(self.open_tb_images_btn)
        layout.addLayout(bar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        self.file_list = QtWidgets.QListWidget(splitter)
        self.file_list.setMinimumWidth(190)
        self.file_list.currentRowChanged.connect(self._on_row_changed)
        self.file_list.itemDoubleClicked.connect(
            lambda _item: self._open_selected_image()
        )

        self.preview = _ScaledImagePreview(splitter)
        splitter.addWidget(self.file_list)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)

    def set_run(self, run: Optional[DiscoveredRun]) -> None:
        if run is None:
            self._run_dir = None
            self._paths = []
            self.file_list.clear()
            self.preview.set_image_path(None)
            self.open_img_btn.setEnabled(False)
            self.open_tb_images_btn.setEnabled(False)
            self.info_label.setText(
                self.tr("Select a run to browse training plots and visual previews.")
            )
            return

        self._run_dir = run.run_dir
        new_paths = list(run.plot_images)
        self.open_tb_images_btn.setEnabled(run.tensorboard_dir is not None)

        if new_paths == self._paths:
            return

        prev_selected = self._selected_path()
        self._paths = new_paths
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for p in self._paths:
            try:
                rel = str(p.relative_to(run.run_dir))
            except Exception:
                rel = p.name
            item = QtWidgets.QListWidgetItem(rel)
            item.setData(QtCore.Qt.UserRole, str(p))
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)

        if not self._paths:
            self.preview.set_image_path(None)
            self.open_img_btn.setEnabled(False)
            if run.tensorboard_dir is not None:
                self.info_label.setText(
                    self.tr(
                        "No standalone PNG/JPG plot files found yet. "
                        "Use 'View TensorBoard Images' to inspect TensorBoard image summaries."
                    )
                )
            else:
                self.info_label.setText(
                    self.tr("No training plot images found in this run folder.")
                )
            return

        self.info_label.setText(
            self.tr("%d visual artifact(s) found in %s")
            % (len(self._paths), run.run_dir.name)
        )
        target_row = 0
        if prev_selected is not None and prev_selected in self._paths:
            target_row = self._paths.index(prev_selected)
        self.file_list.setCurrentRow(target_row)
        self._on_row_changed(target_row)

    def _selected_path(self) -> Optional[Path]:
        item = self.file_list.currentItem()
        if item is None:
            return None
        raw = item.data(QtCore.Qt.UserRole)
        return Path(str(raw)) if raw else None

    def _on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._paths):
            self.preview.set_image_path(None)
            self.open_img_btn.setEnabled(False)
            return
        path = self._paths[row]
        self.preview.set_image_path(path)
        self.open_img_btn.setEnabled(path.is_file())

    def _open_selected_image(self) -> None:
        path = self._selected_path()
        if path is not None and path.is_file():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))


def _fmt_opt_val(val: Optional[float], digits: int = 4) -> str:
    if val is None:
        return "—"
    if abs(val) > 0 and abs(val) < 1e-3:
        return f"{val:.2e}"
    return f"{val:.{digits}f}"


class SystemTelemetryWidget(QtWidgets.QWidget):
    """Live CPU, RAM, GPU, VRAM, and disk telemetry plot for training runs."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._samples: List[TelemetrySample] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        self.summary_label = QtWidgets.QLabel(
            self.tr("System: collecting hardware telemetry…"), self
        )
        font = self.summary_label.font()
        font.setBold(True)
        self.summary_label.setFont(font)
        self.detail_label = QtWidgets.QLabel("", self)
        self.detail_label.setProperty("muted", True)
        header.addWidget(self.summary_label, 1)
        header.addWidget(self.detail_label)
        layout.addLayout(header)

        self.plot = TrainingCurvePlotWidget(
            title="Host & GPU Utilization (%)",
            empty_text="Waiting for system telemetry samples…",
            fixed_y_range=(0.0, 100.0),
            parent=self,
        )
        layout.addWidget(self.plot, 1)

    def set_samples(self, samples: Sequence[TelemetrySample]) -> None:
        self._samples = list(samples)
        if not self._samples:
            self.summary_label.setText(self.tr("System: —"))
            self.detail_label.setText("")
            self.plot.set_chart_data(
                title="Host & GPU Utilization (%)",
                series=[],
                empty_text="Waiting for system telemetry samples…",
                step_unit="sample",
                fixed_y_range=(0.0, 100.0),
            )
            return

        latest = self._samples[-1]
        self.summary_label.setText(format_telemetry_summary(latest))
        detail_parts: List[str] = []
        if latest.gpu_name:
            detail_parts.append(latest.gpu_name)
        if latest.disk_free_gb is not None and latest.disk_total_gb is not None:
            detail_parts.append(
                f"Volume: {latest.disk_free_gb:.1f}/{latest.disk_total_gb:.0f} GB free"
            )
        self.detail_label.setText(" · ".join(detail_parts))

        cpu_steps: List[int] = []
        cpu_vals: List[float] = []
        ram_steps: List[int] = []
        ram_vals: List[float] = []
        gpu_steps: List[int] = []
        gpu_vals: List[float] = []
        vram_steps: List[int] = []
        vram_vals: List[float] = []

        for idx, sample in enumerate(self._samples, start=1):
            if sample.cpu_percent is not None:
                cpu_steps.append(idx)
                cpu_vals.append(float(sample.cpu_percent))
            if sample.ram_percent is not None:
                ram_steps.append(idx)
                ram_vals.append(float(sample.ram_percent))
            if sample.gpu_util_percent is not None:
                gpu_steps.append(idx)
                gpu_vals.append(float(sample.gpu_util_percent))
            vram_pct = sample.gpu_mem_percent
            if vram_pct is not None:
                vram_steps.append(idx)
                vram_vals.append(float(vram_pct))

        curves: List[ChartSeries] = []
        if cpu_steps:
            curves.append(
                ChartSeries("CPU %", tuple(cpu_steps), tuple(cpu_vals), series_color(0))
            )
        if ram_steps:
            curves.append(
                ChartSeries("RAM %", tuple(ram_steps), tuple(ram_vals), series_color(1))
            )
        if gpu_steps:
            curves.append(
                ChartSeries("GPU %", tuple(gpu_steps), tuple(gpu_vals), series_color(2))
            )
        if vram_steps:
            curves.append(
                ChartSeries(
                    "VRAM %",
                    tuple(vram_steps),
                    tuple(vram_vals),
                    series_color(3),
                    dashed=True,
                )
            )

        self.plot.set_chart_data(
            title="Host & GPU Utilization (%)",
            series=curves,
            empty_text="Hardware metrics unavailable on this system.",
            step_unit="sample",
            fixed_y_range=(0.0, 100.0),
        )


class RunComparisonMatrixWidget(QtWidgets.QWidget):
    """Multi-run best-metrics summary and hyperparameter diff table."""

    export_report_requested = QtCore.Signal()
    export_csv_requested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._runs: List[DiscoveredRun] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(8)
        self.summary_label = QtWidgets.QLabel(
            self.tr(
                "Select or check runs to compare best metrics and hyperparameters."
            ),
            self,
        )
        self.summary_label.setProperty("muted", True)
        self.only_diffs_check = QtWidgets.QCheckBox(
            self.tr("Show differences only"), self
        )
        self.only_diffs_check.setChecked(True)
        self.only_diffs_check.toggled.connect(self._rebuild_hparam_table)

        self.export_report_btn = QtWidgets.QPushButton(
            self.tr("Export HTML + SVG Report…"), self
        )
        self.export_report_btn.setEnabled(False)
        self.export_report_btn.clicked.connect(self.export_report_requested.emit)

        self.export_csv_btn = QtWidgets.QPushButton(self.tr("Export CSV…"), self)
        self.export_csv_btn.setEnabled(False)
        self.export_csv_btn.clicked.connect(self.export_csv_requested.emit)

        bar.addWidget(self.summary_label, 1)
        bar.addWidget(self.only_diffs_check)
        bar.addWidget(self.export_report_btn)
        bar.addWidget(self.export_csv_btn)
        layout.addLayout(bar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)

        self.metrics_table = QtWidgets.QTableWidget(0, 10, splitter)
        self.metrics_table.setHorizontalHeaderLabels(
            [
                "Run",
                "★",
                "Task / Model",
                "Status",
                "Step",
                "Train Loss (min)",
                "Val Loss (min)",
                "Best Metric",
                "Tags",
                "Health",
            ]
        )
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.metrics_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.metrics_table)

        self.hparam_table = QtWidgets.QTableWidget(0, 1, splitter)
        self.hparam_table.setHorizontalHeaderLabels(["Hyperparameter"])
        self.hparam_table.verticalHeader().setVisible(False)
        self.hparam_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.hparam_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.hparam_table.setAlternatingRowColors(True)
        self.hparam_table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.hparam_table)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, 1)

    @staticmethod
    def _run_title(run: DiscoveredRun) -> str:
        if run.metadata.alias:
            return run.metadata.alias
        name = run.run_dir.name
        if run.model and run.model != name:
            return f"{run.model}/{name}"
        return name

    def set_runs(
        self, runs: Sequence[DiscoveredRun], *, subtitle: Optional[str] = None
    ) -> None:
        self._runs = list(runs)
        has_runs = bool(self._runs)
        self.export_report_btn.setEnabled(has_runs)
        self.export_csv_btn.setEnabled(has_runs)

        if subtitle is not None:
            self.summary_label.setText(subtitle)
        elif not self._runs:
            self.summary_label.setText(
                self.tr(
                    "Select or check runs to compare best metrics and hyperparameters."
                )
            )
        elif len(self._runs) == 1:
            self.summary_label.setText(
                self.tr("Inspecting 1 run (check Cmp boxes to compare multiple runs)")
            )
        else:
            self.summary_label.setText(
                self.tr("Comparing %d runs · highlighted rows differ") % len(self._runs)
            )

        self.metrics_table.setRowCount(len(self._runs))
        for r_idx, run in enumerate(self._runs):
            step_txt = (
                f"{run.last_step}/{run.total_steps}"
                if run.last_step is not None and run.total_steps
                else (str(run.last_step) if run.last_step is not None else "—")
            )
            tr_txt = (
                f"{_fmt_opt_val(run.train_loss)} ({_fmt_opt_val(run.min_train_loss)})"
                if run.train_loss is not None
                else "—"
            )
            va_txt = (
                f"{_fmt_opt_val(run.val_loss)} ({_fmt_opt_val(run.min_val_loss)})"
                if run.val_loss is not None
                else "—"
            )
            if run.best_metric_value is not None:
                m_short = (run.primary_metric_name or "").split("/")[-1]
                best_txt = (
                    f"{m_short}={_fmt_opt_val(run.best_metric_value)} (@{run.best_metric_step})"
                    if m_short
                    else f"{_fmt_opt_val(run.best_metric_value)} (@{run.best_metric_step})"
                )
            else:
                best_txt = "—"
            tags_txt = ", ".join(run.metadata.tags) if run.metadata.tags else "—"
            if run.health_findings:
                health_txt = "; ".join(
                    f"[{f.severity.upper()}] {f.code}" for f in run.health_findings
                )
                health_tip = "\n".join(
                    f"[{f.severity.upper()}] {f.message}" for f in run.health_findings
                )
            else:
                health_txt = "Healthy"
                health_tip = "No anomalies detected."

            cells = [
                self._run_title(run),
                "★" if run.metadata.starred else "",
                f"{run.task} · {run.model}",
                run.status,
                step_txt,
                tr_txt,
                va_txt,
                best_txt,
                tags_txt,
                health_txt,
            ]
            for c_idx, val in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(val)
                if c_idx == 0:
                    item.setToolTip(str(run.run_dir))
                elif c_idx == 9:
                    item.setToolTip(health_tip)
                    if any(f.severity == "critical" for f in run.health_findings):
                        item.setForeground(QtGui.QBrush(QtGui.QColor(220, 38, 38)))
                    elif any(f.severity == "warning" for f in run.health_findings):
                        item.setForeground(QtGui.QBrush(QtGui.QColor(234, 88, 12)))
                    else:
                        item.setForeground(QtGui.QBrush(QtGui.QColor(22, 163, 74)))
                self.metrics_table.setItem(r_idx, c_idx, item)
        self.metrics_table.resizeColumnsToContents()
        self._rebuild_hparam_table()

    def _rebuild_hparam_table(self) -> None:
        if not self._runs:
            self.hparam_table.setRowCount(0)
            self.hparam_table.setColumnCount(1)
            self.hparam_table.setHorizontalHeaderLabels(["Hyperparameter"])
            return

        only_diffs = bool(self.only_diffs_check.isChecked() and len(self._runs) > 1)
        rows = compute_hparam_diff(self._runs, only_diffs=only_diffs)

        headers = ["Hyperparameter"] + [self._run_title(r) for r in self._runs]
        self.hparam_table.setColumnCount(len(headers))
        self.hparam_table.setHorizontalHeaderLabels(headers)
        self.hparam_table.setRowCount(len(rows))

        diff_bg = QtGui.QBrush(QtGui.QColor(37, 99, 235, 34))
        for r_idx, diff_row in enumerate(rows):
            key_item = QtWidgets.QTableWidgetItem(diff_row.key)
            if diff_row.differs:
                f = key_item.font()
                f.setBold(True)
                key_item.setFont(f)
                key_item.setBackground(diff_bg)
            self.hparam_table.setItem(r_idx, 0, key_item)
            for c_idx, val in enumerate(diff_row.values, start=1):
                item = QtWidgets.QTableWidgetItem(val)
                if diff_row.differs:
                    item.setBackground(diff_bg)
                self.hparam_table.setItem(r_idx, c_idx, item)
        self.hparam_table.resizeColumnsToContents()


__all__ = [
    "ChartSeries",
    "RunComparisonMatrixWidget",
    "SystemTelemetryWidget",
    "TrainingArtifactsGalleryWidget",
    "TrainingCurvePlotWidget",
    "TrainingMetricsCanvas",
    "series_color",
]
