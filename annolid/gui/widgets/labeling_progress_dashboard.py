from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence

from qtpy import QtCore, QtGui, QtWidgets

from annolid.datasets.label_index_stats import (
    build_stats_from_label_index,
    load_label_stats_snapshot,
)
from annolid.datasets.labelme_collection import default_label_index_path
from annolid.utils.annotation_store import load_labelme_json


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class GamificationStats:
    score: int
    level: int
    xp_into_level: int
    xp_to_next_level: int
    streak_days: int


@dataclass(frozen=True)
class Achievement:
    name: str
    description: str
    unlocked: bool


@dataclass(frozen=True)
class LabelingProjectStats:
    project_root: Path
    annotation_root: Path
    total_images: int
    total_annotation_files: int
    labeled_files: int
    total_shapes: int
    created_files: int
    edited_files: int
    coverage_percent: float
    top_labels: List[tuple[str, int]]
    annotator_file_counts: List[tuple[str, int]]
    annotator_shape_counts: List[tuple[str, int]]
    shape_type_counts: List[tuple[str, int]]
    activity_last_7_days: List[tuple[date, int]]
    gamification: GamificationStats
    achievements: List[Achievement]


def _level_floor(level: int) -> int:
    # Triangular progression: 0, 200, 600, 1200, ...
    return int(100 * level * (level - 1))


def _compute_level(score: int) -> tuple[int, int, int]:
    level = 1
    while score >= _level_floor(level + 1):
        level += 1
    floor = _level_floor(level)
    next_floor = _level_floor(level + 1)
    return level, max(0, score - floor), max(1, next_floor - floor)


def _compute_streak(activity_days: Sequence[date], *, today: date) -> int:
    days = set(activity_days)
    streak = 0
    cur = today
    while cur in days:
        streak += 1
        cur = cur - timedelta(days=1)
    return streak


def _build_achievements(
    *,
    labeled_files: int,
    total_shapes: int,
    streak_days: int,
    coverage_percent: float,
) -> List[Achievement]:
    return [
        Achievement(
            "First Label", "Create your first labeled frame", labeled_files >= 1
        ),
        Achievement("Shape Starter", "Annotate at least 50 shapes", total_shapes >= 50),
        Achievement("Centurion", "Annotate at least 100 shapes", total_shapes >= 100),
        Achievement("Consistency", "Maintain a 3-day streak", streak_days >= 3),
        Achievement("Week Warrior", "Maintain a 7-day streak", streak_days >= 7),
        Achievement(
            "Halfway There", "Reach 50% labeling coverage", coverage_percent >= 50.0
        ),
        Achievement(
            "Project Hero",
            "Reach 90% coverage with at least 500 shapes",
            coverage_percent >= 90.0 and total_shapes >= 500,
        ),
    ]


def _extract_annotator(payload: dict, shape: dict | None = None) -> str:
    candidates = []
    if isinstance(shape, dict):
        sflags = shape.get("flags")
        if isinstance(sflags, dict):
            candidates.extend(
                [
                    sflags.get("annotator"),
                    sflags.get("user"),
                    sflags.get("editor"),
                ]
            )
    flags = payload.get("flags")
    if isinstance(flags, dict):
        candidates.extend(
            [
                flags.get("annotator"),
                flags.get("user"),
                flags.get("editor"),
            ]
        )
    candidates.extend(
        [payload.get("annotator"), payload.get("user"), payload.get("editor")]
    )
    for raw in candidates:
        token = str(raw or "").strip()
        if token:
            return token
    return "unknown"


def analyze_labeling_project(
    *,
    project_root: Path,
    annotation_root: Path | None = None,
    now: datetime | None = None,
) -> LabelingProjectStats:
    project_root = Path(project_root).expanduser().resolve()
    annotation_root = (
        Path(annotation_root).expanduser().resolve()
        if annotation_root is not None
        else project_root
    )
    current_dt = now or datetime.now()
    today = current_dt.date()

    image_count = sum(
        1
        for p in project_root.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )

    total_annotation_files = 0
    labeled_files = 0
    total_shapes = 0
    created_files = 0
    edited_files = 0
    label_counter: Counter[str] = Counter()
    annotator_file_counter: Counter[str] = Counter()
    annotator_shape_counter: Counter[str] = Counter()
    shape_type_counter: Counter[str] = Counter()
    day_activity: Dict[date, int] = defaultdict(int)

    for json_path in annotation_root.rglob("*.json"):
        if not json_path.is_file():
            continue
        try:
            payload = load_labelme_json(json_path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        shapes = payload.get("shapes")
        if not isinstance(shapes, list):
            continue

        total_annotation_files += 1
        shape_count = 0
        file_annotator = _extract_annotator(payload)
        annotator_file_counter[file_annotator] += 1
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            shape_count += 1
            shape_annotator = _extract_annotator(payload, shape=shape)
            annotator_shape_counter[shape_annotator] += 1
            label = str(shape.get("label") or "unknown").strip() or "unknown"
            shape_type = str(shape.get("shape_type") or "unknown").strip() or "unknown"
            label_counter[label] += 1
            shape_type_counter[shape_type] += 1

        total_shapes += shape_count
        if shape_count > 0:
            labeled_files += 1

        try:
            st = json_path.stat()
            mdate = datetime.fromtimestamp(st.st_mtime).date()
            day_activity[mdate] += 1
            if abs(st.st_mtime - st.st_ctime) <= 1.0:
                created_files += 1
            else:
                edited_files += 1
        except OSError:
            continue

    if edited_files <= 0 and total_annotation_files > 0:
        # ctime semantics vary by platform; keep split stable.
        edited_files = max(0, total_annotation_files - created_files)
    if created_files + edited_files < total_annotation_files:
        created_files = total_annotation_files - edited_files

    coverage = (labeled_files / image_count * 100.0) if image_count > 0 else 0.0

    activity_dates = sorted(day_activity.keys())
    streak = _compute_streak(activity_dates, today=today)
    score = int(
        total_shapes * 2 + labeled_files * 8 + streak * 30 + len(activity_dates) * 12
    )
    level, xp_into, xp_span = _compute_level(score)
    achievements = _build_achievements(
        labeled_files=labeled_files,
        total_shapes=total_shapes,
        streak_days=streak,
        coverage_percent=coverage,
    )

    recent_days: List[tuple[date, int]] = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        recent_days.append((d, int(day_activity.get(d, 0))))

    return LabelingProjectStats(
        project_root=project_root,
        annotation_root=annotation_root,
        total_images=int(image_count),
        total_annotation_files=int(total_annotation_files),
        labeled_files=int(labeled_files),
        total_shapes=int(total_shapes),
        created_files=int(created_files),
        edited_files=int(edited_files),
        coverage_percent=float(coverage),
        top_labels=label_counter.most_common(8),
        annotator_file_counts=annotator_file_counter.most_common(8),
        annotator_shape_counts=annotator_shape_counter.most_common(8),
        shape_type_counts=shape_type_counter.most_common(8),
        activity_last_7_days=recent_days,
        gamification=GamificationStats(
            score=int(score),
            level=int(level),
            xp_into_level=int(xp_into),
            xp_to_next_level=max(1, int(xp_span - xp_into)),
            streak_days=int(streak),
        ),
        achievements=achievements,
    )


class _StatCard(QtWidgets.QFrame):
    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statCard")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)
        self.title_label = QtWidgets.QLabel(title, self)
        self.title_label.setObjectName("cardTitle")
        self.value_label = QtWidgets.QLabel("--", self)
        self.value_label.setObjectName("cardValue")
        self.sub_label = QtWidgets.QLabel("", self)
        self.sub_label.setObjectName("cardSub")
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.sub_label)


class _BarsWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._values: List[int] = []
        self._labels: List[str] = []
        self._bar_color = QtGui.QColor("#2f80ed")
        self.setMinimumHeight(90)

    def set_series(self, rows: Sequence[tuple[str, int]]) -> None:
        self._labels = [str(k) for k, _ in rows]
        self._values = [int(v) for _, v in rows]
        self.update()

    def paintEvent(
        self, event: QtGui.QPaintEvent
    ) -> None:  # pragma: no cover - UI paint
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect().adjusted(6, 8, -6, -18)
        if rect.width() <= 10 or rect.height() <= 10 or not self._values:
            return
        vmax = max(self._values) or 1
        n = len(self._values)
        gap = 6
        bar_w = max(4, int((rect.width() - gap * (n - 1)) / max(1, n)))
        for i, val in enumerate(self._values):
            h = int((val / vmax) * rect.height())
            x = rect.x() + i * (bar_w + gap)
            y = rect.bottom() - h
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(self._bar_color)
            painter.drawRoundedRect(QtCore.QRectF(x, y, bar_w, h), 3, 3)
            painter.setPen(QtGui.QColor("#334e68"))
            painter.drawText(
                QtCore.QRectF(x - 6, rect.bottom() + 2, bar_w + 12, 14),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop,
                str(i + 1),
            )


class _DashboardRefreshWorker(QtCore.QObject):
    finished = QtCore.Signal(int, object, str, str)

    def __init__(
        self,
        *,
        token: int,
        project_root: Path,
        annotation_root: Path,
        index_file: Path | None,
    ) -> None:
        super().__init__()
        self._token = int(token)
        self._project_root = Path(project_root).expanduser().resolve()
        self._annotation_root = Path(annotation_root).expanduser().resolve()
        self._index_file = (
            Path(index_file).expanduser().resolve() if index_file else None
        )

    @QtCore.Slot()
    def run(self) -> None:
        try:
            stats = None
            source = "filesystem_scan"
            if self._index_file is not None and self._index_file.exists():
                payload = load_label_stats_snapshot(self._index_file)
                if payload is None:
                    payload = build_stats_from_label_index(
                        index_file=self._index_file,
                        project_root=self._project_root,
                    )
                stats = _stats_from_index_payload(
                    payload,
                    project_root=self._project_root,
                    annotation_root=self._annotation_root,
                )
                source = "label_index"

            if stats is None:
                stats = analyze_labeling_project(
                    project_root=self._project_root,
                    annotation_root=self._annotation_root,
                )
                source = "filesystem_scan"

            self.finished.emit(self._token, stats, source, "")
        except Exception as exc:
            self.finished.emit(self._token, None, "", str(exc))


def _stats_from_index_payload(
    payload: Dict[str, object],
    *,
    project_root: Path,
    annotation_root: Path,
) -> LabelingProjectStats | None:
    if not isinstance(payload, dict):
        return None
    try:
        total_images = int(payload.get("total_images", 0))
        total_annotation_files = int(payload.get("total_annotation_files", 0))
        labeled_files = int(payload.get("labeled_files", 0))
        total_shapes = int(payload.get("total_shapes", 0))
        created_files = int(payload.get("created_files", 0))
        edited_files = int(payload.get("edited_files", 0))
        coverage_percent = float(payload.get("coverage_percent", 0.0))
    except Exception:
        return None

    def _pairs(rows: object) -> List[tuple[str, int]]:
        parsed: List[tuple[str, int]] = []
        if not isinstance(rows, list):
            return parsed
        for row in rows:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            parsed.append((str(row[0]), int(row[1])))
        return parsed

    top_labels = _pairs(payload.get("top_labels"))
    shape_type_counts = _pairs(payload.get("shape_type_counts"))
    annotator_file_counts = _pairs(payload.get("annotator_file_counts"))
    annotator_shape_counts = _pairs(payload.get("annotator_shape_counts"))

    activity_rows: List[tuple[date, int]] = []
    raw_activity = payload.get("activity_last_7_days")
    if isinstance(raw_activity, list):
        for row in raw_activity:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            dval = str(row[0]).strip()
            try:
                d = datetime.fromisoformat(dval).date()
            except Exception:
                continue
            activity_rows.append((d, int(row[1])))
    if len(activity_rows) != 7:
        today = datetime.now().date()
        activity_rows = [(today - timedelta(days=i), 0) for i in range(6, -1, -1)]

    active_days = [d for d, c in activity_rows if c > 0]
    streak = _compute_streak(active_days, today=datetime.now().date())
    score = int(
        total_shapes * 2 + labeled_files * 8 + streak * 30 + len(active_days) * 12
    )
    level, xp_into, xp_span = _compute_level(score)
    achievements = _build_achievements(
        labeled_files=labeled_files,
        total_shapes=total_shapes,
        streak_days=streak,
        coverage_percent=coverage_percent,
    )

    return LabelingProjectStats(
        project_root=project_root.resolve(),
        annotation_root=annotation_root.resolve(),
        total_images=total_images,
        total_annotation_files=total_annotation_files,
        labeled_files=labeled_files,
        total_shapes=total_shapes,
        created_files=created_files,
        edited_files=edited_files,
        coverage_percent=coverage_percent,
        top_labels=top_labels,
        annotator_file_counts=annotator_file_counts,
        annotator_shape_counts=annotator_shape_counts,
        shape_type_counts=shape_type_counts,
        activity_last_7_days=activity_rows,
        gamification=GamificationStats(
            score=score,
            level=level,
            xp_into_level=xp_into,
            xp_to_next_level=max(1, xp_span - xp_into),
            streak_days=streak,
        ),
        achievements=achievements,
    )


class LabelingProgressDashboardWidget(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        initial_project_root: str | Path | None = None,
        initial_annotation_root: str | Path | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._project_root = (
            Path(initial_project_root).expanduser().resolve()
            if initial_project_root
            else Path.cwd()
        )
        self._annotation_root = (
            Path(initial_annotation_root).expanduser().resolve()
            if initial_annotation_root
            else self._project_root
        )
        self._last_stats: LabelingProjectStats | None = None
        self._toast: QtWidgets.QLabel | None = None
        self._milestone_settings = QtCore.QSettings("Annolid", "Annolid")
        self._refresh_token = 0
        self._refresh_thread: QtCore.QThread | None = None
        self._refresh_worker: _DashboardRefreshWorker | None = None
        self._setup_ui()
        self.refresh_stats()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        title = QtWidgets.QLabel("Labeling Progress Dashboard", self)
        title.setObjectName("dashTitle")
        subtitle = QtWidgets.QLabel(
            "Track annotation output, activity, and gamified momentum.",
            self,
        )
        subtitle.setObjectName("dashSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        path_row = QtWidgets.QHBoxLayout()
        self.project_path_edit = QtWidgets.QLineEdit(str(self._project_root), self)
        self.project_path_edit.setPlaceholderText("Project root (images)")
        self.annotation_path_edit = QtWidgets.QLineEdit(
            str(self._annotation_root), self
        )
        self.annotation_path_edit.setPlaceholderText("Annotation root (LabelMe JSON)")
        browse_project = QtWidgets.QPushButton("Browse Project", self)
        browse_annotations = QtWidgets.QPushButton("Browse Annotations", self)
        self.refresh_btn = QtWidgets.QPushButton("Refresh", self)
        self.refresh_busy = QtWidgets.QProgressBar(self)
        self.refresh_busy.setRange(0, 0)
        self.refresh_busy.setTextVisible(False)
        self.refresh_busy.setFixedWidth(96)
        self.refresh_busy.setVisible(False)
        browse_project.clicked.connect(self._browse_project_root)
        browse_annotations.clicked.connect(self._browse_annotation_root)
        self.refresh_btn.clicked.connect(self.refresh_stats)
        path_row.addWidget(self.project_path_edit, 2)
        path_row.addWidget(browse_project)
        path_row.addWidget(self.annotation_path_edit, 2)
        path_row.addWidget(browse_annotations)
        path_row.addWidget(self.refresh_busy)
        path_row.addWidget(self.refresh_btn)
        layout.addLayout(path_row)

        cards_layout = QtWidgets.QHBoxLayout()
        cards_layout.setSpacing(10)
        self.files_card = _StatCard("Labeled Files", self)
        self.shapes_card = _StatCard("Total Shapes", self)
        self.coverage_card = _StatCard("Coverage", self)
        self.edits_card = _StatCard("Created vs Edited", self)
        cards_layout.addWidget(self.files_card)
        cards_layout.addWidget(self.shapes_card)
        cards_layout.addWidget(self.coverage_card)
        cards_layout.addWidget(self.edits_card)
        layout.addLayout(cards_layout)

        game_group = QtWidgets.QGroupBox("Gamification", self)
        game_layout = QtWidgets.QVBoxLayout(game_group)
        self.level_label = QtWidgets.QLabel("Level --  |  Score --", game_group)
        self.level_label.setObjectName("gameHeadline")
        self.streak_label = QtWidgets.QLabel("Streak: -- days", game_group)
        self.xp_progress = QtWidgets.QProgressBar(game_group)
        self.xp_progress.setRange(0, 100)
        self.achievements_label = QtWidgets.QLabel("", game_group)
        self.achievements_label.setWordWrap(True)
        game_layout.addWidget(self.level_label)
        game_layout.addWidget(self.streak_label)
        game_layout.addWidget(self.xp_progress)
        game_layout.addWidget(self.achievements_label)
        layout.addWidget(game_group)

        charts_group = QtWidgets.QGroupBox("Quick Visuals", self)
        charts_layout = QtWidgets.QHBoxLayout(charts_group)
        charts_layout.setSpacing(10)
        self.label_bars = _BarsWidget(charts_group)
        self.shape_bars = _BarsWidget(charts_group)
        self.activity_bars = _BarsWidget(charts_group)
        charts_layout.addWidget(self.label_bars, 1)
        charts_layout.addWidget(self.shape_bars, 1)
        charts_layout.addWidget(self.activity_bars, 1)
        layout.addWidget(charts_group)

        lower_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        left = QtWidgets.QGroupBox("Top Labels", lower_split)
        left_layout = QtWidgets.QVBoxLayout(left)
        self.labels_table = QtWidgets.QTableWidget(0, 2, left)
        self.labels_table.setHorizontalHeaderLabels(["Label", "Count"])
        self.labels_table.horizontalHeader().setStretchLastSection(True)
        self.labels_table.verticalHeader().setVisible(False)
        self.labels_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        left_layout.addWidget(self.labels_table)
        self.annotators_table = QtWidgets.QTableWidget(0, 3, left)
        self.annotators_table.setHorizontalHeaderLabels(
            ["Annotator", "Files", "Shapes"]
        )
        self.annotators_table.horizontalHeader().setStretchLastSection(True)
        self.annotators_table.verticalHeader().setVisible(False)
        self.annotators_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        left_layout.addWidget(self.annotators_table)

        right = QtWidgets.QGroupBox("Shape Types + Last 7 Days Activity", lower_split)
        right_layout = QtWidgets.QVBoxLayout(right)
        self.shape_types_table = QtWidgets.QTableWidget(0, 2, right)
        self.shape_types_table.setHorizontalHeaderLabels(["Shape Type", "Count"])
        self.shape_types_table.horizontalHeader().setStretchLastSection(True)
        self.shape_types_table.verticalHeader().setVisible(False)
        self.shape_types_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self.activity_table = QtWidgets.QTableWidget(0, 2, right)
        self.activity_table.setHorizontalHeaderLabels(["Date", "Files Updated"])
        self.activity_table.horizontalHeader().setStretchLastSection(True)
        self.activity_table.verticalHeader().setVisible(False)
        self.activity_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        right_layout.addWidget(self.shape_types_table)
        right_layout.addWidget(self.activity_table)
        lower_split.addWidget(left)
        lower_split.addWidget(right)
        lower_split.setSizes([420, 520])
        layout.addWidget(lower_split, 1)

        self.status_label = QtWidgets.QLabel("", self)
        layout.addWidget(self.status_label)

        self.setStyleSheet(
            """
            QLabel#dashTitle { font-size: 22px; font-weight: 800; color: #102a43; }
            QLabel#dashSubtitle { color: #486581; font-size: 12px; }
            QFrame#statCard {
                border: 1px solid #d9e2ec;
                border-radius: 10px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #f8fbff,stop:1 #eef7ff);
            }
            QLabel#cardTitle { color: #486581; font-size: 11px; font-weight: 600; }
            QLabel#cardValue { color: #102a43; font-size: 24px; font-weight: 800; }
            QLabel#cardSub { color: #627d98; font-size: 11px; }
            QLabel#gameHeadline { font-size: 16px; font-weight: 700; color: #1f3c88; }
            """
        )

    def _browse_project_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Project Root",
            self.project_path_edit.text().strip() or str(Path.cwd()),
        )
        if path:
            self.project_path_edit.setText(path)

    def _browse_annotation_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Annotation Root",
            self.annotation_path_edit.text().strip()
            or self.project_path_edit.text().strip(),
        )
        if path:
            self.annotation_path_edit.setText(path)

    def refresh_stats(self) -> None:
        project_root = Path(self.project_path_edit.text().strip() or ".").expanduser()
        annotation_root = Path(
            self.annotation_path_edit.text().strip() or str(project_root)
        ).expanduser()
        if not project_root.exists():
            self.refresh_busy.setVisible(False)
            self.refresh_btn.setEnabled(True)
            self.status_label.setText(f"Project root not found: {project_root}")
            return
        if not annotation_root.exists():
            self.refresh_busy.setVisible(False)
            self.refresh_btn.setEnabled(True)
            self.status_label.setText(f"Annotation root not found: {annotation_root}")
            return

        self._refresh_token += 1
        token = self._refresh_token
        self.refresh_btn.setEnabled(False)
        self.refresh_busy.setVisible(True)
        self.status_label.setText("Refreshing stats...")
        index_file = self._resolve_index_file(project_root)
        self._start_refresh_worker(
            token=token,
            project_root=project_root,
            annotation_root=annotation_root,
            index_file=index_file,
        )

    def _start_refresh_worker(
        self,
        *,
        token: int,
        project_root: Path,
        annotation_root: Path,
        index_file: Path | None,
    ) -> None:
        old_thread = self._refresh_thread
        if old_thread is not None and old_thread.isRunning():
            old_thread.quit()
            old_thread.wait(200)
        self._refresh_thread = None
        self._refresh_worker = None

        thread = QtCore.QThread(self)
        worker = _DashboardRefreshWorker(
            token=token,
            project_root=project_root,
            annotation_root=annotation_root,
            index_file=index_file,
        )
        worker.moveToThread(thread)
        worker.finished.connect(self._on_refresh_finished)
        worker.finished.connect(thread.quit)
        thread.started.connect(worker.run)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_refresh_thread_finished)
        self._refresh_thread = thread
        self._refresh_worker = worker
        thread.start()

    @QtCore.Slot()
    def _on_refresh_thread_finished(self) -> None:
        self._refresh_thread = None
        self._refresh_worker = None

    @QtCore.Slot(int, object, str, str)
    def _on_refresh_finished(
        self,
        token: int,
        stats: object,
        source: str,
        error: str,
    ) -> None:
        if token != self._refresh_token:
            return
        self.refresh_btn.setEnabled(True)
        self.refresh_busy.setVisible(False)
        if error:
            self.status_label.setText(f"Error scanning stats: {error}")
            return
        if not isinstance(stats, LabelingProjectStats):
            self.status_label.setText("Failed to parse dashboard stats.")
            return
        self._render_stats(stats, source=source)

    def _render_stats(self, stats: LabelingProjectStats, *, source: str) -> None:
        self.files_card.value_label.setText(str(stats.labeled_files))
        self.files_card.sub_label.setText(
            f"{stats.total_annotation_files} annotation JSONs scanned"
        )
        self.shapes_card.value_label.setText(str(stats.total_shapes))
        self.shapes_card.sub_label.setText(f"{len(stats.top_labels)} active labels")
        self.coverage_card.value_label.setText(f"{stats.coverage_percent:.1f}%")
        self.coverage_card.sub_label.setText(
            f"{stats.labeled_files}/{stats.total_images} images labeled"
        )
        self.edits_card.value_label.setText(
            f"{stats.created_files}/{stats.edited_files}"
        )
        self.edits_card.sub_label.setText("created / edited files")

        g = stats.gamification
        self.level_label.setText(f"Level {g.level}  |  Score {g.score}")
        self.streak_label.setText(
            f"Streak: {g.streak_days} day(s)  |  XP to next level: {g.xp_to_next_level}"
        )
        level_span = max(1, g.xp_into_level + g.xp_to_next_level)
        self.xp_progress.setValue(
            int(min(100, max(0, (g.xp_into_level / level_span) * 100)))
        )

        unlocked = [a.name for a in stats.achievements if a.unlocked]
        locked = [a.name for a in stats.achievements if not a.unlocked]
        self.achievements_label.setText(
            f"Unlocked ({len(unlocked)}): {', '.join(unlocked) if unlocked else 'None yet'}\n"
            f"Locked ({len(locked)}): {', '.join(locked[:4]) + ('...' if len(locked) > 4 else '')}"
        )

        self._fill_table(self.labels_table, stats.top_labels)
        self._fill_table(self.shape_types_table, stats.shape_type_counts)
        self._fill_table(
            self.activity_table,
            [(d.isoformat(), c) for d, c in stats.activity_last_7_days],
        )
        annotator_rows = self._merge_annotator_rows(
            stats.annotator_file_counts, stats.annotator_shape_counts
        )
        self._fill_annotators_table(annotator_rows)
        self.label_bars.set_series(stats.top_labels[:7])
        self.shape_bars.set_series(stats.shape_type_counts[:7])
        self.activity_bars.set_series(
            [(d.strftime("%m-%d"), c) for d, c in stats.activity_last_7_days]
        )
        self._notify_milestones(stats)
        self._last_stats = stats

        self.status_label.setText(
            f"Source={source} | project={stats.project_root} | annotations={stats.annotation_root}"
        )

    def apply_index_stats_snapshot(self, payload: Dict[str, object]) -> None:
        project_root = Path(self.project_path_edit.text().strip() or ".").expanduser()
        annotation_root = Path(
            self.annotation_path_edit.text().strip() or str(project_root)
        ).expanduser()
        stats = _stats_from_index_payload(
            payload,
            project_root=project_root,
            annotation_root=annotation_root,
        )
        if stats is None:
            return
        # Invalidate any in-flight refresh so direct-save update wins immediately.
        self._refresh_token += 1
        self.refresh_btn.setEnabled(True)
        self.refresh_busy.setVisible(False)
        self._render_stats(stats, source="label_index (live)")

    def _resolve_index_file(self, project_root: Path) -> Path | None:
        env_value = str(
            QtCore.QProcessEnvironment.systemEnvironment().value(
                "ANNOLID_LABEL_INDEX_FILE", ""
            )
        ).strip()
        if env_value:
            candidate = Path(env_value).expanduser()
            if candidate.is_absolute():
                return candidate.resolve()
            return (project_root.resolve() / candidate).resolve()

        qsettings = QtCore.QSettings("Annolid", "Annolid")
        settings_value = str(
            qsettings.value("dataset/label_index_file", "", type=str) or ""
        ).strip()
        if settings_value:
            candidate = Path(settings_value).expanduser()
            if candidate.is_absolute():
                return candidate.resolve()
            return (project_root.resolve() / candidate).resolve()

        return default_label_index_path(project_root.resolve())

    @staticmethod
    def _fill_table(
        table: QtWidgets.QTableWidget, rows: Sequence[tuple[object, object]]
    ) -> None:
        table.setRowCount(len(rows))
        for r, (col0, col1) in enumerate(rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(col0)))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(col1)))
        table.resizeColumnsToContents()

    @staticmethod
    def _merge_annotator_rows(
        files: Sequence[tuple[str, int]],
        shapes: Sequence[tuple[str, int]],
    ) -> List[tuple[str, int, int]]:
        file_map = {k: int(v) for k, v in files}
        shape_map = {k: int(v) for k, v in shapes}
        keys = sorted(set(file_map.keys()) | set(shape_map.keys()))
        return [(k, file_map.get(k, 0), shape_map.get(k, 0)) for k in keys]

    def _fill_annotators_table(self, rows: Sequence[tuple[str, int, int]]) -> None:
        self.annotators_table.setRowCount(len(rows))
        for r, (name, files, shapes) in enumerate(rows):
            self.annotators_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(name)))
            self.annotators_table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(files)))
            self.annotators_table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(shapes)))
        self.annotators_table.resizeColumnsToContents()

    def _notify_milestones(self, stats: LabelingProjectStats) -> None:
        project_key = f"labeling_dashboard/{stats.project_root.as_posix()}"
        prev_level = int(self._milestone_settings.value(f"{project_key}/level", 0))
        prev_unlocked_raw = str(
            self._milestone_settings.value(f"{project_key}/achievements", "")
        )
        prev_unlocked = {x for x in prev_unlocked_raw.split(",") if x}
        unlocked_now = {a.name for a in stats.achievements if a.unlocked}

        messages: List[str] = []
        if prev_level > 0 and stats.gamification.level > prev_level:
            messages.append(f"Level up! {prev_level} -> {stats.gamification.level}")
        new_unlocks = sorted(unlocked_now - prev_unlocked)
        if prev_unlocked and new_unlocks:
            messages.append(f"New achievements: {', '.join(new_unlocks)}")
        if messages:
            self._show_toast(" | ".join(messages))

        self._milestone_settings.setValue(
            f"{project_key}/level", stats.gamification.level
        )
        self._milestone_settings.setValue(
            f"{project_key}/achievements", ",".join(sorted(unlocked_now))
        )

    def _show_toast(self, text: str) -> None:
        toast = QtWidgets.QLabel(text, self)
        toast.setWindowFlags(QtCore.Qt.ToolTip)
        toast.setStyleSheet(
            "QLabel { background:#102a43; color:#f0f4f8; border:1px solid #486581; padding:8px 10px; border-radius:8px; }"
        )
        toast.adjustSize()
        pos = self.mapToGlobal(
            QtCore.QPoint(
                max(8, self.width() - toast.width() - 16),
                16,
            )
        )
        toast.move(pos)
        toast.show()
        QtCore.QTimer.singleShot(2600, toast.close)
        self._toast = toast


class LabelingProgressDashboardDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        initial_project_root: str | Path | None = None,
        initial_annotation_root: str | Path | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Labeling Progress Dashboard")
        self.resize(1180, 780)
        layout = QtWidgets.QVBoxLayout(self)
        self.dashboard = LabelingProgressDashboardWidget(
            initial_project_root=initial_project_root,
            initial_annotation_root=initial_annotation_root,
            parent=self,
        )
        layout.addWidget(self.dashboard)
