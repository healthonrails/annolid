from __future__ import annotations

import os
from pathlib import Path


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _get_qapp():
    from qtpy import QtWidgets

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_sim2real_studio_dialog_controls_and_state_sync(tmp_path: Path) -> None:
    _get_qapp()
    from annolid.gui.widgets.sim2real_dialog import Sim2RealStudioDialog
    from annolid.sim2real import (
        Sim2RealIterationRecord,
        Sim2RealWorkflowService,
        SimDatasetConfig,
        SimIterationAdjustment,
        SimIterationEvaluation,
        VideoEnvironmentProfile,
    )

    proj_dir = tmp_path / "gui_proj"
    svc = Sim2RealWorkflowService(proj_dir)
    state = svc.load_state()
    state.example_videos = [str(tmp_path / "demo.mp4")]
    state.profile = VideoEnvironmentProfile(
        example_videos=[str(tmp_path / "demo.mp4")],
        species="mouse",
        recommended_arena="circular_open_field",
        recommended_camera_view="top-close",
        recommended_lighting="calms_gray",
        recommended_coat_pair="c57_pair",
    ).to_dict()
    state.current_sim_config = SimDatasetConfig(
        species="mouse",
        arena="circular_open_field",
        camera_view="top-close",
        lighting="calms_gray",
        coat_pair="c57_pair",
        scenarios=["arena_explore", "social_huddle"],
        frames_per_scenario=16,
    ).to_dict()
    state.iterations = [
        Sim2RealIterationRecord(
            iteration=1,
            status="completed",
            dataset={"renderer": "threejs_two_mice_3d", "total_frames": 32},
            evaluation=SimIterationEvaluation(
                val_mask_map50=0.81,
                evaluated_frames=12,
                frames_with_detections=10,
                detection_coverage=0.8333,
                mean_confidence=0.76,
                temporal_iou_stability=0.72,
                sim2real_score=0.79,
            ).to_dict(),
            adjustment=SimIterationAdjustment(
                diagnosis=["Moderate coverage gap on edge frames."],
                recommendations=["Increase frames per scenario to 24."],
                suggested_sim_config=SimDatasetConfig(
                    species="mouse",
                    arena="circular_open_field",
                    frames_per_scenario=24,
                    seed=143,
                ).to_dict(),
                suggested_train_overrides={"epochs": 30, "imgsz": 768},
            ).to_dict(),
            best_weights_path=str(tmp_path / "best.pt"),
        ).to_dict()
    ]
    svc.save_state(state)

    dlg = Sim2RealStudioDialog(project_dir=proj_dir)
    try:
        assert dlg.tabs.count() == 5
        assert dlg.sim_species_combo.currentData() == "mouse"
        assert dlg.sim_arena_combo.currentData() == "circular_open_field"
        assert dlg.sim_lighting_combo.currentData() == "calms_gray"
        assert dlg.iter_table.rowCount() == 1
        assert "Increase frames per scenario to 24" in dlg.advisor_text.toPlainText()

        # Switch to FlyBody engine and verify camera/behavior options switch cleanly
        fly_idx = dlg.sim_species_combo.findData("fly")
        assert fly_idx >= 0
        dlg.sim_species_combo.setCurrentIndex(fly_idx)
        assert dlg.fly_count_spin.isEnabled() is True
        assert dlg.sim_coat_combo.isEnabled() is False
        cfg_fly = dlg.build_sim_config_from_ui()
        assert cfg_fly.species == "fly"
        assert cfg_fly.camera_view in {
            "walker/track1",
            "walker/track2",
            "walker/track3",
            "walker/hero",
            "walker/side",
            "walker/back",
            "top_camera",
        }

        # Apply advisor recommendations -> should update UI controls and switch back to Mouse config
        dlg._on_apply_recommendation_clicked()
        assert dlg.frames_per_scen_spin.value() == 24
        assert dlg.epochs_spin.value() == 30
        assert dlg.imgsz_spin.value() == 768

        # Generate report from UI button
        dlg._on_generate_report_clicked()
        assert (proj_dir / "reports" / "sim2real_report.md").is_file()
        assert (proj_dir / "reports" / "sim2real_report.html").is_file()
    finally:
        dlg.close()


def test_colab_dialog_has_sim2real_launcher_button(tmp_path: Path) -> None:
    _get_qapp()
    from annolid.colab.service import ColabWorkflowService
    from annolid.gui.widgets.colab_dialog import ColabStudioDialog

    dlg = ColabStudioDialog(
        service=ColabWorkflowService(home_dir=tmp_path / "colab_home"),
        auto_refresh_account=False,
    )
    try:
        assert hasattr(dlg, "header_sim2real_btn")
        assert "Sim-to-Real" in dlg.header_sim2real_btn.text()
    finally:
        dlg.close()
