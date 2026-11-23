"""Stateless builder for Canvas's flat, icon-first context menu."""

from __future__ import annotations

from qtpy import QtWidgets


def build_canvas_context_menu(canvas, main_window) -> QtWidgets.QMenu:
    """Build a flat, icon-first context menu for drawing and shape actions."""
    menu = QtWidgets.QMenu(canvas)
    menu.setToolTipsVisible(True)

    def _add_existing_action(
        act: QtWidgets.QAction | None,
        *,
        icon_filename: str | None = None,
        fallback_standard: QtWidgets.QStyle.StandardPixmap | None = None,
    ) -> bool:
        if act is not None:
            canvas._ensure_action_icon(
                act,
                icon_filename=icon_filename,
                fallback_standard=fallback_standard,
            )
            menu.addAction(act)
            return True
        return False

    def _add_draw_mode_action(
        mode: str,
        *,
        source_action: QtWidgets.QAction | None = None,
        text: str | None = None,
        edit: bool = False,
        icon_filename: str | None = None,
        fallback_standard: QtWidgets.QStyle.StandardPixmap | None = None,
    ) -> bool:
        if source_action is None and text is None:
            return False
        action = QtWidgets.QAction(text or source_action.text(), menu)
        if source_action is not None:
            action.setEnabled(source_action.isEnabled())
            action.setToolTip(source_action.toolTip())
            if not source_action.icon().isNull():
                action.setIcon(source_action.icon())
        canvas._ensure_action_icon(
            action,
            icon_filename=icon_filename,
            fallback_standard=fallback_standard,
        )

        def _trigger():
            canvas.cancelCurrentDrawing(clear_sam_mask=True)
            if hasattr(main_window, "toggleDrawMode"):
                main_window.toggleDrawMode(
                    bool(edit),
                    createMode="polygon" if edit else mode,
                )

        action.triggered.connect(_trigger)
        menu.addAction(action)
        return True

    # Right-drag Move Here / Copy Here when shadow copies exist.
    if getattr(canvas, "selectedShapesCopy", None) and getattr(
        canvas, "selectedShapes", None
    ):
        has_locked = any(s.is_locked() for s in canvas.selectedShapes)
        copy_here = QtWidgets.QAction(
            canvas._icon("duplicate_polygons.svg"),
            "Copy Here",
            menu,
        )
        canvas._ensure_action_icon(
            copy_here, fallback_standard=QtWidgets.QStyle.SP_FileDialogNewFolder
        )
        copy_here.triggered.connect(lambda: canvas.completeRightDragMove(copy=True))
        menu.addAction(copy_here)

        move_here = QtWidgets.QAction(
            canvas._icon("edit_polygons.svg"),
            "Move Here",
            menu,
        )
        canvas._ensure_action_icon(
            move_here, fallback_standard=QtWidgets.QStyle.SP_ArrowForward
        )
        move_here.setEnabled(not has_locked)
        move_here.triggered.connect(lambda: canvas.completeRightDragMove(copy=False))
        menu.addAction(move_here)
        menu.addSeparator()

    # Flat, user-friendly context menu: edit first, then draw modes, then AI.
    actions = getattr(main_window, "actions", None)
    if actions is not None:
        _add_draw_mode_action(
            "polygon",
            source_action=getattr(actions, "editMode", None),
            edit=True,
            icon_filename="edit_polygons.svg",
        )
        menu.addSeparator()
        _add_draw_mode_action(
            "polygon",
            source_action=getattr(actions, "createMode", None),
            icon_filename="create_polygons.svg",
        )
        _add_draw_mode_action(
            "rectangle",
            source_action=getattr(actions, "createRectangleMode", None),
            fallback_standard=QtWidgets.QStyle.SP_FileDialogDetailedView,
        )
        _add_draw_mode_action(
            "oriented_rectangle",
            source_action=getattr(actions, "createOrientedRectangleMode", None),
            fallback_standard=QtWidgets.QStyle.SP_FileDialogDetailedView,
        )
        _add_draw_mode_action(
            "circle",
            source_action=getattr(actions, "createCircleMode", None),
            fallback_standard=QtWidgets.QStyle.SP_BrowserReload,
        )
        _add_draw_mode_action(
            "circle_3p",
            source_action=getattr(actions, "createCircle3PMode", None),
            fallback_standard=QtWidgets.QStyle.SP_BrowserReload,
        )
        _add_draw_mode_action(
            "line",
            source_action=getattr(actions, "createLineMode", None),
            fallback_standard=QtWidgets.QStyle.SP_ArrowRight,
        )
        _add_draw_mode_action(
            "point",
            source_action=getattr(actions, "createPointMode", None),
            fallback_standard=QtWidgets.QStyle.SP_DialogApplyButton,
        )
        _add_draw_mode_action(
            "linestrip",
            source_action=getattr(actions, "createLineStripMode", None),
            fallback_standard=QtWidgets.QStyle.SP_MediaSeekForward,
        )
        menu.addSeparator()
        _add_draw_mode_action(
            "ai_polygon",
            source_action=getattr(actions, "createAiPolygonMode", None),
            text="AI Polygon",
            icon_filename="ai_polygons.svg",
        )
        _add_draw_mode_action(
            "ai_mask",
            source_action=getattr(actions, "createAiMaskMode", None),
            text="AI Mask",
            icon_filename="ai_polygons.svg",
        )
        _add_draw_mode_action(
            "grounding_sam",
            source_action=getattr(actions, "createGroundingSAMMode", None),
            text="Grounding SAM",
            fallback_standard=QtWidgets.QStyle.SP_ComputerIcon,
        )

    # ------------------------------------------------------------
    # SAM3 session operations
    # ------------------------------------------------------------
    if (
        hasattr(main_window, "reset_sam3_session")
        or hasattr(main_window, "remove_sam3_object")
        or hasattr(main_window, "close_sam3_session")
    ):
        menu.addSeparator()
        if hasattr(main_window, "reset_sam3_session"):
            sam3_reset_action = QtWidgets.QAction(
                canvas._icon("undo.svg"),
                "Reset SAM3 Session",
                menu,
            )
            if sam3_reset_action.icon().isNull():
                sam3_reset_action.setIcon(
                    canvas.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
                )
            sam3_reset_action.triggered.connect(
                lambda: main_window.reset_sam3_session()
            )
            menu.addAction(sam3_reset_action)
        if hasattr(main_window, "close_sam3_session"):
            sam3_close_action = QtWidgets.QAction(
                canvas._icon("close.svg"),
                "Close SAM3 Session",
                menu,
            )
            if sam3_close_action.icon().isNull():
                sam3_close_action.setIcon(
                    canvas.style().standardIcon(QtWidgets.QStyle.SP_DialogCloseButton)
                )
            sam3_close_action.triggered.connect(
                lambda: main_window.close_sam3_session()
            )
            menu.addAction(sam3_close_action)
        if hasattr(main_window, "remove_sam3_object"):
            sam3_remove_action = QtWidgets.QAction(
                canvas._icon("delete_polygons.svg"),
                "Remove SAM3 Object...",
                menu,
            )
            if sam3_remove_action.icon().isNull():
                sam3_remove_action.setIcon(
                    canvas.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon)
                )
            sam3_remove_action.triggered.connect(
                lambda: main_window.remove_sam3_object()
            )
            menu.addAction(sam3_remove_action)

    active_editor = getattr(main_window, "_active_shape_editor", None)
    editor = active_editor() if callable(active_editor) else None
    tiled_editor = getattr(main_window, "large_image_view", None)
    selected_shapes = list(canvas.selectedShapes or [])
    if not selected_shapes and tiled_editor is not None:
        selected_shapes = list(getattr(tiled_editor, "selectedShapes", []) or [])
    can_start_adjoining = bool(
        getattr(editor, "canStartAdjoiningPolygon", lambda: False)()
    ) or bool(getattr(tiled_editor, "canStartAdjoiningPolygon", lambda: False)())
    can_start_boundary_reshape = bool(
        getattr(editor, "canStartSharedBoundaryReshape", lambda: False)()
    ) or bool(getattr(tiled_editor, "canStartSharedBoundaryReshape", lambda: False)())
    can_infer_page_polygons = bool(
        getattr(main_window, "canInferCurrentLargeImagePagePolygons", lambda: False)()
    )
    has_clipboard = bool(getattr(main_window, "_copied_shapes", None))

    # ------------------------------------------------------------
    # Shape operations
    # ------------------------------------------------------------
    if (
        selected_shapes
        or has_clipboard
        or can_start_adjoining
        or can_start_boundary_reshape
        or can_infer_page_polygons
    ):
        menu.addSeparator()
        if can_infer_page_polygons:
            _add_existing_action(
                getattr(actions, "inferPagePolygons", None),
                icon_filename="duplicate_polygons.svg",
            )
        if selected_shapes:
            if actions is not None:
                _add_existing_action(
                    getattr(actions, "editLabel", None),
                    icon_filename="edit_polygons.svg",
                )
                lock_action = getattr(actions, "toggleLockShapes", None)
                if lock_action is not None:
                    all_locked = all(s.is_locked() for s in selected_shapes)
                    lock_action.setText("Unlock Shape" if all_locked else "Lock Shape")
                    _add_existing_action(
                        lock_action,
                        fallback_standard=QtWidgets.QStyle.SP_DialogApplyButton,
                    )
                _add_existing_action(
                    getattr(actions, "fitSelection", None),
                    icon_filename="fit_window.svg",
                )
            propagate_action = QtWidgets.QAction(
                canvas._icon("next_frame.svg"),
                "Propagate Selected Shape",
                menu,
            )
            propagate_action.triggered.connect(
                lambda: canvas.propagateSelectedShapeFromCanvas()
            )
            menu.addAction(propagate_action)
            if len(selected_shapes) == 2:
                switch_labels_action = QtWidgets.QAction(
                    canvas._icon("edit_polygons.svg"),
                    "Switch Selected Shape Labels",
                    menu,
                )
                switch_labels_action.triggered.connect(
                    lambda: canvas.switchSelectedShapeLabelsFromCanvas()
                )
                menu.addAction(switch_labels_action)

            has_point_selected = any(
                str(getattr(s, "shape_type", "")).lower() == "point"
                for s in selected_shapes
            )
            if has_point_selected:
                menu.addSeparator()
                first_pt = next(
                    s
                    for s in selected_shapes
                    if str(getattr(s, "shape_type", "")).lower() == "point"
                )
                if hasattr(canvas, "selectPoseInstanceShapes"):
                    select_inst_action = QtWidgets.QAction(
                        "Select Whole Pose Instance",
                        menu,
                    )
                    canvas._ensure_action_icon(
                        select_inst_action,
                        icon_filename="edit_polygons.svg",
                        fallback_standard=QtWidgets.QStyle.SP_FileDialogContentsView,
                    )
                    select_inst_action.triggered.connect(
                        lambda: canvas.selectPoseInstanceShapes(first_pt)
                    )
                    menu.addAction(select_inst_action)
                if hasattr(canvas, "transformSelectedPoseShapes"):
                    for label_txt, kwargs, fallback_icon in (
                        (
                            "Rotate Pose Left (-15°)",
                            {"angle_deg": -15.0},
                            QtWidgets.QStyle.SP_ArrowLeft,
                        ),
                        (
                            "Rotate Pose Right (+15°)",
                            {"angle_deg": 15.0},
                            QtWidgets.QStyle.SP_ArrowRight,
                        ),
                        (
                            "Scale Pose Up (+10%)",
                            {"scale": 1.10},
                            QtWidgets.QStyle.SP_ArrowUp,
                        ),
                        (
                            "Scale Pose Down (-10%)",
                            {"scale": 0.90},
                            QtWidgets.QStyle.SP_ArrowDown,
                        ),
                    ):
                        act = QtWidgets.QAction(label_txt, menu)
                        canvas._ensure_action_icon(act, fallback_standard=fallback_icon)
                        act.triggered.connect(
                            lambda _checked=False, kw=kwargs: (
                                canvas.selectPoseInstanceShapes(first_pt)
                                if len(canvas.selectedShapes) <= 1
                                else None,
                                canvas.transformSelectedPoseShapes(**kw),
                            )
                        )
                        menu.addAction(act)
                _add_existing_action(
                    getattr(main_window, "toggle_keypoint_visibility_action", None),
                    fallback_standard=QtWidgets.QStyle.SP_DialogApplyButton,
                )
                _add_existing_action(
                    getattr(main_window, "mark_keypoint_visible_action", None),
                    fallback_standard=QtWidgets.QStyle.SP_DialogYesButton,
                )
                _add_existing_action(
                    getattr(main_window, "mark_keypoint_occluded_action", None),
                    fallback_standard=QtWidgets.QStyle.SP_DialogNoButton,
                )
                if hasattr(main_window, "open_pose_workbench"):
                    swap_pose_action = QtWidgets.QAction(
                        "Swap Pose Instances Forward…",
                        menu,
                    )
                    canvas._ensure_action_icon(
                        swap_pose_action,
                        icon_filename="next_frame.svg",
                        fallback_standard=QtWidgets.QStyle.SP_BrowserReload,
                    )
                    swap_pose_action.triggered.connect(
                        lambda: main_window.open_pose_workbench(initial_tab="video")
                    )
                    menu.addAction(swap_pose_action)

        if actions is not None:
            if selected_shapes:
                _add_existing_action(
                    getattr(actions, "copyShapes", None),
                    icon_filename="duplicate_polygons.svg",
                )
            if has_clipboard:
                _add_existing_action(
                    getattr(actions, "pasteShapes", None),
                    icon_filename="duplicate_polygons.svg",
                )
            if selected_shapes:
                _add_existing_action(
                    getattr(actions, "duplicateShapes", None),
                    icon_filename="duplicate_polygons.svg",
                )
            merge_action = getattr(actions, "mergeShapes", None)
            if merge_action is not None and len(selected_shapes) >= 2:
                merge_action.setEnabled(
                    bool(
                        getattr(main_window, "canMergeSelectedShapes", lambda: False)()
                    )
                )
                _add_existing_action(merge_action, icon_filename="edit_polygons.svg")
            adjoining_action = getattr(actions, "startAdjoiningPolygon", None)
            if adjoining_action is not None and can_start_adjoining:
                adjoining_action.setEnabled(True)
                _add_existing_action(
                    adjoining_action, icon_filename="duplicate_polygons.svg"
                )
            collapse_action = getattr(actions, "collapsePolygons", None)
            if collapse_action is not None and selected_shapes:
                collapse_action.setEnabled(
                    bool(
                        getattr(
                            main_window,
                            "canCollapseSelectedPolygons",
                            lambda: False,
                        )()
                    )
                )
                _add_existing_action(
                    collapse_action, icon_filename="delete_polygons.svg"
                )
            restore_action = getattr(actions, "restorePolygons", None)
            if restore_action is not None and selected_shapes:
                restore_action.setEnabled(
                    bool(
                        getattr(
                            main_window,
                            "canRestoreSelectedPolygons",
                            lambda: False,
                        )()
                    )
                )
                _add_existing_action(restore_action, icon_filename="undo.svg")
            if can_start_boundary_reshape:
                boundary_reshape_action = QtWidgets.QAction(
                    canvas._icon("edit_polygons.svg"),
                    "Reshape Shared Boundary",
                    menu,
                )
                if boundary_reshape_action.icon().isNull():
                    boundary_reshape_action.setIcon(
                        canvas.style().standardIcon(
                            QtWidgets.QStyle.SP_DialogApplyButton
                        )
                    )

                def _trigger_boundary_reshape():
                    starter = getattr(main_window, "startSharedBoundaryReshape", None)
                    if callable(starter):
                        starter()

                boundary_reshape_action.triggered.connect(_trigger_boundary_reshape)
                menu.addAction(boundary_reshape_action)
            if selected_shapes:
                _add_existing_action(
                    getattr(actions, "deleteShapes", None),
                    icon_filename="delete_polygons.svg",
                )

        if selected_shapes and hasattr(main_window, "run_sam3d_reconstruction"):
            sam3d_action = QtWidgets.QAction(
                canvas._icon("reconstruct_3d.svg"),
                "Reconstruct 3D (SAM 3D)",
                menu,
            )
            if sam3d_action.icon().isNull():
                sam3d_action.setIcon(
                    canvas.style().standardIcon(QtWidgets.QStyle.SP_ComputerIcon)
                )
            sam3d_action.triggered.connect(
                lambda: main_window.run_sam3d_reconstruction()
            )
            menu.addAction(sam3d_action)
    return menu
