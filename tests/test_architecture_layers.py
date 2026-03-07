from __future__ import annotations

from pathlib import Path

from annolid import domain, infrastructure, interfaces, services
from annolid.domain import (
    BehaviorEvent,
    DeepLabCutTrainingImportConfig,
    InstanceRegistry,
    ProjectSchema,
    TimeBudgetRow,
    Track,
)
from annolid.infrastructure import AnnotationStore, configure_ultralytics_cache
from annolid.interfaces.background import TrackingWorker
from annolid.interfaces.bot import ChannelManager
from annolid.interfaces.cli import build_parser, parse_cli
from annolid.interfaces.gui import AnnolidWindow, create_qapp
from annolid.services import (
    audit_agent_secrets,
    add_agent_feedback,
    add_agent_cron_job,
    build_agent_regression_eval,
    build_chat_pdf_search_roots,
    build_chat_gui_context_payload,
    build_chat_vcs_read_roots,
    build_yolo_dataset_from_index,
    capture_chat_web_screenshot,
    check_for_agent_update,
    evaluate_agent_eval_gate,
    flush_agent_memory,
    clear_chat_session,
    emit_chat_chunk,
    emit_chat_final,
    emit_chat_progress,
    extract_chat_first_web_url,
    extract_chat_web_structured,
    find_chat_pdf_sections,
    find_chat_web_forms,
    get_agent_status,
    get_chat_allowed_read_roots,
    get_chat_pdf_state,
    get_chat_pdf_text,
    get_chat_workspace,
    get_chat_session_store,
    get_chat_web_dom_text,
    get_chat_web_state,
    get_chat_widget_action_result,
    has_chat_image_context,
    inspect_agent_memory,
    inspect_agent_skills,
    list_agent_cron_jobs,
    load_chat_history_messages,
    migrate_agent_secrets,
    invoke_chat_widget_json_slot,
    invoke_chat_widget_slot,
    label_chat_behavior_segments_tool,
    open_chat_in_browser_tool,
    open_chat_pdf_tool,
    open_chat_url_tool,
    open_chat_video_tool,
    onboard_agent_workspace,
    predict_behavior,
    persist_chat_turn,
    refresh_agent_skills,
    remove_agent_cron_job,
    remove_agent_secret,
    resolve_chat_pdf_path,
    resolve_chat_provider_kind,
    resolve_chat_video_path_for_gui_tool,
    rollback_agent_update,
    run_agent_acp_bridge,
    run_agent_cron_job,
    run_agent_pipeline,
    run_agent_eval,
    run_legacy_agent_update,
    run_agent_update,
    run_agent_security_audit,
    run_agent_security_check,
    run_behavior_training_cli,
    run_embedding_search,
    run_chat_fast_mode,
    run_chat_fast_provider_chat,
    run_chat_gemini,
    run_chat_ollama,
    run_chat_openai,
    run_chat_provider_fallback,
    run_chat_awaitable_sync,
    run_chat_web_steps,
    scroll_chat_web,
    segment_track_chat_video_tool,
    set_agent_secret,
    set_agent_cron_job_enabled,
    shadow_agent_skills,
    type_chat_web,
    validate_agent_tools,
)


def test_domain_layer_exports() -> None:
    assert domain.ProjectSchema is ProjectSchema
    assert domain.BehaviorEvent is BehaviorEvent
    assert domain.Track is Track
    assert domain.InstanceRegistry is InstanceRegistry
    assert domain.TimeBudgetRow is TimeBudgetRow
    assert domain.DeepLabCutTrainingImportConfig is DeepLabCutTrainingImportConfig


def test_services_layer_exports() -> None:
    assert services.run_agent_pipeline is run_agent_pipeline
    assert services.run_embedding_search is run_embedding_search
    assert services.predict_behavior is predict_behavior
    assert services.run_behavior_training_cli is run_behavior_training_cli
    assert services.build_yolo_dataset_from_index is build_yolo_dataset_from_index
    assert services.run_agent_security_check is run_agent_security_check
    assert services.run_agent_security_audit is run_agent_security_audit
    assert services.audit_agent_secrets is audit_agent_secrets
    assert services.set_agent_secret is set_agent_secret
    assert services.remove_agent_secret is remove_agent_secret
    assert services.migrate_agent_secrets is migrate_agent_secrets
    assert services.refresh_agent_skills is refresh_agent_skills
    assert services.inspect_agent_skills is inspect_agent_skills
    assert services.shadow_agent_skills is shadow_agent_skills
    assert services.add_agent_feedback is add_agent_feedback
    assert services.flush_agent_memory is flush_agent_memory
    assert services.clear_chat_session is clear_chat_session
    assert services.capture_chat_web_screenshot is capture_chat_web_screenshot
    assert services.emit_chat_chunk is emit_chat_chunk
    assert services.emit_chat_final is emit_chat_final
    assert services.emit_chat_progress is emit_chat_progress
    assert services.extract_chat_first_web_url is extract_chat_first_web_url
    assert services.extract_chat_web_structured is extract_chat_web_structured
    assert services.find_chat_pdf_sections is find_chat_pdf_sections
    assert services.find_chat_web_forms is find_chat_web_forms
    assert (
        services.label_chat_behavior_segments_tool is label_chat_behavior_segments_tool
    )
    assert services.inspect_agent_memory is inspect_agent_memory
    assert services.run_agent_eval is run_agent_eval
    assert services.build_agent_regression_eval is build_agent_regression_eval
    assert services.evaluate_agent_eval_gate is evaluate_agent_eval_gate
    assert services.check_for_agent_update is check_for_agent_update
    assert services.run_agent_update is run_agent_update
    assert services.rollback_agent_update is rollback_agent_update
    assert services.run_agent_acp_bridge is run_agent_acp_bridge
    assert services.run_legacy_agent_update is run_legacy_agent_update
    assert services.validate_agent_tools is validate_agent_tools
    assert services.onboard_agent_workspace is onboard_agent_workspace
    assert services.get_agent_status is get_agent_status
    assert services.list_agent_cron_jobs is list_agent_cron_jobs
    assert services.add_agent_cron_job is add_agent_cron_job
    assert services.remove_agent_cron_job is remove_agent_cron_job
    assert services.set_agent_cron_job_enabled is set_agent_cron_job_enabled
    assert services.run_agent_cron_job is run_agent_cron_job
    assert services.get_chat_workspace is get_chat_workspace
    assert services.get_chat_session_store is get_chat_session_store
    assert services.get_chat_allowed_read_roots is get_chat_allowed_read_roots
    assert services.get_chat_pdf_state is get_chat_pdf_state
    assert services.get_chat_pdf_text is get_chat_pdf_text
    assert services.build_chat_pdf_search_roots is build_chat_pdf_search_roots
    assert services.build_chat_gui_context_payload is build_chat_gui_context_payload
    assert services.build_chat_vcs_read_roots is build_chat_vcs_read_roots
    assert services.get_chat_web_dom_text is get_chat_web_dom_text
    assert services.get_chat_web_state is get_chat_web_state
    assert services.resolve_chat_pdf_path is resolve_chat_pdf_path
    assert services.resolve_chat_provider_kind is resolve_chat_provider_kind
    assert (
        services.resolve_chat_video_path_for_gui_tool
        is resolve_chat_video_path_for_gui_tool
    )
    assert services.get_chat_widget_action_result is get_chat_widget_action_result
    assert services.has_chat_image_context is has_chat_image_context
    assert services.run_chat_fast_mode is run_chat_fast_mode
    assert services.run_chat_fast_provider_chat is run_chat_fast_provider_chat
    assert services.run_chat_provider_fallback is run_chat_provider_fallback
    assert services.run_chat_ollama is run_chat_ollama
    assert services.run_chat_openai is run_chat_openai
    assert services.run_chat_gemini is run_chat_gemini
    assert services.run_chat_awaitable_sync is run_chat_awaitable_sync
    assert services.load_chat_history_messages is load_chat_history_messages
    assert services.invoke_chat_widget_slot is invoke_chat_widget_slot
    assert services.invoke_chat_widget_json_slot is invoke_chat_widget_json_slot
    assert services.open_chat_in_browser_tool is open_chat_in_browser_tool
    assert services.open_chat_pdf_tool is open_chat_pdf_tool
    assert services.open_chat_url_tool is open_chat_url_tool
    assert services.open_chat_video_tool is open_chat_video_tool
    assert services.persist_chat_turn is persist_chat_turn
    assert services.run_chat_web_steps is run_chat_web_steps
    assert services.scroll_chat_web is scroll_chat_web
    assert services.segment_track_chat_video_tool is segment_track_chat_video_tool
    assert services.type_chat_web is type_chat_web


def test_interfaces_layer_exports() -> None:
    assert interfaces.gui.AnnolidWindow is AnnolidWindow
    assert interfaces.gui.create_qapp is create_qapp
    assert interfaces.cli.build_parser is build_parser
    assert interfaces.cli.parse_cli is parse_cli
    assert interfaces.background.TrackingWorker is TrackingWorker
    assert interfaces.bot.ChannelManager is ChannelManager


def test_infrastructure_layer_exports() -> None:
    assert infrastructure.AnnotationStore is AnnotationStore
    assert infrastructure.configure_ultralytics_cache is configure_ultralytics_cache
    assert callable(infrastructure.find_most_recent_file)


def test_execution_paths_use_layer_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    cli_source = (repo_root / "annolid" / "engine" / "cli.py").read_text(
        encoding="utf-8"
    )
    assert (
        "from annolid.services.export import build_yolo_dataset_from_index"
        in cli_source
    )
    assert "from annolid.domain import DeepLabCutTrainingImportConfig" in cli_source
    assert (
        "from annolid.services.agent_admin import run_agent_security_check"
        in cli_source
    )
    assert (
        "from annolid.services.agent_admin import run_agent_security_audit"
        in cli_source
    )
    assert "from annolid.services.agent_admin import audit_agent_secrets" in cli_source
    assert "from annolid.services.agent_admin import set_agent_secret" in cli_source
    assert "from annolid.services.agent_admin import remove_agent_secret" in cli_source
    assert (
        "from annolid.services.agent_admin import migrate_agent_secrets" in cli_source
    )
    assert (
        "from annolid.services.agent_workspace import refresh_agent_skills"
        in cli_source
    )
    assert (
        "from annolid.services.agent_workspace import inspect_agent_skills"
        in cli_source
    )
    assert (
        "from annolid.services.agent_workspace import shadow_agent_skills" in cli_source
    )
    assert (
        "from annolid.services.agent_workspace import add_agent_feedback" in cli_source
    )
    assert (
        "from annolid.services.agent_workspace import flush_agent_memory" in cli_source
    )
    assert (
        "from annolid.services.agent_workspace import inspect_agent_memory"
        in cli_source
    )
    assert "from annolid.services.agent_eval import run_agent_eval" in cli_source
    assert (
        "from annolid.services.agent_eval import build_agent_regression_eval"
        in cli_source
    )
    assert (
        "from annolid.services.agent_eval import evaluate_agent_eval_gate" in cli_source
    )
    assert (
        "from annolid.services.agent_update import check_for_agent_update" in cli_source
    )
    assert "from annolid.services.agent_update import run_agent_update" in cli_source
    assert (
        "from annolid.services.agent_update import rollback_agent_update" in cli_source
    )
    assert (
        "from annolid.services.agent_update import run_legacy_agent_update"
        in cli_source
    )
    assert (
        "from annolid.services.agent_bridge import run_agent_acp_bridge" in cli_source
    )
    assert (
        "from annolid.services.agent_tooling import validate_agent_tools" in cli_source
    )
    assert (
        "from annolid.services.agent_cron import onboard_agent_workspace" in cli_source
    )
    assert "from annolid.services.agent_cron import get_agent_status" in cli_source
    assert "from annolid.services.agent_cron import list_agent_cron_jobs" in cli_source
    assert "from annolid.services.agent_cron import add_agent_cron_job" in cli_source
    assert "from annolid.services.agent_cron import remove_agent_cron_job" in cli_source
    assert (
        "from annolid.services.agent_cron import set_agent_cron_job_enabled"
        in cli_source
    )
    assert "from annolid.services.agent_cron import run_agent_cron_job" in cli_source

    behavior_controller = (
        repo_root / "annolid" / "gui" / "behavior_controller.py"
    ).read_text(encoding="utf-8")
    assert (
        "from annolid.domain import ProjectSchema, normalize_event_label"
        in behavior_controller
    )
    assert "from annolid.infrastructure import AnnotationStore" in behavior_controller

    gui_workers = (repo_root / "annolid" / "gui" / "workers.py").read_text(
        encoding="utf-8"
    )
    assert "from annolid.infrastructure.filesystem import (" in gui_workers

    tracking_worker = (repo_root / "annolid" / "jobs" / "tracking_worker.py").read_text(
        encoding="utf-8"
    )
    assert "from annolid.infrastructure.filesystem import (" in tracking_worker

    gui_app = (repo_root / "annolid" / "gui" / "app.py").read_text(encoding="utf-8")
    assert "from annolid.domain import ProjectSchema" in gui_app
    assert (
        "from annolid.infrastructure.runtime import create_qapp, sanitize_qt_plugin_env"
        in gui_app
    )
    assert "from annolid.infrastructure.runtime import (" in gui_app

    schema_loader = (
        repo_root / "annolid" / "gui" / "mixins" / "schema_behavior_loader_mixin.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.domain import (" in schema_loader

    project_workflow = (
        repo_root / "annolid" / "gui" / "mixins" / "project_workflow_mixin.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.domain import (" in project_workflow

    tooling_dialogs = (
        repo_root / "annolid" / "gui" / "mixins" / "tooling_dialogs_mixin.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.domain import (" in tooling_dialogs

    behavior_interaction = (
        repo_root / "annolid" / "gui" / "mixins" / "behavior_interaction_mixin.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.domain import ProjectSchema" in behavior_interaction
    assert "from annolid.infrastructure import AnnotationStore" in behavior_interaction

    persistence_lifecycle = (
        repo_root / "annolid" / "gui" / "mixins" / "persistence_lifecycle_mixin.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.infrastructure import AnnotationStore" in persistence_lifecycle
    assert "from annolid.infrastructure.filesystem import (" in persistence_lifecycle

    prediction_progress = (
        repo_root / "annolid" / "gui" / "mixins" / "prediction_progress_mixin.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.infrastructure import AnnotationStore" in prediction_progress

    annotation_loading = (
        repo_root / "annolid" / "gui" / "mixins" / "annotation_loading_mixin.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.infrastructure import AnnotationStore" in annotation_loading

    prediction_execution = (
        repo_root / "annolid" / "gui" / "mixins" / "prediction_execution_mixin.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.infrastructure import AnnotationStore" in prediction_execution
    assert "from annolid.infrastructure.filesystem import (" in prediction_execution

    project_dialog = (
        repo_root / "annolid" / "gui" / "widgets" / "project_dialog.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.domain import (" in project_dialog

    project_wizard = (
        repo_root / "annolid" / "gui" / "widgets" / "project_wizard.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.domain import (" in project_wizard

    behavior_controls = (
        repo_root / "annolid" / "gui" / "widgets" / "behavior_controls.py"
    ).read_text(encoding="utf-8")
    assert (
        "from annolid.domain import ModifierDefinition, SubjectDefinition"
        in behavior_controls
    )

    embedding_search_widget = (
        repo_root / "annolid" / "gui" / "widgets" / "embedding_search_widget.py"
    ).read_text(encoding="utf-8")
    assert (
        "from annolid.infrastructure import AnnotationStore" in embedding_search_widget
    )

    frame_similarity = (
        repo_root / "annolid" / "gui" / "frame_similarity_service.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.infrastructure import load_labelme_json" in frame_similarity
    assert "from annolid.infrastructure import AnnotationStore" in frame_similarity

    dino_frame_search = (
        repo_root / "annolid" / "gui" / "dino_frame_search_service.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.infrastructure import AnnotationStore" in dino_frame_search

    video_manager = (
        repo_root / "annolid" / "gui" / "widgets" / "video_manager.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.infrastructure import AnnotationStore" in video_manager
    assert (
        "from annolid.infrastructure.filesystem import find_most_recent_file"
        in video_manager
    )

    label_file = (repo_root / "annolid" / "gui" / "label_file.py").read_text(
        encoding="utf-8"
    )
    assert (
        "from annolid.infrastructure import AnnotationStoreError, load_labelme_json"
        in label_file
    )

    labeling_progress = (
        repo_root / "annolid" / "gui" / "widgets" / "labeling_progress_dashboard.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.infrastructure import load_labelme_json" in labeling_progress

    convert_dlc_dialog = (
        repo_root / "annolid" / "gui" / "widgets" / "convert_deeplabcut_dialog.py"
    ).read_text(encoding="utf-8")
    assert (
        "from annolid.domain import DeepLabCutTrainingImportConfig"
        in convert_dlc_dialog
    )
    assert (
        "from annolid.services.export import import_deeplabcut_dataset"
        in convert_dlc_dialog
    )

    agent_analysis = (
        repo_root / "annolid" / "gui" / "mixins" / "agent_analysis_mixin.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.domain import load_behavior_spec" in agent_analysis

    ai_chat_backend = (
        repo_root / "annolid" / "gui" / "widgets" / "ai_chat_backend.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.services.chat_runtime import (" in ai_chat_backend
    assert "from annolid.services.chat_provider_runtime import (" in ai_chat_backend
    assert "from annolid.services.chat_context import (" in ai_chat_backend
    assert "from annolid.services.chat_agent_core import (" in ai_chat_backend
    assert "from annolid.services.chat_backend_support import (" in ai_chat_backend
    assert "from annolid.services.chat_devtools import (" in ai_chat_backend
    assert "from annolid.services.chat_session import (" in ai_chat_backend
    assert "from annolid.services.chat_widget_bridge import (" in ai_chat_backend
    assert "from annolid.services.chat_video import (" in ai_chat_backend
    assert "from annolid.services.chat_web_pdf import (" in ai_chat_backend
    assert "from annolid.services.chat_controls import (" in ai_chat_backend
    assert "from annolid.services.chat_arxiv import (" in ai_chat_backend
    assert "from annolid.services.chat_citations import (" in ai_chat_backend
    assert "from annolid.services.chat_filesystem import (" in ai_chat_backend
    assert "from annolid.services.chat_realtime import (" in ai_chat_backend
    assert "from annolid.services.chat_shapes import (" in ai_chat_backend
    assert "from annolid.services.chat_shape_files import (" in ai_chat_backend
    assert "from annolid.infrastructure.agent_config import " in ai_chat_backend
    assert "from annolid.infrastructure.agent_workspace import " in ai_chat_backend
    assert "from annolid.core.agent" not in ai_chat_backend
    assert "get_chat_workspace()" in ai_chat_backend
    assert "get_chat_session_store()" in ai_chat_backend
    assert "resolve_chat_pdf_path(raw_path)" in ai_chat_backend
    assert "build_chat_pdf_search_roots()" in ai_chat_backend
    assert "build_chat_vcs_read_roots()" in ai_chat_backend
    assert "get_chat_tutorials_dir()" in ai_chat_backend
    assert "get_chat_realtime_defaults()" in ai_chat_backend
    assert "get_chat_email_defaults()" in ai_chat_backend
    assert 'read_chat_memory_text("MEMORY.md")' in ai_chat_backend
    assert "get_chat_attachment_roots()" in ai_chat_backend
    assert "get_chat_camera_snapshots_dir()" in ai_chat_backend
    assert "is_chat_provider_config_error(exc)" in ai_chat_backend
    assert (
        "format_chat_provider_config_error(raw_error, provider=self.provider)"
        in ai_chat_backend
    )
    assert "has_chat_image_context(self.image_path)" in ai_chat_backend
    assert "chat_provider_dependency_error(" in ai_chat_backend
    assert "format_chat_dependency_error(" in ai_chat_backend
    assert "chat_agent_loop_llm_timeout_seconds(" in ai_chat_backend
    assert "chat_agent_loop_tool_timeout_seconds(" in ai_chat_backend
    assert "chat_browser_first_for_web(self.settings)" in ai_chat_backend
    assert "chat_fast_mode_timeout_seconds(self.settings)" in ai_chat_backend
    assert "log_chat_runtime_timeouts(" in ai_chat_backend
    assert "build_chat_agent_loop(" in ai_chat_backend
    assert "build_chat_gui_context_payload(" in ai_chat_backend
    assert "open_chat_url_tool(" in ai_chat_backend
    assert "open_chat_in_browser_tool(" in ai_chat_backend
    assert "open_chat_pdf_tool(" in ai_chat_backend
    assert "open_chat_video_tool(" in ai_chat_backend
    assert "get_chat_web_dom_text(" in ai_chat_backend
    assert "get_chat_web_state(" in ai_chat_backend
    assert "capture_chat_web_screenshot(" in ai_chat_backend
    assert "describe_chat_web_view(" in ai_chat_backend
    assert "extract_chat_web_structured(" in ai_chat_backend
    assert "click_chat_web(" in ai_chat_backend
    assert "type_chat_web(" in ai_chat_backend
    assert "scroll_chat_web(" in ai_chat_backend
    assert "find_chat_web_forms(" in ai_chat_backend
    assert "run_chat_web_steps(" in ai_chat_backend
    assert "extract_chat_first_web_url(" in ai_chat_backend
    assert "get_chat_pdf_state(" in ai_chat_backend
    assert "get_chat_pdf_text(" in ai_chat_backend
    assert "find_chat_pdf_sections(" in ai_chat_backend
    assert "resolve_chat_video_path_for_gui_tool(" in ai_chat_backend
    assert "resolve_chat_provider_kind(" in ai_chat_backend
    assert "run_chat_fast_mode(" in ai_chat_backend
    assert "run_chat_fast_provider_chat(" in ai_chat_backend
    assert "run_chat_provider_fallback(" in ai_chat_backend
    assert "run_chat_awaitable_sync(" in ai_chat_backend
    assert "run_chat_ollama(" in ai_chat_backend
    assert "run_chat_openai(" in ai_chat_backend
    assert "run_chat_gemini(" in ai_chat_backend
    assert "emit_chat_chunk(" in ai_chat_backend
    assert "emit_chat_progress(" in ai_chat_backend
    assert "emit_chat_final(" in ai_chat_backend
    assert "load_chat_history_messages(" in ai_chat_backend
    assert "persist_chat_turn(" in ai_chat_backend
    assert "invoke_chat_widget_slot(" in ai_chat_backend
    assert "invoke_chat_widget_json_slot(" in ai_chat_backend
    assert "execute_chat_direct_gui_command(" in ai_chat_backend
    assert "get_chat_widget_action_result(" in ai_chat_backend
    assert "segment_track_chat_video_tool(" in ai_chat_backend
    assert "label_chat_behavior_segments_tool(" in ai_chat_backend
    assert "gui_run_ai_text_segmentation_tool(" in ai_chat_backend
    assert "gui_select_annotation_model_tool(" in ai_chat_backend
    assert "gui_send_chat_prompt_tool(" in ai_chat_backend
    assert "gui_set_ai_text_prompt_tool(" in ai_chat_backend
    assert "gui_set_chat_model_tool(" in ai_chat_backend
    assert "gui_set_chat_prompt_tool(" in ai_chat_backend
    assert "gui_set_frame_tool(" in ai_chat_backend
    assert "gui_track_next_frames_tool(" in ai_chat_backend
    assert "gui_arxiv_search_tool(" in ai_chat_backend
    assert "gui_safe_run_arxiv_search(" in ai_chat_backend
    assert "gui_list_local_pdfs(" in ai_chat_backend
    assert "gui_add_citation_raw_tool(" in ai_chat_backend
    assert "gui_list_citations_tool(" in ai_chat_backend
    assert "gui_save_citation_tool(" in ai_chat_backend
    assert "gui_rename_file_tool(" in ai_chat_backend
    assert "gui_check_stream_source_tool(" in ai_chat_backend
    assert "gui_get_realtime_status_tool(" in ai_chat_backend
    assert "gui_list_logs_tool(" in ai_chat_backend
    assert "gui_list_log_files_tool(" in ai_chat_backend
    assert "gui_read_log_file_tool(" in ai_chat_backend
    assert "gui_search_logs_tool(" in ai_chat_backend
    assert "gui_open_log_folder_tool(" in ai_chat_backend
    assert "gui_remove_log_folder_tool(" in ai_chat_backend
    assert "gui_list_realtime_logs_tool(" in ai_chat_backend
    assert "gui_list_realtime_models_tool(" in ai_chat_backend
    assert "gui_start_realtime_stream_tool(" in ai_chat_backend
    assert "gui_stop_realtime_stream_tool(" in ai_chat_backend
    assert "gui_delete_selected_shapes_tool(" in ai_chat_backend
    assert "gui_list_shapes_tool(" in ai_chat_backend
    assert "gui_select_shapes_tool(" in ai_chat_backend
    assert "gui_set_selected_shape_label_tool(" in ai_chat_backend
    assert "gui_delete_shapes_in_annotation_tool(" in ai_chat_backend
    assert "gui_list_shapes_in_annotation_tool(" in ai_chat_backend
    assert "gui_relabel_shapes_in_annotation_tool(" in ai_chat_backend
    assert "chat_list_dir(" in ai_chat_backend
    assert "chat_read_file(" in ai_chat_backend
    assert "chat_exec_command(" in ai_chat_backend
    assert "chat_git_status(" in ai_chat_backend
    assert "chat_git_diff(" in ai_chat_backend
    assert "chat_git_log(" in ai_chat_backend
    assert "chat_github_pr_status(" in ai_chat_backend
    assert "chat_github_pr_checks(" in ai_chat_backend
    assert "chat_exec_start(" in ai_chat_backend
    assert "chat_exec_process(" in ai_chat_backend
    assert "load_chat_execution_prerequisites()" in ai_chat_backend
    assert "prepare_chat_context_tools(" in ai_chat_backend
    assert "register_chat_gui_toolset(" in ai_chat_backend

    ai_chat_widget = (
        repo_root / "annolid" / "gui" / "widgets" / "ai_chat_widget.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.infrastructure.agent_config import (" in ai_chat_widget
    assert "from annolid.infrastructure.agent_workspace import " in ai_chat_widget
    assert "from annolid.services.chat_bus import (" in ai_chat_widget
    assert "from annolid.services.chat_session import (" in ai_chat_widget
    assert "from annolid.services.chat_backend_support import (" in ai_chat_widget
    assert "from annolid.core.agent" not in ai_chat_widget

    ai_chat_manager = (
        repo_root / "annolid" / "gui" / "widgets" / "ai_chat_manager.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.infrastructure.agent_config import " in ai_chat_manager
    assert "from annolid.infrastructure.agent_workspace import " in ai_chat_manager
    assert "from annolid.services.chat_bus import " in ai_chat_manager
    assert "from annolid.services.chat_manager_runtime import (" in ai_chat_manager
    assert "from annolid.core.agent" not in ai_chat_manager

    chat_session_dialog = (
        repo_root / "annolid" / "gui" / "widgets" / "ai_chat_session_dialog.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.services.chat_session import (" in chat_session_dialog
    assert "from annolid.core.agent" not in chat_session_dialog

    llm_settings_dialog = (
        repo_root / "annolid" / "gui" / "widgets" / "llm_settings_dialog.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.infrastructure.agent_config import (" in llm_settings_dialog
    assert "from annolid.services.agent_update import (" in llm_settings_dialog
    assert "from annolid.core.agent" not in llm_settings_dialog

    web_viewer = (
        repo_root / "annolid" / "gui" / "widgets" / "web_viewer.py"
    ).read_text(encoding="utf-8")
    assert "from annolid.infrastructure.agent_workspace import " in web_viewer
    assert "from annolid.core.agent" not in web_viewer
