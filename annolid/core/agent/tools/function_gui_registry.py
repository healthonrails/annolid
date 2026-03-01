from __future__ import annotations

from typing import Optional

from .function_registry import FunctionToolRegistry
from .function_gui_base import ActionCallback, ContextCallback, PathCallback

from .function_gui_core import (
    GuiContextTool,
    GuiGenerateAnnolidTutorialTool,
    GuiSharedImagePathTool,
    GuiSetPromptTool,
    GuiSendPromptTool,
    GuiSetChatModelTool,
    GuiSelectAnnotationModelTool,
    GuiSaveCitationTool,
)
from .function_gui_web import (
    GuiOpenUrlTool,
    GuiOpenInBrowserTool,
    GuiWebGetDomTextTool,
    GuiWebClickTool,
    GuiWebTypeTool,
    GuiWebScrollTool,
    GuiWebFindFormsTool,
    GuiWebRunStepsTool,
)
from .function_gui_threejs import (
    GuiOpenThreeJsExampleTool,
    GuiOpenThreeJsTool,
)
from .function_gui_video import (
    GuiCheckStreamSourceTool,
    GuiGetRealtimeStatusTool,
    GuiListLogsTool,
    GuiOpenLogFolderTool,
    GuiRemoveLogFolderTool,
    GuiListLogFilesTool,
    GuiReadLogFileTool,
    GuiSearchLogsTool,
    GuiListRealtimeLogsTool,
    GuiListRealtimeModelsTool,
    GuiOpenVideoTool,
    GuiSetFrameTool,
    GuiTrackNextFramesTool,
    GuiSetAiTextPromptTool,
    GuiRunAiTextSegmentationTool,
    GuiSegmentTrackVideoTool,
    GuiLabelBehaviorSegmentsTool,
    GuiStartRealtimeStreamTool,
    GuiStopRealtimeStreamTool,
)
from .function_gui_pdf import (
    GuiOpenPdfTool,
    GuiPdfGetStateTool,
    GuiPdfGetTextTool,
    GuiPdfFindSectionsTool,
    GuiArxivSearchTool,
    GuiListPdfsTool,
)
from .function_gui_shapes import (
    GuiDeleteShapesInAnnotationTool,
    GuiDeleteSelectedShapesTool,
    GuiListShapesInAnnotationTool,
    GuiListShapesTool,
    GuiRelabelShapesInAnnotationTool,
    GuiSelectShapesTool,
    GuiSetSelectedShapeLabelTool,
)


def register_annolid_gui_tools(
    registry: FunctionToolRegistry,
    *,
    context_callback: Optional[ContextCallback] = None,
    image_path_callback: Optional[PathCallback] = None,
    open_video_callback: Optional[ActionCallback] = None,
    open_url_callback: Optional[ActionCallback] = None,
    open_in_browser_callback: Optional[ActionCallback] = None,
    open_threejs_callback: Optional[ActionCallback] = None,
    open_threejs_example_callback: Optional[ActionCallback] = None,
    web_get_dom_text_callback: Optional[ActionCallback] = None,
    web_click_callback: Optional[ActionCallback] = None,
    web_type_callback: Optional[ActionCallback] = None,
    web_scroll_callback: Optional[ActionCallback] = None,
    web_find_forms_callback: Optional[ActionCallback] = None,
    web_run_steps_callback: Optional[ActionCallback] = None,
    open_pdf_callback: Optional[ActionCallback] = None,
    pdf_get_state_callback: Optional[ActionCallback] = None,
    pdf_get_text_callback: Optional[ActionCallback] = None,
    pdf_find_sections_callback: Optional[ActionCallback] = None,
    arxiv_search_callback: Optional[ActionCallback] = None,
    list_pdfs_callback: Optional[ActionCallback] = None,
    save_citation_callback: Optional[ActionCallback] = None,
    generate_annolid_tutorial_callback: Optional[ActionCallback] = None,
    set_frame_callback: Optional[ActionCallback] = None,
    set_prompt_callback: Optional[ActionCallback] = None,
    send_prompt_callback: Optional[ActionCallback] = None,
    set_chat_model_callback: Optional[ActionCallback] = None,
    select_annotation_model_callback: Optional[ActionCallback] = None,
    track_next_frames_callback: Optional[ActionCallback] = None,
    set_ai_text_prompt_callback: Optional[ActionCallback] = None,
    run_ai_text_segmentation_callback: Optional[ActionCallback] = None,
    segment_track_video_callback: Optional[ActionCallback] = None,
    label_behavior_segments_callback: Optional[ActionCallback] = None,
    start_realtime_stream_callback: Optional[ActionCallback] = None,
    stop_realtime_stream_callback: Optional[ActionCallback] = None,
    get_realtime_status_callback: Optional[ActionCallback] = None,
    list_realtime_models_callback: Optional[ActionCallback] = None,
    list_realtime_logs_callback: Optional[ActionCallback] = None,
    list_logs_callback: Optional[ActionCallback] = None,
    open_log_folder_callback: Optional[ActionCallback] = None,
    remove_log_folder_callback: Optional[ActionCallback] = None,
    list_log_files_callback: Optional[ActionCallback] = None,
    read_log_file_callback: Optional[ActionCallback] = None,
    search_logs_callback: Optional[ActionCallback] = None,
    check_stream_source_callback: Optional[ActionCallback] = None,
    list_shapes_callback: Optional[ActionCallback] = None,
    select_shapes_callback: Optional[ActionCallback] = None,
    set_selected_shape_label_callback: Optional[ActionCallback] = None,
    delete_selected_shapes_callback: Optional[ActionCallback] = None,
    list_shapes_in_annotation_callback: Optional[ActionCallback] = None,
    relabel_shapes_in_annotation_callback: Optional[ActionCallback] = None,
    delete_shapes_in_annotation_callback: Optional[ActionCallback] = None,
) -> None:
    """Register GUI-only tools for Annolid Bot sessions."""
    registry.register(GuiContextTool(context_callback=context_callback))
    registry.register(GuiSharedImagePathTool(image_path_callback=image_path_callback))
    registry.register(GuiOpenVideoTool(open_video_callback=open_video_callback))
    registry.register(GuiOpenUrlTool(open_url_callback=open_url_callback))
    registry.register(
        GuiOpenInBrowserTool(open_in_browser_callback=open_in_browser_callback)
    )
    registry.register(GuiOpenThreeJsTool(open_threejs_callback=open_threejs_callback))
    registry.register(
        GuiOpenThreeJsExampleTool(
            open_threejs_example_callback=open_threejs_example_callback
        )
    )
    registry.register(
        GuiWebGetDomTextTool(web_get_dom_text_callback=web_get_dom_text_callback)
    )
    registry.register(GuiWebClickTool(web_click_callback=web_click_callback))
    registry.register(GuiWebTypeTool(web_type_callback=web_type_callback))
    registry.register(GuiWebScrollTool(web_scroll_callback=web_scroll_callback))
    registry.register(
        GuiWebFindFormsTool(web_find_forms_callback=web_find_forms_callback)
    )
    registry.register(GuiWebRunStepsTool(web_run_steps_callback=web_run_steps_callback))
    registry.register(GuiOpenPdfTool(open_pdf_callback=open_pdf_callback))
    registry.register(GuiPdfGetStateTool(pdf_get_state_callback=pdf_get_state_callback))
    registry.register(GuiPdfGetTextTool(pdf_get_text_callback=pdf_get_text_callback))
    registry.register(
        GuiPdfFindSectionsTool(pdf_find_sections_callback=pdf_find_sections_callback)
    )
    registry.register(GuiArxivSearchTool(arxiv_search_callback=arxiv_search_callback))
    registry.register(GuiListPdfsTool(list_pdfs_callback=list_pdfs_callback))
    registry.register(
        GuiSaveCitationTool(save_citation_callback=save_citation_callback)
    )
    registry.register(
        GuiGenerateAnnolidTutorialTool(
            generate_tutorial_callback=generate_annolid_tutorial_callback
        )
    )
    registry.register(GuiSetFrameTool(set_frame_callback=set_frame_callback))
    registry.register(GuiSetPromptTool(set_prompt_callback=set_prompt_callback))
    registry.register(GuiSendPromptTool(send_prompt_callback=send_prompt_callback))
    registry.register(GuiSetChatModelTool(set_model_callback=set_chat_model_callback))
    registry.register(
        GuiSelectAnnotationModelTool(
            select_model_callback=select_annotation_model_callback
        )
    )
    registry.register(GuiTrackNextFramesTool(track_callback=track_next_frames_callback))
    registry.register(
        GuiSetAiTextPromptTool(set_ai_text_prompt_callback=set_ai_text_prompt_callback)
    )
    registry.register(
        GuiRunAiTextSegmentationTool(
            run_ai_text_segmentation_callback=run_ai_text_segmentation_callback
        )
    )
    registry.register(
        GuiSegmentTrackVideoTool(
            segment_track_video_callback=segment_track_video_callback
        )
    )
    registry.register(
        GuiLabelBehaviorSegmentsTool(
            label_behavior_segments_callback=label_behavior_segments_callback
        )
    )
    registry.register(
        GuiStartRealtimeStreamTool(
            start_realtime_stream_callback=start_realtime_stream_callback
        )
    )
    registry.register(
        GuiStopRealtimeStreamTool(
            stop_realtime_stream_callback=stop_realtime_stream_callback
        )
    )
    registry.register(
        GuiGetRealtimeStatusTool(
            get_realtime_status_callback=get_realtime_status_callback
        )
    )
    registry.register(
        GuiListRealtimeModelsTool(
            list_realtime_models_callback=list_realtime_models_callback
        )
    )
    registry.register(
        GuiListRealtimeLogsTool(list_realtime_logs_callback=list_realtime_logs_callback)
    )
    registry.register(GuiListLogsTool(list_logs_callback=list_logs_callback))
    registry.register(
        GuiOpenLogFolderTool(open_log_folder_callback=open_log_folder_callback)
    )
    registry.register(
        GuiRemoveLogFolderTool(remove_log_folder_callback=remove_log_folder_callback)
    )
    registry.register(
        GuiListLogFilesTool(list_log_files_callback=list_log_files_callback)
    )
    registry.register(GuiReadLogFileTool(read_log_file_callback=read_log_file_callback))
    registry.register(GuiSearchLogsTool(search_logs_callback=search_logs_callback))
    registry.register(
        GuiCheckStreamSourceTool(
            check_stream_source_callback=check_stream_source_callback
        )
    )
    registry.register(GuiListShapesTool(list_shapes_callback=list_shapes_callback))
    registry.register(
        GuiSelectShapesTool(select_shapes_callback=select_shapes_callback)
    )
    registry.register(
        GuiSetSelectedShapeLabelTool(
            set_selected_shape_label_callback=set_selected_shape_label_callback
        )
    )
    registry.register(
        GuiDeleteSelectedShapesTool(
            delete_selected_shapes_callback=delete_selected_shapes_callback
        )
    )
    registry.register(
        GuiListShapesInAnnotationTool(
            list_shapes_in_annotation_callback=list_shapes_in_annotation_callback
        )
    )
    registry.register(
        GuiRelabelShapesInAnnotationTool(
            relabel_shapes_in_annotation_callback=relabel_shapes_in_annotation_callback
        )
    )
    registry.register(
        GuiDeleteShapesInAnnotationTool(
            delete_shapes_in_annotation_callback=delete_shapes_in_annotation_callback
        )
    )
