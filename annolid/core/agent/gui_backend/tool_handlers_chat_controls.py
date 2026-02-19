from __future__ import annotations

from typing import Callable, Dict


def set_frame_tool(
    frame_index: int,
    *,
    invoke_set_frame: Callable[[int], bool],
) -> Dict[str, object]:
    target_frame = int(frame_index)
    if target_frame < 0:
        return {"ok": False, "error": "frame_index must be >= 0"}
    if not invoke_set_frame(target_frame):
        return {"ok": False, "error": "Failed to queue frame action"}
    return {"ok": True, "queued": True, "frame_index": target_frame}


def set_chat_prompt_tool(
    text: str,
    *,
    invoke_set_chat_prompt: Callable[[str], bool],
) -> Dict[str, object]:
    prompt_text = str(text or "").strip()
    if not prompt_text:
        return {"ok": False, "error": "text is required"}
    if not invoke_set_chat_prompt(prompt_text):
        return {"ok": False, "error": "Failed to queue prompt update"}
    return {"ok": True, "queued": True, "chars": len(prompt_text)}


def send_chat_prompt_tool(
    *,
    invoke_send_chat_prompt: Callable[[], bool],
) -> Dict[str, object]:
    if not invoke_send_chat_prompt():
        return {"ok": False, "error": "Failed to queue chat send action"}
    return {"ok": True, "queued": True}


def set_chat_model_tool(
    provider: str,
    model: str,
    *,
    invoke_set_chat_model: Callable[[str, str], bool],
) -> Dict[str, object]:
    provider_text = str(provider or "").strip().lower()
    model_text = str(model or "").strip()
    if not provider_text:
        return {"ok": False, "error": "provider is required"}
    if not model_text:
        return {"ok": False, "error": "model is required"}
    if not invoke_set_chat_model(provider_text, model_text):
        return {"ok": False, "error": "Failed to queue provider/model update"}
    return {
        "ok": True,
        "queued": True,
        "provider": provider_text,
        "model": model_text,
    }


def select_annotation_model_tool(
    model_name: str,
    *,
    invoke_select_annotation_model: Callable[[str], bool],
) -> Dict[str, object]:
    model_text = str(model_name or "").strip()
    if not model_text:
        return {"ok": False, "error": "model_name is required"}
    if not invoke_select_annotation_model(model_text):
        return {"ok": False, "error": "Failed to queue model selection"}
    return {"ok": True, "queued": True, "model_name": model_text}


def track_next_frames_tool(
    to_frame: int,
    *,
    invoke_track_next_frames: Callable[[int], bool],
) -> Dict[str, object]:
    frame = int(to_frame)
    if frame < 1:
        return {"ok": False, "error": "to_frame must be >= 1"}
    if not invoke_track_next_frames(frame):
        return {"ok": False, "error": "Failed to queue tracking action"}
    return {"ok": True, "queued": True, "to_frame": frame}


def set_ai_text_prompt_tool(
    text: str,
    *,
    use_countgd: bool,
    invoke_set_ai_text_prompt: Callable[[str, bool], bool],
) -> Dict[str, object]:
    prompt_text = str(text or "").strip()
    if not prompt_text:
        return {"ok": False, "error": "text is required"}
    flag = bool(use_countgd)
    if not invoke_set_ai_text_prompt(prompt_text, flag):
        return {"ok": False, "error": "Failed to queue AI prompt update"}
    return {
        "ok": True,
        "queued": True,
        "text": prompt_text,
        "use_countgd": flag,
    }


def run_ai_text_segmentation_tool(
    *,
    invoke_run_ai_text_segmentation: Callable[[], bool],
) -> Dict[str, object]:
    if not invoke_run_ai_text_segmentation():
        return {"ok": False, "error": "Failed to queue AI text segmentation action"}
    return {"ok": True, "queued": True}
