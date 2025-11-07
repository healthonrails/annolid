"""
Prompt builders and helper utilities for Qwen-style behavior descriptions.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence
from pathlib import Path
import textwrap

_PROMPT_DIR = Path.home() / ".annolid"
_PROMPT_FILE = _PROMPT_DIR / "behavior_prompt_template.txt"


DEFAULT_BEHAVIOR_PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are an animal behavior observer.
    {segment_sentence}
    {view_guidance}
    {roi_guidance}
    Report only observable facts about the mouse—pose, locomotion, grooming, rearing, sniffing, tail posture, limb contact.
    Mention whether it is stationary or moving and describe the pace qualitatively (slow, moderate, rapid) if motion is visible.
    Note head orientation, activities (sniffing, grooming), and what each set of paws is doing relative to the ground.
    Keep the description focused on the animal; do not mention the arena, cameras, or lighting.
    {facts_guidance}
    {length_guidance}
    """
).strip()


def _prompt_file_path() -> Path:
    return _PROMPT_FILE


def load_user_behavior_prompt_template() -> Optional[str]:
    path = _prompt_file_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    stripped = text.strip()
    return stripped or None


def save_user_behavior_prompt_template(text: str) -> None:
    path = _prompt_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def reset_user_behavior_prompt_template() -> None:
    path = _prompt_file_path()
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def get_effective_behavior_prompt_template() -> str:
    return load_user_behavior_prompt_template() or DEFAULT_BEHAVIOR_PROMPT_TEMPLATE

def build_mouse_behavior_json_prompt(
    *,
    use_velocity_qualitative: bool = True,
    multi_view: bool = False,
    include_roi_notes: bool = False,
) -> str:
    """
    Construct a strict instruction prompt for producing behavior JSON.
    """

    qualitative_velocity = (
        "Refer to any supplied velocity only qualitatively (very low, low, moderate, high)"
        " and never repeat numeric values."
        if use_velocity_qualitative
        else "Use the supplied information as-is without rewriting numeric values."
    )

    view_language = (
        "You receive synchronized multi-view images of the same moment."
        " Fuse observations across all views before describing the pose or motion."
        if multi_view
        else "Base the description on the provided image."
    )

    roi_language = (
        "Additional crops may focus on head, forelimbs, hind limbs, or tail;"
        " use them to refine contact and orientation details."
        if include_roi_notes
        else ""
    )

    prompt_lines: List[str] = [
        "You analyze frames of a single mouse inside a behavior arena.",
        view_language,
        roi_language,
        qualitative_velocity,
        "Describe only the animal. Avoid references to the arena, cameras, or lighting.",
        "State observable facts. If a body part is obscured or blurred, say it is 'unclear due to occlusion/blur'.",
        "Do not speculate about motivations, emotions, or future behavior.",
        "Respond with a valid JSON object only. Do not add prose outside the JSON.",
        "Required keys: Overall, Head, Limb, Torso, Others, Keywords.",
        "Use complete sentences for each key except Keywords.",
        "Keywords must be a single comma-separated string summarizing the observations.",
        "For motion, specify if the mouse is stationary or moving, and describe pace qualitatively.",
        "Head: orientation relative to body or ground, visible activities (sniffing, grooming, looking).",
        "Limb: note contact or lift for forepaws and hind paws, tail posture.",
        "Torso: outline spine curvature and whether chest/abdomen/hip contact the ground.",
        "Others: any additional factual behavior details (e.g., whisker position, body elongation).",
        "Do not mention missing speed direction; velocity is magnitude only.",
        "Never introduce new keys or numerical speed values.",
    ]

    prompt = "\n".join(line for line in prompt_lines if line)
    prompt += (
        "\nReturn JSON with the fields in the order: Overall, Head, Limb, Torso, Others, Keywords."
    )
    return prompt


def build_behavior_narrative_prompt(
    *,
    segment_label: Optional[str] = None,
    multi_view: bool = False,
    include_roi_notes: bool = False,
) -> str:
    """
    Prompt for a prose-style behavior description saved as caption text.
    """

    view_text = (
        "You receive synchronized multi-view images of the same timepoint; fuse the evidence."
        if multi_view
        else "Base your description on the provided image."
    )
    roi_text = (
        "Cropped views may highlight head, paws, or tail. Use them to refine details."
        if include_roi_notes
        else ""
    )
    segment_text = (
        f"Focus on the segment {segment_label}. Describe what the mouse is doing during that interval."
        if segment_label
        else "Describe what the mouse is doing at this moment."
    )

    lines = [
        view_text,
        roi_text,
        segment_text,
    ]
    facts_guidance = "Avoid referencing the arena, camera, or environment. Do not speculate about intent or emotion."
    length_guidance = "Write 2–4 sentences that are concise but detailed."
    context = {
        "segment_sentence": segment_text,
        "segment_label": segment_label or "this moment",
        "view_guidance": view_text,
        "roi_guidance": roi_text,
        "facts_guidance": facts_guidance,
        "length_guidance": length_guidance,
    }
    template = get_effective_behavior_prompt_template()
    try:
        rendered = template.format_map(context)
    except KeyError:
        rendered = template
    return "\n".join(
        line for line in rendered.splitlines() if line.strip()
    )


def qwen_messages(images: Sequence[str], text: str) -> List[dict]:
    """
    Build a single-message payload compatible with Qwen/Ollama chat APIs.
    """

    image_list = [img for img in images if isinstance(img, str) and img]
    message: dict = {"role": "user", "content": text}
    if image_list:
        message["images"] = list(image_list)
    return [message]
