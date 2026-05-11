from __future__ import annotations

from functools import lru_cache
from pathlib import Path

BEHAVIOR_LABELING_SKILL_NAME = "behavior-vlm-labeling"
_SKILL_PATH = (
    Path(__file__).resolve().parents[1]
    / "core"
    / "agent"
    / "skills"
    / BEHAVIOR_LABELING_SKILL_NAME
    / "SKILL.md"
)
_PROMPT_SECTION = "Model Prompt Contract"


@lru_cache(maxsize=1)
def load_behavior_labeling_skill_text() -> str:
    """Return the built-in behavior VLM labeling skill text when available."""
    try:
        return _SKILL_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


def behavior_labeling_prompt_guidance() -> str:
    """Extract the skill section that is safe to inject into VLM prompts."""
    text = load_behavior_labeling_skill_text()
    if not text:
        return ""
    heading = f"## {_PROMPT_SECTION}"
    start = text.find(heading)
    if start < 0:
        return ""
    section = text[start + len(heading) :].strip()
    next_heading = section.find("\n## ")
    if next_heading >= 0:
        section = section[:next_heading].strip()
    lines = [line.rstrip() for line in section.splitlines()]
    compact = "\n".join(line for line in lines if line.strip())
    return compact.strip()
