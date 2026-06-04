from __future__ import annotations

import pytest

from annolid.segmentation.SAM.sam3.agent_video_orchestrator import AgentConfig
from annolid.segmentation.SAM.sam3.prompt_builder import (
    normalize_text_prompt,
    require_text_prompt,
    split_text_prompts,
)
from annolid.segmentation.SAM.sam3.session import Sam3SessionManager


def test_split_text_prompts_strips_empty_values_and_deduplicates() -> None:
    assert split_text_prompts(" mouse, pup. mouse;\nvole ") == [
        "mouse",
        "pup",
        "vole",
    ]


def test_normalize_text_prompt_returns_canonical_prompt_string() -> None:
    assert normalize_text_prompt(" mouse, pup. mouse ") == "mouse, pup"
    assert normalize_text_prompt(" , . \n ") is None


def test_require_text_prompt_raises_actionable_error_for_empty_prompt() -> None:
    with pytest.raises(
        ValueError,
        match="SAM3 agent video tracking requires a text prompt",
    ):
        require_text_prompt("  ", context="SAM3 agent video tracking")


def test_agent_config_normalizes_prompt_for_direct_callers() -> None:
    cfg = AgentConfig(prompt=" mouse, pup. mouse ")
    assert cfg.prompt == "mouse, pup"


def test_prompt_transaction_normalizes_direct_session_text_prompt() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    steps = session._build_prompt_transaction_steps(
        text=" mouse. pup, mouse ",
        boxes=None,
        box_labels=None,
        mask_inputs=None,
        mask_labels=None,
        points=None,
        point_labels=None,
        obj_id=None,
    )

    assert steps == [{"kind": "semantic", "text": "mouse, pup"}]
