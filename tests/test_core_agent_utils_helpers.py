from __future__ import annotations

from pathlib import Path

import pytest

from annolid.core.agent import utils as u
from annolid.core.agent.utils import helpers


def test_workspace_memory_skills_paths(tmp_path: Path) -> None:
    ws = u.get_agent_workspace_path(str(tmp_path / "ws"))
    assert ws.exists() and ws.is_dir()

    memory = u.get_memory_path(ws)
    skills = u.get_skills_path(ws)
    assert memory.exists() and memory.is_dir()
    assert skills.exists() and skills.is_dir()


def test_sessions_path_uses_data_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(helpers, "get_agent_data_path", lambda: tmp_path / "data")
    sessions = u.get_sessions_path()
    assert sessions == (tmp_path / "data" / "sessions")
    assert sessions.exists()


def test_safe_filename_truncate_and_parse_session_key() -> None:
    assert u.safe_filename('a<b>c:d/e\\f|g?*"') == "a_b_c_d_e_f_g___"
    assert u.truncate_string("abcdef", 4) == "a..."
    assert u.truncate_string("abc", 10) == "abc"
    assert u.parse_session_key("telegram:123") == ("telegram", "123")
    with pytest.raises(ValueError):
        u.parse_session_key("badkey")
