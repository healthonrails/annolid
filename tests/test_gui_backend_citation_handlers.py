from __future__ import annotations

from pathlib import Path

from annolid.core.agent.gui_backend.tool_handlers_citations import add_citation_raw_tool
from annolid.utils.citations import load_bibtex, parse_bibtex, save_bibtex, upsert_entry


def _resolve_local(path_text: str, workspace: Path) -> Path:
    path = str(path_text or "").strip()
    if not path:
        return workspace / "citations.bib"
    return workspace / path


def test_add_citation_raw_tool_accepts_markdown_fenced_bibtex(tmp_path: Path) -> None:
    test_key = "annolidref"
    payload = add_citation_raw_tool(
        bibtex=(
            "add citation to refs.bib please:\n"
            "```bibtex\n"
            f"@article{{{test_key},\n"
            "  title={PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data},\n"
            "  author={Zhu, Zhe and Wan, Le},\n"
            "  journal={arXiv preprint arXiv:2509.21965},\n"
            "  year={2025}\n"
            "}\n"
            "```"
        ),
        bib_file="refs.bib",
        parse_bibtex=parse_bibtex,
        resolve_bib_path=lambda value: _resolve_local(value, tmp_path),
        load_bibtex=load_bibtex,
        upsert_entry=upsert_entry,
        save_bibtex=lambda path, entries, sort_keys=True: save_bibtex(
            path, entries, sort_keys=sort_keys
        ),
    )
    assert payload["ok"] is True
    assert payload["key"] == test_key
    assert payload["entry_count"] == 1
    bib_path = tmp_path / "refs.bib"
    entries = load_bibtex(bib_path)
    assert len(entries) == 1
    assert entries[0].key == test_key
    assert entries[0].fields.get("year") == "2025"


def test_add_citation_raw_tool_upserts_multiple_entries(tmp_path: Path) -> None:
    payload = add_citation_raw_tool(
        bibtex=(
            "@article{a2024,title={A},year={2024}}\n\n"
            "@article{b2025,title={B},year={2025}}"
        ),
        bib_file="refs.bib",
        parse_bibtex=parse_bibtex,
        resolve_bib_path=lambda value: _resolve_local(value, tmp_path),
        load_bibtex=load_bibtex,
        upsert_entry=upsert_entry,
        save_bibtex=lambda path, entries, sort_keys=True: save_bibtex(
            path, entries, sort_keys=sort_keys
        ),
    )
    assert payload["ok"] is True
    assert payload["entry_count"] == 2
    assert payload["created_count"] == 2
    assert set(payload["keys"]) == {"a2024", "b2025"}
    entries = load_bibtex(tmp_path / "refs.bib")
    assert len(entries) == 2
