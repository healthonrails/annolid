from __future__ import annotations

import json
from pathlib import Path

from annolid.engine.cli import main as annolid_run


def _read_json_line(capsys) -> dict:
    out = capsys.readouterr().out.strip()
    assert out
    return json.loads(out)


def test_citations_upsert_list_and_remove(tmp_path: Path, capsys) -> None:
    bib_path = tmp_path / "refs.bib"

    rc = annolid_run(
        [
            "citations-upsert",
            "--bib-file",
            str(bib_path),
            "--key",
            "annolid2024",
            "--entry-type",
            "article",
            "--title",
            "Annolid Toolkit",
            "--author",
            "Liu, Jun",
            "--year",
            "2024",
            "--doi",
            "10.1000/annolid",
        ]
    )
    assert rc == 0
    upsert_payload = _read_json_line(capsys)
    assert upsert_payload["created"] is True

    rc = annolid_run(
        [
            "citations-list",
            "--bib-file",
            str(bib_path),
            "--query",
            "toolkit",
            "--limit",
            "5",
        ]
    )
    assert rc == 0
    list_payload = _read_json_line(capsys)
    assert list_payload["returned"] == 1
    assert list_payload["entries"][0]["key"] == "annolid2024"

    rc = annolid_run(
        [
            "citations-remove",
            "--bib-file",
            str(bib_path),
            "--key",
            "annolid2024",
        ]
    )
    assert rc == 0
    remove_payload = _read_json_line(capsys)
    assert remove_payload["removed"] is True


def test_citations_format_rewrites_file(tmp_path: Path, capsys) -> None:
    bib_path = tmp_path / "refs.bib"
    bib_path.write_text(
        "@article{zkey,title={z}}\n\n@article{akey,title={a}}\n", encoding="utf-8"
    )
    rc = annolid_run(["citations-format", "--bib-file", str(bib_path)])
    assert rc == 0
    payload = _read_json_line(capsys)
    assert payload["entries"] == 2
    text = bib_path.read_text(encoding="utf-8")
    assert text.find("@article{akey") < text.find("@article{zkey")
