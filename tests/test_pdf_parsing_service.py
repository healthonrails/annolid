from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

from annolid.core.agent.gui_backend.pdf_summary import (
    build_llm_summary_source_candidates,
    clean_pdf_text_for_summary,
    summarize_active_pdf_with_cache,
)
from annolid.services.pdf_parsing import (
    dehyphenate_pdf_text,
    detect_document_language,
    extract_pdf_outline,
    extract_pdf_pages,
    extract_pdf_sections,
    extract_pdf_text,
    extract_pdf_text_sample,
    extract_pdf_title,
    is_section_heading_candidate,
    normalize_pdf_text,
    pdf_needs_repair_hint,
    repair_pdf,
)


def _create_sample_pdf(path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text(
        (72, 72),
        "Behavior Analysis in Rodents\n"
        "Abstract\n"
        "Our methods and results demonstrate synergis-\n"
        "tic tracking across trials.\n"
        "1. Introduction\n"
        "10 mice were tested in the open field arena.\n"
        "I have observed grooming bouts.",
    )
    page2 = doc.new_page()
    page2.insert_text(
        (72, 72),
        "2.1 Behavior Tracking\n"
        "We evaluated pose accuracy on benchmark videos.\n"
        "Conclusion\n"
        "The pipeline scales reliably.",
    )
    doc.set_metadata({"title": "Rodent Behavior Tracking Study"})
    doc.set_toc(
        [
            [1, "Abstract", 1],
            [1, "1. Introduction", 1],
            [2, "2.1 Behavior Tracking", 2],
            [1, "Conclusion", 2],
        ]
    )
    doc.save(str(path))
    doc.close()


def test_pdf_needs_repair_hint_handles_header_offset_and_eof(tmp_path: Path) -> None:
    valid = tmp_path / "valid.pdf"
    valid.write_bytes(b"%PDF-1.7\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
    assert not pdf_needs_repair_hint(valid)

    # PDF spec allows leading preamble bytes before %PDF- within first 1024 bytes.
    preamble = tmp_path / "preamble.pdf"
    preamble.write_bytes(b"PREAMBLE" * 4 + b"%PDF-1.7\nbody\n%%EOF\n")
    assert not pdf_needs_repair_hint(preamble)

    missing_eof = tmp_path / "truncated.pdf"
    missing_eof.write_bytes(b"%PDF-1.7\n1 0 obj\n")
    assert pdf_needs_repair_hint(missing_eof)

    missing_header = tmp_path / "bad.pdf"
    missing_header.write_bytes(b"NOT_A_PDF\n%%EOF\n")
    assert pdf_needs_repair_hint(missing_header)


def test_repair_pdf_falls_back_to_pymupdf_atomically(
    tmp_path: Path, monkeypatch
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _create_sample_pdf(pdf_path)

    # Simulate missing pypdf so PyMuPDF fallback repair runs.
    monkeypatch.setitem(sys.modules, "pypdf", None)
    repaired = repair_pdf(pdf_path, reason="unit-test", out_dir=tmp_path / "repairs")
    assert repaired is not None
    assert repaired.exists()
    assert not pdf_needs_repair_hint(repaired)
    assert list((tmp_path / "repairs").glob("*.tmp")) == []


def test_dehyphenate_and_normalize_pdf_text() -> None:
    raw = "synergis-\ntic model\u00ad with non-linear\nPy-\nTorch\n\n中文\n换行"
    assert "synergistic" in dehyphenate_pdf_text(raw)
    assert "\u00ad" not in dehyphenate_pdf_text(raw)
    assert "Py-\nTorch" in dehyphenate_pdf_text(raw)

    flat = normalize_pdf_text(raw, preserve_paragraphs=False)
    assert "synergistic model with non-linear" in flat
    assert "中文换行" in flat


def test_extract_pdf_text_sections_outline_and_title_with_real_pdf(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _create_sample_pdf(pdf_path)

    text, pages_read, total_pages, backend = extract_pdf_text(
        pdf_path, start_index=0, pages_limit=2, dehyphenate=True
    )
    assert backend == "pymupdf"
    assert pages_read == 2
    assert total_pages == 2
    assert "synergistic tracking" in text

    assert extract_pdf_title(pdf_path) == "Rodent Behavior Tracking Study"

    outline = extract_pdf_outline(pdf_path)
    assert [item["title"] for item in outline] == [
        "Abstract",
        "1. Introduction",
        "2.1 Behavior Tracking",
        "Conclusion",
    ]
    assert outline[2]["page"] == 1  # 0-based page index

    sections_payload = extract_pdf_sections(pdf_path, max_sections=10, max_pages=5)
    assert sections_payload["ok"] is True
    titles = [entry["title"] for entry in sections_payload["sections"]]
    assert "1. Introduction" in titles
    assert "2.1 Behavior Tracking" in titles
    assert not any("10 mice were tested" in t for t in titles)
    assert not any("I have observed" in t for t in titles)

    sample = extract_pdf_text_sample(pdf_path)
    assert "Rodent Behavior Tracking Study" in sample
    assert detect_document_language(sample) == "en-us"


def test_section_heading_candidate_rejects_prose_false_positives() -> None:
    assert is_section_heading_candidate("Abstract")
    assert is_section_heading_candidate("1. Introduction")
    assert is_section_heading_candidate("2.3 Experimental Setup")
    assert is_section_heading_candidate("II. Methods")
    assert not is_section_heading_candidate(
        "10 mice were tested in the open field arena."
    )
    assert not is_section_heading_candidate(
        "I have observed strong behavioral differences across cohorts."
    )


def test_extract_pdf_text_and_pages_fallback_to_pypdf(
    tmp_path: Path, monkeypatch
) -> None:
    pdf_path = tmp_path / "mock.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n%%EOF\n")

    class _FakePypdfPage:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _FakePdfReader:
        def __init__(self, _path: str) -> None:
            self.pages = [
                _FakePypdfPage("Abstract\nCompu-\ntational ethology."),
                _FakePypdfPage("Results\nHigh accuracy."),
            ]
            self.metadata = types.SimpleNamespace(title="Fallback Title")

    fake_pypdf = types.SimpleNamespace(PdfReader=_FakePdfReader)
    monkeypatch.setitem(sys.modules, "fitz", None)
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    text, pages_read, total_pages, backend = extract_pdf_text(
        pdf_path, start_index=0, pages_limit=2, dehyphenate=True
    )
    assert backend == "pypdf"
    assert pages_read == 2
    assert total_pages == 2
    assert "Computational ethology." in text

    pages, scanned, backend_pages = extract_pdf_pages(pdf_path, max_pages=2)
    assert backend_pages == "pypdf"
    assert scanned == 2
    assert pages[0][0] == 1
    assert "Computational ethology." in pages[0][1]

    assert extract_pdf_title(pdf_path) == "Fallback Title"

    summary = summarize_active_pdf_with_cache(
        workspace=tmp_path,
        get_pdf_state=lambda: {"ok": True, "has_pdf": True, "path": str(pdf_path)},
        build_summary=lambda src, **_: f"ok:{len(src)}",
    )
    assert "Summary of the open PDF (mock.pdf):" in summary


def test_clean_pdf_text_and_section_slices_do_not_truncate_on_inline_words() -> None:
    paper_without_abstract = (
        "## Page 1\n"
        "A Novel Tracker\n"
        "Jane Doe, Department of Biology, Some University\n"
        "jane@example.edu\n"
        "We introduce a fast multi-animal tracker.\n"
        "## Page 2\n"
        "In this section we discuss an abstract representation of trajectories.\n"
    )
    cleaned_no_abs = clean_pdf_text_for_summary(paper_without_abstract)
    assert "We introduce a fast multi-animal tracker." in cleaned_no_abs
    assert "jane@example.edu" not in cleaned_no_abs

    paper_with_abstract = (
        "Title Line\n"
        "Abstract\n"
        "Our methods and results demonstrate that this introduction to pose\n"
        "tracking improves accuracy by 15% across diverse arenas.\n"
        "1. Introduction\n"
        "Animal behavior quantification is essential for neuroscience.\n"
        "2. Methods\n"
        "Videos were recorded at 30 fps.\n"
        "3. Conclusion\n"
        "The system generalizes without finetuning.\n"
        "References\n"
        "[1] Author 2024.\n"
    )
    candidates = build_llm_summary_source_candidates(paper_with_abstract)
    by_label = {item["label"]: item["text"] for item in candidates}
    assert "abstract" in by_label
    assert "improves accuracy by 15% across diverse arenas." in by_label["abstract"]
    assert "Animal behavior quantification" not in by_label["abstract"]
    assert "abstract_intro_conclusion" in by_label
    assert "generalizes without finetuning." in by_label["abstract_intro_conclusion"]
    assert "[1] Author 2024." not in by_label["abstract_intro_conclusion"]


def test_detect_document_language_variants() -> None:
    assert (
        detect_document_language(
            "This study analyzes mouse social interaction across multiple behavioral sessions."
        )
        == "en-us"
    )
    assert (
        detect_document_language(
            "本研究探讨了多只小鼠在开放场地中的社交行为轨迹分析方法。"
        )
        == "zh"
    )
    assert (
        detect_document_language("これはマウスの行動解析に関する研究論文の要旨です。")
        == "ja"
    )


def test_encrypted_pdf_with_empty_user_password_extracts_and_repairs_via_pymupdf(
    tmp_path: Path, monkeypatch
) -> None:
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "restricted.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Abstract\nProtected journal article text with syner-\ngistic tracking.",
    )
    doc.set_metadata({"title": "Restricted Journal Paper"})
    doc.set_toc([[1, "Abstract", 1]])
    doc.save(
        str(pdf_path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="",
    )
    doc.close()

    text, pages_read, total_pages, backend = extract_pdf_text(
        pdf_path, dehyphenate=True
    )
    assert backend == "pymupdf"
    assert pages_read == 1
    assert total_pages == 1
    assert "synergistic tracking." in text
    assert extract_pdf_title(pdf_path) == "Restricted Journal Paper"
    assert [entry["title"] for entry in extract_pdf_outline(pdf_path)] == ["Abstract"]
    sections = extract_pdf_sections(pdf_path)
    assert sections["ok"] is True
    assert sections["sections"][0]["title"] == "Abstract"

    monkeypatch.setitem(sys.modules, "pypdf", None)
    repaired = repair_pdf(pdf_path, reason="encrypted-test", out_dir=tmp_path / "rep")
    assert repaired is not None and repaired.exists()


def test_pypdf_contract_handles_encryption_nested_outline_and_corrupt_pages(
    tmp_path: Path, monkeypatch
) -> None:
    pdf_path = tmp_path / "pypdf_contract.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n%%EOF\n")

    decrypt_calls: list[str] = []
    strict_flags: list[bool] = []

    class _GoodPage:
        def __init__(self, text: str | None) -> None:
            self._text = text

        def extract_text(self):
            return self._text

    class _CorruptPage:
        def extract_text(self):
            raise ValueError("corrupt content stream")

    class _DestObj:
        def __init__(self, title: str, page_idx: int) -> None:
            self.title = title
            self.page_idx = page_idx

    class _ContractPdfReader:
        def __init__(self, _path: str, strict: bool = True) -> None:
            strict_flags.append(strict)
            self.is_encrypted = True
            self.pages = [
                _GoodPage("Abstract\nSyner-\ngistic tracking."),
                _CorruptPage(),
                _GoodPage(None),
                _GoodPage("Conclusion\nRobust pipeline."),
            ]
            self.metadata = types.SimpleNamespace(title="Contract Paper")
            self.outline = [
                _DestObj("Abstract", 0),
                [{"/Title": "1.1 Methods", "_page": 0}],
                _DestObj("Conclusion", 3),
            ]

        def decrypt(self, password: str) -> int:
            decrypt_calls.append(password)
            self.is_encrypted = False
            return 1

        def get_destination_page_number(self, dest) -> int:
            if isinstance(dest, _DestObj):
                return dest.page_idx
            if isinstance(dest, dict) and "_page" in dest:
                return int(dest["_page"])
            raise ValueError("unknown dest")

    class _FailingPdfWriter:
        def add_page(self, _page) -> None:
            return

        def write(self, _fh) -> None:
            raise OSError("disk write error")

    fake_pypdf = types.SimpleNamespace(
        PdfReader=_ContractPdfReader,
        PdfWriter=_FailingPdfWriter,
    )
    monkeypatch.setitem(sys.modules, "fitz", None)
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    text, pages_read, total_pages, backend = extract_pdf_text(
        pdf_path, start_index=0, pages_limit=4, dehyphenate=True
    )
    assert backend == "pypdf"
    assert pages_read == 4
    assert total_pages == 4
    assert "Synergistic tracking." in text
    assert "Robust pipeline." in text
    assert strict_flags and all(flag is False for flag in strict_flags)
    assert decrypt_calls and decrypt_calls[0] == ""

    pages, scanned, _ = extract_pdf_pages(pdf_path, max_pages=4)
    assert scanned == 4
    # Corrupt page 2 and None page 3 skipped cleanly without losing pages 1 and 4.
    assert [p_num for p_num, _ in pages] == [1, 4]

    outline = extract_pdf_outline(pdf_path)
    assert outline == [
        {"level": 1, "title": "Abstract", "page": 0},
        {"level": 2, "title": "1.1 Methods", "page": 0},
        {"level": 1, "title": "Conclusion", "page": 3},
    ]

    sections = extract_pdf_sections(pdf_path, max_sections=10, max_pages=4)
    assert sections["ok"] is True
    assert [s["title"] for s in sections["sections"][:3]] == [
        "Abstract",
        "1.1 Methods",
        "Conclusion",
    ]
    assert all(s["source"] == "outline" for s in sections["sections"][:3])

    # Atomic temp file must be cleaned up when pypdf PdfWriter fails.
    repair_dir = tmp_path / "failed_repair"
    assert repair_pdf(pdf_path, reason="write-fail", out_dir=repair_dir) is None
    assert list(repair_dir.glob("*.tmp")) == []
