from __future__ import annotations

from annolid.utils.citations import (
    BibEntry,
    load_bibtex,
    merge_validated_fields,
    remove_entry,
    save_bibtex,
    search_entries,
    upsert_entry,
    validate_citation_metadata,
)


def test_bibtex_round_trip_parse_and_save(tmp_path) -> None:
    bib_path = tmp_path / "refs.bib"
    original = [
        BibEntry(
            entry_type="article",
            key="annolid2024",
            fields={
                "title": "Annolid toolkit",
                "author": "Liu, Jun and Team",
                "year": "2024",
                "doi": "10.1000/example",
            },
        ),
        BibEntry(
            entry_type="inproceedings",
            key="vision2023",
            fields={"title": "Vision", "author": "Doe, Jane", "year": "2023"},
        ),
    ]
    save_bibtex(bib_path, original)
    loaded = load_bibtex(bib_path)
    assert [entry.key for entry in loaded] == ["annolid2024", "vision2023"]
    assert loaded[0].fields["doi"] == "10.1000/example"


def test_upsert_and_remove_entries() -> None:
    entries: list[BibEntry] = []
    entries, created = upsert_entry(
        entries,
        BibEntry(
            entry_type="article",
            key="alpha",
            fields={"title": "A", "year": "2024"},
        ),
    )
    assert created is True
    assert len(entries) == 1

    entries, created = upsert_entry(
        entries,
        BibEntry(
            entry_type="article",
            key="alpha",
            fields={"title": "A2", "year": "2025"},
        ),
    )
    assert created is False
    assert len(entries) == 1
    assert entries[0].fields["title"] == "A2"

    entries, removed = remove_entry(entries, "alpha")
    assert removed is True
    assert entries == []


def test_search_entries_by_field() -> None:
    entries = [
        BibEntry(
            entry_type="article",
            key="one",
            fields={"title": "Deep Learning for Mice", "author": "Ada"},
        ),
        BibEntry(
            entry_type="article",
            key="two",
            fields={"title": "Another Paper", "author": "Bob"},
        ),
    ]
    by_title = search_entries(entries, "mice", field="title")
    assert [entry.key for entry in by_title] == ["one"]
    by_all = search_entries(entries, "bob")
    assert [entry.key for entry in by_all] == ["two"]


def test_validate_citation_metadata_prefers_google_scholar(monkeypatch) -> None:
    import annolid.utils.citations as citations

    monkeypatch.setattr(
        citations,
        "_google_scholar_lookup",
        lambda **kwargs: {
            "ok": True,
            "candidate": {
                "title": "Annolid toolkit",
                "author": "Liu, Jun and Team",
                "year": "2024",
                "doi": "10.1000/example",
            },
        },
    )
    monkeypatch.setattr(
        citations,
        "_crossref_lookup_doi",
        lambda *args, **kwargs: {"ok": False, "candidate": {}},
    )

    result = validate_citation_metadata(
        {"title": "Annolid toolkit", "year": "2024", "doi": "10.1000/example"}
    )
    assert result["checked"] is True
    assert result["provider"] == "google_scholar"
    assert result["verified"] is True


def test_google_scholar_lookup_query_parses_bibtex(monkeypatch) -> None:
    import annolid.utils.citations as citations

    responses = [
        '<a href="/scholar?q=info:abc123:scholar.google.com/&output=cite&hl=en">Cite</a>',
        '<a href="/scholar.bib?q=info:abc123:scholar.google.com&output=citation&scisdr=1">BibTeX</a>',
        (
            "@article{annolid2024,\n"
            "  title={Annolid toolkit},\n"
            "  author={Liu, Jun and Team},\n"
            "  year={2024},\n"
            "  doi={10.1000/example},\n"
            "}\n"
        ),
    ]

    def _fake_http_text(url: str, *, timeout_s: float) -> str:
        assert timeout_s > 0
        return responses.pop(0)

    monkeypatch.setattr(citations, "_http_text", _fake_http_text)
    result = citations._google_scholar_lookup_query(
        query="Annolid toolkit", timeout_s=1.0
    )
    assert result["ok"] is True
    candidate = dict(result["candidate"])
    assert candidate["title"] == "Annolid toolkit"
    assert candidate["year"] == "2024"
    assert candidate["doi"] == "10.1000/example"


def test_merge_validated_fields_replaces_with_scholar_on_exact_doi() -> None:
    fields = {
        "title": "ZEBrA - Zebra finch Expression Brain Atlas ... - PMC",
        "year": "2021",
        "doi": "10.1002/cne.24879",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8219259/",
    }
    validation = {
        "provider": "google_scholar",
        "score": 0.66,
        "candidate": {
            "__bibkey__": "lovell2020zebra",
            "title": "ZEBrA: Zebra finch Expression Brain Atlas",
            "author": "Lovell, Peter V and Wirthlin, Morgan",
            "journal": "Journal of Comparative Neurology",
            "year": "2020",
            "doi": "10.1002/cne.24879",
            "volume": "528",
            "number": "12",
            "pages": "2099--2131",
            "publisher": "Wiley Online Library",
        },
    }
    merged = merge_validated_fields(fields, validation, replace_when_confident=True)
    assert merged["title"] == "ZEBrA: Zebra finch Expression Brain Atlas"
    assert merged["year"] == "2020"
    assert merged["volume"] == "528"
    assert merged["pages"] == "2099--2131"


def test_crossref_candidate_includes_all_authors() -> None:
    import annolid.utils.citations as citations

    candidate = citations._crossref_message_to_candidate(
        {
            "title": ["ZEBrA"],
            "issued": {"date-parts": [[2020]]},
            "DOI": "10.1002/cne.24879",
            "author": [
                {"given": "Peter V.", "family": "Lovell"},
                {"given": "Morgan", "family": "Wirthlin"},
                {"given": "Taylor", "family": "Kaser"},
            ],
            "container-title": ["Journal of Comparative Neurology"],
        }
    )
    assert (
        candidate["author"] == "Lovell, Peter V. and Wirthlin, Morgan and Kaser, Taylor"
    )


def test_parse_bibtex_skips_comment_preamble_and_string_directives(tmp_path) -> None:
    from annolid.utils.citations import parse_bibtex

    raw = """
@comment{This is a file header comment}
@preamble{"\\newcommand{\\noop}[1]{}"}
@string{CVPR = "IEEE CVPR"}
@article{smith2024,
  title = {Behavior Tracking},
  author = {Smith, Alice},
  year = {2024},
}
"""
    entries = parse_bibtex(raw)
    assert [e.key for e in entries] == ["smith2024"]

    bib_path = tmp_path / "atomic.bib"
    save_bibtex(bib_path, entries)
    assert bib_path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_google_scholar_lookup_rejects_untrusted_redirect_hosts_and_caps_bytes(
    monkeypatch,
) -> None:
    import pytest
    import annolid.utils.citations as citations

    # Untrusted host in extracted scholar.bib link must be rejected (no SSRF).
    fetched_urls: list[str] = []
    responses = [
        '<a href="/scholar?q=info:abc:scholar.google.com/&output=cite">Cite</a>',
        '<a href="https://evil.example.org/scholar.bib?id=1">BibTeX</a>',
    ]

    def _fake_http_text(url: str, *, timeout_s: float) -> str:
        fetched_urls.append(url)
        return responses.pop(0)

    monkeypatch.setattr(citations, "_http_text", _fake_http_text)
    result = citations._google_scholar_lookup_query(query="Test", timeout_s=1.0)
    assert result["ok"] is False
    assert not any("evil.example.org" in u for u in fetched_urls)

    # Unsupported URL scheme rejected directly
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        citations._validate_http_url("file:///etc/passwd")

    # Bounded HTTP read rejects oversized responses
    class _OversizedResponse:
        def read(self, n: int = -1) -> bytes:
            return b"x" * (n if n > 0 else 1024)

    with pytest.raises(ValueError, match="exceeded"):
        citations._read_bounded_http(_OversizedResponse(), max_bytes=32)
