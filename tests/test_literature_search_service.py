from __future__ import annotations

import annolid.services.literature_search as literature
import urllib.error


def test_literature_result_from_dict_round_trips_basic_fields() -> None:
    original = literature.LiteratureResult(
        source="openalex",
        title="Tracking Animals",
        summary="A summary.",
        id_url="https://example.org/work",
        doi="10.1234/example",
        year=2024,
    )

    restored = literature.LiteratureResult.from_dict(original.to_dict())
    assert restored == original


def test_search_literature_fallback_and_source_stats(monkeypatch) -> None:
    with literature._CACHE_LOCK:
        literature._CACHE.clear()

    def _openalex(*_, **__):  # type: ignore[no-untyped-def]
        raise RuntimeError("rate limit")

    monkeypatch.setattr(literature, "_search_openalex", _openalex)
    monkeypatch.setattr(
        literature,
        "_search_crossref",
        lambda *_, **__: [
            literature.LiteratureResult(
                source="crossref",
                title="A CrossRef Entry",
                doi="10.1000/demo",
            )
        ],
    )
    monkeypatch.setattr(
        literature,
        "_search_arxiv",
        lambda *_, **__: [
            literature.LiteratureResult(
                source="arxiv",
                title="An arXiv Entry",
                arxiv_id="2602.17594v1",
                pdf_url="https://arxiv.org/pdf/2602.17594v1.pdf",
            )
        ],
    )

    payload = literature.search_literature(
        "mouse behavior",
        max_results=5,
        sources=("openalex", "crossref", "arxiv"),
        use_cache=False,
    )

    assert payload["cache_hit"] is False
    assert payload["source_counts"] == {"crossref": 1, "arxiv": 1}
    assert payload["degraded_sources"] == ["openalex"]
    metrics = payload["source_metrics"]
    assert metrics["openalex"]["success"] is False
    assert int(metrics["openalex"]["attempts"]) >= 1
    assert metrics["crossref"]["success"] is True
    assert metrics["arxiv"]["success"] is True
    rows = payload["results"]
    assert isinstance(rows, list)
    assert len(rows) == 2
    assert rows[0]["source"] == "crossref"
    assert rows[1]["source"] == "arxiv"


def test_search_literature_uses_cache(monkeypatch) -> None:
    with literature._CACHE_LOCK:
        literature._CACHE.clear()
    calls = {"n": 0}

    def _arxiv(*_, **__):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return [
            literature.LiteratureResult(
                source="arxiv",
                title="Cached Entry",
                arxiv_id="2602.17594",
                pdf_url="https://arxiv.org/pdf/2602.17594.pdf",
            )
        ]

    monkeypatch.setattr(literature, "_search_arxiv", _arxiv)
    monkeypatch.setattr(literature, "_search_openalex", lambda *_, **__: [])
    monkeypatch.setattr(literature, "_search_crossref", lambda *_, **__: [])

    import tempfile
    from pathlib import Path

    cache_dir = Path(tempfile.mkdtemp()) / "lit_cache"
    first = literature.search_literature(
        "cached query",
        max_results=2,
        sources=("arxiv",),
        use_cache=True,
        cache_dir=cache_dir,
    )
    second = literature.search_literature(
        "cached query",
        max_results=2,
        sources=("arxiv",),
        use_cache=True,
        cache_dir=cache_dir,
    )

    assert calls["n"] == 1
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert float(second["cache_age_seconds"]) >= 0.0
    assert first["results"] == second["results"]


def test_search_literature_uses_disk_cache_across_memory_clear(
    monkeypatch, tmp_path
) -> None:
    with literature._CACHE_LOCK:
        literature._CACHE.clear()
    calls = {"n": 0}

    def _arxiv(*_, **__):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return [
            literature.LiteratureResult(
                source="arxiv",
                title="Disk Cached Entry",
                arxiv_id="2602.17594",
                pdf_url="https://arxiv.org/pdf/2602.17594.pdf",
            )
        ]

    monkeypatch.setattr(literature, "_search_arxiv", _arxiv)
    monkeypatch.setattr(literature, "_search_openalex", lambda *_, **__: [])
    monkeypatch.setattr(literature, "_search_crossref", lambda *_, **__: [])

    cache_dir = tmp_path / "lit_cache"
    first = literature.search_literature(
        "disk cached query",
        max_results=2,
        sources=("arxiv",),
        use_cache=True,
        cache_dir=cache_dir,
    )
    assert first["cache_hit"] is False
    with literature._CACHE_LOCK:
        literature._CACHE.clear()
    second = literature.search_literature(
        "disk cached query",
        max_results=2,
        sources=("arxiv",),
        use_cache=True,
        cache_dir=cache_dir,
    )
    assert calls["n"] == 1
    assert second["cache_hit"] is True
    assert float(second["cache_age_seconds"]) >= 0.0


def test_search_literature_retries_transient_source_error(monkeypatch) -> None:
    with literature._CACHE_LOCK:
        literature._CACHE.clear()
    calls = {"n": 0}
    sleeps: list[float] = []

    def _openalex(*_, **__):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("temporary upstream error")
        return [
            literature.LiteratureResult(
                source="openalex",
                title="Recovered entry",
            )
        ]

    monkeypatch.setattr(literature, "_search_openalex", _openalex)
    monkeypatch.setattr(literature, "time", literature.time)
    monkeypatch.setattr(literature.time, "sleep", lambda s: sleeps.append(float(s)))
    monkeypatch.setattr(literature, "_search_crossref", lambda *_, **__: [])
    monkeypatch.setattr(literature, "_search_arxiv", lambda *_, **__: [])

    payload = literature.search_literature(
        "transient query",
        max_results=3,
        sources=("openalex",),
        use_cache=False,
    )

    assert calls["n"] == 2
    assert payload["degraded_sources"] == []
    assert payload["source_counts"] == {"openalex": 1}
    assert payload["source_metrics"]["openalex"]["success"] is True
    assert int(payload["source_metrics"]["openalex"]["attempts"]) == 2
    assert sleeps, "Expected backoff sleep to be invoked"


def test_search_literature_rate_limit_degrades_to_other_sources(monkeypatch) -> None:
    with literature._CACHE_LOCK:
        literature._CACHE.clear()

    def _openalex(*_, **__):  # type: ignore[no-untyped-def]
        raise urllib.error.HTTPError(
            url="https://api.openalex.org/works",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(literature, "_search_openalex", _openalex)
    monkeypatch.setattr(
        literature,
        "_search_crossref",
        lambda *_, **__: [
            literature.LiteratureResult(
                source="crossref",
                title="CrossRef fallback result",
                doi="10.1000/fallback",
            )
        ],
    )
    monkeypatch.setattr(literature, "_search_arxiv", lambda *_, **__: [])

    payload = literature.search_literature(
        "rate limited query",
        max_results=3,
        sources=("openalex", "crossref"),
        use_cache=False,
    )
    assert payload["degraded_sources"] == ["openalex"]
    assert payload["source_metrics"]["openalex"]["success"] is False
    assert payload["source_counts"] == {"crossref": 1}
    assert payload["results"][0]["source"] == "crossref"


def test_search_backends_parse_authors_venue_abstract_and_block_xxe(
    monkeypatch,
) -> None:
    import pytest

    # 1) OpenAlex abstract_inverted_index + authors + venue
    monkeypatch.setattr(
        literature,
        "_http_json",
        lambda url, timeout_s=10.0: (
            {
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "title": "Pose Tracking in Rodents",
                        "publication_year": 2024,
                        "doi": "https://doi.org/10.1000/rodent",
                        "ids": {"arxiv": "https://arxiv.org/abs/2401.12345"},
                        "primary_location": {
                            "source": {"display_name": "Nature Methods"},
                            "pdf_url": "https://example.org/rodent.pdf",
                        },
                        "authorships": [
                            {"author": {"display_name": "Alice Smith"}},
                            {"author": {"display_name": "Bob Chen"}},
                        ],
                        "abstract_inverted_index": {
                            "Accurate": [0],
                            "rodent": [1],
                            "tracking.": [2],
                        },
                    }
                ]
            }
            if "openalex" in url
            else {
                "message": {
                    "items": [
                        {
                            "title": ["CrossRef Behavior Study"],
                            "DOI": "10.2000/cr",
                            "URL": "https://doi.org/10.2000/cr",
                            "issued": {"date-parts": [[2023]]},
                            "author": [{"given": "Jane", "family": "Doe"}],
                            "container-title": ["eLife"],
                            "abstract": "<jats:p>Measured social bouts.</jats:p>",
                        }
                    ]
                }
            }
        ),
    )
    oa_rows = literature._search_openalex("rodent", max_results=2, timeout_s=5.0)
    assert len(oa_rows) == 1
    assert oa_rows[0].authors == ("Alice Smith", "Bob Chen")
    assert oa_rows[0].venue == "Nature Methods"
    assert oa_rows[0].summary == "Accurate rodent tracking."
    assert oa_rows[0].pdf_url == "https://example.org/rodent.pdf"

    cr_rows = literature._search_crossref("behavior", max_results=2, timeout_s=5.0)
    assert len(cr_rows) == 1
    assert cr_rows[0].authors == ("Jane Doe",)
    assert cr_rows[0].venue == "eLife"
    assert cr_rows[0].summary == "Measured social bouts."

    # 2) arXiv parses authors + published year
    valid_atom = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>http://arxiv.org/abs/2401.12345v2</id>
        <published>2024-01-19T10:00:00Z</published>
        <title>Pose Tracking in Rodents</title>
        <summary>Detailed arXiv abstract for rodent tracking.</summary>
        <author><name>Alice Smith</name></author>
        <author><name>Bob Chen</name></author>
      </entry>
    </feed>"""
    monkeypatch.setattr(
        literature, "_http_bytes", lambda _u, timeout_s=10.0: valid_atom
    )
    ax_rows = literature._search_arxiv("rodent", max_results=2, timeout_s=5.0)
    assert len(ax_rows) == 1
    assert ax_rows[0].year == 2024
    assert ax_rows[0].authors == ("Alice Smith", "Bob Chen")
    assert ax_rows[0].venue == "arXiv"

    # 3) arXiv rejects DOCTYPE/ENTITY declarations
    xxe_atom = b'<!DOCTYPE feed [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><feed/>'
    monkeypatch.setattr(literature, "_http_bytes", lambda _u, timeout_s=10.0: xxe_atom)
    with pytest.raises(ValueError, match="DOCTYPE/ENTITY"):
        literature._search_arxiv("rodent", max_results=1, timeout_s=5.0)


def test_deduplicate_merges_complementary_metadata_across_sources() -> None:
    oa = literature.LiteratureResult(
        source="openalex",
        title="Pose Tracking in Rodents",
        doi="10.1000/rodent",
        year=2024,
        venue="Nature Methods",
    )
    ax = literature.LiteratureResult(
        source="arxiv",
        title="Pose Tracking in Rodents",
        summary="Full abstract from arXiv.",
        arxiv_id="2401.12345v1",
        pdf_url="https://arxiv.org/pdf/2401.12345v1.pdf",
        authors=("Alice Smith", "Bob Chen"),
        venue="arXiv",
    )
    merged = literature._deduplicate([oa, ax])
    assert len(merged) == 1
    item = merged[0]
    assert item.doi == "10.1000/rodent"
    assert item.arxiv_id == "2401.12345v1"
    assert item.summary == "Full abstract from arXiv."
    assert item.authors == ("Alice Smith", "Bob Chen")
    assert item.venue == "Nature Methods"
    assert item.pdf_url == "https://arxiv.org/pdf/2401.12345v1.pdf"


def test_literature_search_handles_legacy_arxiv_best_oa_entities_and_404_fail_fast(
    monkeypatch,
) -> None:
    import pytest

    # Legacy category/number arXiv ID preserved
    assert (
        literature._extract_arxiv_id("https://arxiv.org/abs/cs/0112017v1")
        == "cs/0112017v1"
    )
    assert (
        literature._extract_arxiv_id("https://arxiv.org/pdf/2401.12345v2.pdf")
        == "2401.12345v2"
    )

    # URL scheme check blocks file://
    with pytest.raises(ValueError, match="Unsupported URL scheme"):
        literature._validate_http_url("file:///etc/passwd")

    # OpenAlex falls back to best_oa_location.pdf_url and unescapes HTML entities
    monkeypatch.setattr(
        literature,
        "_http_json",
        lambda url, timeout_s=10.0: (
            {
                "results": [
                    {
                        "id": "https://openalex.org/W2",
                        "title": "OA Best Location",
                        "publication_year": 2024,
                        "primary_location": {"source": {"display_name": "PLOS ONE"}},
                        "best_oa_location": {"pdf_url": "https://oa.example.org/p.pdf"},
                        "abstract_inverted_index": {"A": [0], "&amp;": [1], "B": [2]},
                    }
                ]
            }
            if "openalex" in url
            else {
                "message": {
                    "items": [
                        {
                            "title": ["Entity Study"],
                            "DOI": "10.3000/ent",
                            "abstract": "<jats:p>Cats &amp; mice &lt; 0.05</jats:p>",
                        }
                    ]
                }
            }
        ),
    )
    oa = literature._search_openalex("oa", max_results=1, timeout_s=5.0)
    assert oa[0].pdf_url == "https://oa.example.org/p.pdf"
    assert oa[0].summary == "A & B"

    cr = literature._search_crossref("ent", max_results=1, timeout_s=5.0)
    assert cr[0].summary == "Cats & mice < 0.05"

    # Permanent 404 fails fast in 1 attempt instead of sleeping through 3 retries
    attempts = {"n": 0}

    def _fail_404():
        attempts["n"] += 1
        raise urllib.error.HTTPError(
            url="https://api.openalex.org/works",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    rows, metrics = literature._run_source_with_retry(
        source="openalex", run_once=_fail_404
    )
    assert rows == []
    assert attempts["n"] == 1
    assert metrics["attempts"] == 1
    assert metrics["success"] is False
