from __future__ import annotations

import annolid.services.literature_search as literature
import urllib.error


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
