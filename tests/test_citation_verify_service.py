from __future__ import annotations

from pathlib import Path

from annolid.services.citation_verify import (
    build_citation_batch_report,
    build_citation_verification_report,
    classify_citation_status,
    write_citation_verification_report,
)


def test_classify_citation_status_variants() -> None:
    verified = classify_citation_status(
        fields={"title": "Paper"},
        validation={"checked": True, "verified": True, "score": 0.88},
    )
    assert verified[0] == "verified"

    suspicious = classify_citation_status(
        fields={"title": "Paper"},
        validation={"checked": True, "verified": False, "score": 0.51},
    )
    assert suspicious[0] == "suspicious"

    hallucinated = classify_citation_status(
        fields={"title": "Paper"},
        validation={"checked": True, "verified": False, "score": 0.1, "candidate": {}},
    )
    assert hallucinated[0] == "hallucinated"

    skipped = classify_citation_status(
        fields={},
        validation={"checked": False, "verified": False, "score": 0.0},
    )
    assert skipped[0] == "skipped"


def test_build_and_write_citation_verification_report(tmp_path: Path) -> None:
    report = build_citation_verification_report(
        key="annolid2024",
        bib_file=str(tmp_path / "refs.bib"),
        source="pdf",
        fields={"title": "Annolid Toolkit", "year": "2024", "doi": "10.1000/example"},
        validation={
            "checked": True,
            "verified": True,
            "provider": "crossref",
            "score": 0.95,
            "message": "strong match",
            "candidate": {"title": "Annolid Toolkit"},
        },
    )
    assert report["verification"]["status"] == "verified"
    assert report["summary"]["counts"]["verified"] == 1
    output = write_citation_verification_report(
        report,
        reports_dir=tmp_path / "verification",
        report_stem="refs_annolid2024",
    )
    assert output.exists()


def test_build_citation_batch_report_counts_and_score() -> None:
    batch = build_citation_batch_report(
        bib_file="refs.bib",
        entries=[
            {"key": "a", "status": "verified", "integrity_score": 0.9},
            {"key": "b", "status": "suspicious", "integrity_score": 0.5},
            {"key": "c", "status": "hallucinated", "integrity_score": 0.0},
        ],
    )
    assert batch["summary"]["total"] == 3
    assert batch["summary"]["counts"]["verified"] == 1
    assert batch["summary"]["counts"]["suspicious"] == 1
    assert batch["summary"]["counts"]["hallucinated"] == 1


def test_classify_citation_status_candidate_promotes_suspicious() -> None:
    status, integrity, _reason = classify_citation_status(
        fields={"title": "Candidate-assisted citation"},
        validation={
            "checked": True,
            "verified": False,
            "score": 0.1,
            "candidate": {"title": "Close match"},
        },
    )
    assert status == "suspicious"
    assert integrity >= 0.2


def test_classify_citation_status_unavailable_check_is_skipped() -> None:
    status, integrity, reason = classify_citation_status(
        fields={"title": "No external check"},
        validation={
            "checked": False,
            "verified": False,
            "score": 0.0,
            "candidate": {},
        },
    )
    assert status == "skipped"
    assert integrity == 0.2
    assert "unavailable" in reason.lower() or "skipped" in reason.lower()


def test_classify_citation_status_tolerates_non_numeric_score() -> None:
    status, integrity, _reason = classify_citation_status(
        fields={"title": "Malformed score case"},
        validation={
            "checked": True,
            "verified": False,
            "score": "not-a-number",
            "candidate": {},
        },
    )
    assert status == "hallucinated"
    assert integrity == 0.0
