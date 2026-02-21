from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict

_BIBTEX_ENTRY_RE = re.compile(r"@[a-zA-Z][a-zA-Z0-9_-]*\s*[\{\(]")


def _extract_bibtex_text(raw: str) -> str:
    text = str(raw or "")
    if not text.strip():
        return ""
    code_blocks = re.findall(
        r"```(?:\s*(?:bibtex|bib|tex))?\s*([\s\S]*?)```",
        text,
        flags=re.IGNORECASE,
    )
    candidates = [
        blk.strip() for blk in code_blocks if _BIBTEX_ENTRY_RE.search(blk or "")
    ]
    if candidates:
        return "\n\n".join(candidates).strip()
    marker = _BIBTEX_ENTRY_RE.search(text)
    if marker:
        return text[marker.start() :].strip()
    return text.strip()


def extract_doi(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    match = re.search(
        r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b",
        raw,
        flags=re.IGNORECASE,
    )
    return str(match.group(1) or "").rstrip(").,;!?") if match else ""


def extract_year(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", raw)
    return years[0] if years else ""


def normalize_citation_key(title: str, year: str, fallback: str = "paper") -> str:
    text = str(title or "").strip().lower()
    if not text:
        text = str(fallback or "paper").strip().lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    stem = "_".join(tokens[:3]) if tokens else "paper"
    yr = str(year or "").strip()
    if yr:
        return f"{stem}_{yr}"
    return stem


def resolve_bib_output_path(
    bib_file: str,
    *,
    workspace: Path,
) -> Path:
    target = str(bib_file or "").strip()
    if not target:
        return workspace / "citations.bib"
    candidate = Path(target).expanduser()
    if candidate.is_absolute():
        try:
            candidate.relative_to(workspace)
            return candidate
        except Exception:
            return workspace / candidate.name
    return workspace / candidate


def citation_fields_from_pdf_state(
    *,
    get_pdf_state: Callable[[], Dict[str, Any]],
    get_pdf_text: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    state = get_pdf_state()
    if not isinstance(state, dict) or not bool(state.get("ok")):
        return {}
    if not bool(state.get("has_pdf")):
        return {}
    path = str(state.get("path") or "").strip()
    if not path:
        return {}
    pdf_text_payload = get_pdf_text(max_chars=8000, pages=2)
    text = (
        str(pdf_text_payload.get("text") or "")
        if isinstance(pdf_text_payload, dict)
        else ""
    )
    title = str(state.get("title") or "").strip()
    if title.lower().endswith(".pdf"):
        title = title[:-4]
    doi = extract_doi(text)
    year = extract_year(text)
    fields: Dict[str, Any] = {
        "title": title or Path(path).stem.replace("_", " "),
        "year": year,
        "doi": doi,
        "url": "",
        "source_path": path,
        "note": "Saved from active Annolid PDF viewer.",
    }
    if doi:
        fields["url"] = f"https://doi.org/{doi}"
    arxiv_match = re.search(
        r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    if arxiv_match:
        fields.setdefault("archiveprefix", "arXiv")
        fields.setdefault("eprint", str(arxiv_match.group(1) or "").strip())
    return {"source": "pdf", "path": path, "fields": fields}


def citation_fields_from_web_state(
    *,
    get_web_state: Callable[[], Dict[str, Any]],
    get_web_text: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    state = get_web_state()
    if not isinstance(state, dict) or not bool(state.get("ok")):
        return {}
    if not bool(state.get("has_page")):
        return {}
    url = str(state.get("url") or "").strip()
    if not url:
        return {}
    dom_payload = get_web_text(max_chars=9000)
    text = str(dom_payload.get("text") or "") if isinstance(dom_payload, dict) else ""
    title = str(state.get("title") or "").strip()
    doi = extract_doi(text or url)
    year = extract_year(text)
    fields: Dict[str, Any] = {
        "title": title or "Web page citation",
        "year": year,
        "doi": doi,
        "url": url,
        "note": "Saved from active Annolid web viewer.",
    }
    arxiv_match = re.search(
        r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)",
        url,
        flags=re.IGNORECASE,
    )
    if arxiv_match:
        fields.setdefault("archiveprefix", "arXiv")
        fields.setdefault("eprint", str(arxiv_match.group(1) or "").strip())
    return {"source": "web", "url": url, "fields": fields}


def save_citation_tool(
    *,
    key: str,
    bib_file: str,
    source: str,
    entry_type: str,
    validate_before_save: bool,
    strict_validation: bool,
    choose_pdf_fields: Callable[[], Dict[str, Any]],
    choose_web_fields: Callable[[], Dict[str, Any]],
    resolve_bib_path: Callable[[str], Path],
    validate_basic_fields: Callable[[Dict[str, str]], Any],
    validate_metadata: Callable[[Dict[str, str], float], Dict[str, Any]],
    merge_fields: Callable[[Dict[str, str], Dict[str, Any], bool], Dict[str, str]],
    load_bibtex: Callable[[Path], Any],
    upsert_entry: Callable[[Any, Any], Any],
    save_bibtex: Callable[[Path, Any], None],
    bib_entry_cls: Any,
) -> Dict[str, Any]:
    source_pref = str(source or "auto").strip().lower()
    if source_pref not in {"auto", "pdf", "web"}:
        source_pref = "auto"
    chosen: Dict[str, Any] = {}
    if source_pref in {"auto", "pdf"}:
        chosen = choose_pdf_fields()
    if (not chosen) and source_pref in {"auto", "web"}:
        chosen = choose_web_fields()
    if not chosen:
        return {
            "ok": False,
            "error": (
                "No active paper context found. Open a PDF or web page first, "
                "then ask to save citation."
            ),
        }

    fields_raw = dict(chosen.get("fields") or {})
    fields: Dict[str, str] = {}
    for k, v in fields_raw.items():
        name = str(k or "").strip().lower()
        value = str(v or "").strip()
        if name and value:
            fields[name] = value

    basic_errors = validate_basic_fields(
        {
            "__key__": str(key or "").strip(),
            "year": str(fields.get("year") or ""),
            "doi": str(fields.get("doi") or ""),
        }
    )
    if basic_errors:
        return {"ok": False, "error": " ".join(basic_errors)}

    resolved_bib = resolve_bib_path(str(bib_file or ""))
    resolved_bib.parent.mkdir(parents=True, exist_ok=True)
    title = str(fields.get("title") or "").strip()
    year = str(fields.get("year") or "").strip()
    key_override = str(key or "").strip()
    normalized_key = key_override
    if not normalized_key:
        normalized_key = normalize_citation_key(
            title, year, fallback=str(chosen.get("source") or "paper")
        )
    normalized_key = re.sub(r"[^a-zA-Z0-9:_\-.]+", "_", normalized_key).strip("_")
    if not normalized_key:
        normalized_key = "paper"

    validation: Dict[str, Any] = {
        "checked": False,
        "verified": False,
        "provider": "",
        "score": 0.0,
        "message": "",
        "candidate": {},
    }
    if bool(validate_before_save):
        validation = validate_metadata(fields, 1.8)
        fields = merge_fields(fields, validation, True)
        if bool(strict_validation) and not bool(validation.get("verified")):
            return {
                "ok": False,
                "error": (
                    "Citation validation failed strict mode. "
                    + str(validation.get("message") or "")
                ).strip(),
                "validation": validation,
            }
        if not key_override:
            candidate_key = str(
                dict(validation.get("candidate") or {}).get("__bibkey__") or ""
            ).strip()
            if candidate_key:
                normalized_key = candidate_key
            else:
                normalized_key = normalize_citation_key(
                    str(fields.get("title") or "").strip(),
                    str(fields.get("year") or "").strip(),
                    fallback=str(chosen.get("source") or "paper"),
                )
            normalized_key = re.sub(r"[^a-zA-Z0-9:_\-.]+", "_", normalized_key).strip(
                "_"
            )
            if not normalized_key:
                normalized_key = "paper"

    entries = load_bibtex(resolved_bib) if resolved_bib.exists() else []
    updated, created = upsert_entry(
        entries,
        bib_entry_cls(
            entry_type=str(entry_type or "article").strip().lower() or "article",
            key=normalized_key,
            fields=fields,
        ),
    )
    save_bibtex(resolved_bib, updated, sort_keys=True)
    return {
        "ok": True,
        "created": bool(created),
        "key": normalized_key,
        "bib_file": str(resolved_bib),
        "source": str(chosen.get("source") or source_pref),
        "fields": fields,
        "validation": validation,
    }


def add_citation_raw_tool(
    *,
    bibtex: str,
    bib_file: str,
    parse_bibtex: Callable[[str], Any],
    resolve_bib_path: Callable[[str], Path],
    load_bibtex: Callable[[Path], Any],
    upsert_entry: Callable[[Any, Any], Any],
    save_bibtex: Callable[[Path, Any], None],
) -> Dict[str, Any]:
    raw = str(bibtex or "").strip()
    if not raw:
        return {"ok": False, "error": "No BibTeX entry provided."}
    entries_in = parse_bibtex(raw)
    if not entries_in:
        extracted = _extract_bibtex_text(raw)
        entries_in = parse_bibtex(extracted) if extracted else []
    if not entries_in:
        return {"ok": False, "error": "Could not parse a valid BibTeX entry."}
    entries_valid = [e for e in entries_in if str(getattr(e, "key", "") or "").strip()]
    if not entries_valid:
        return {"ok": False, "error": "BibTeX entry key is required."}
    resolved_bib = resolve_bib_path(str(bib_file or ""))
    resolved_bib.parent.mkdir(parents=True, exist_ok=True)
    existing = load_bibtex(resolved_bib) if resolved_bib.exists() else []
    created_count = 0
    updated_count = 0
    keys: list[str] = []
    for entry in entries_valid:
        existing, created = upsert_entry(existing, entry)
        keys.append(str(entry.key))
        if created:
            created_count += 1
        else:
            updated_count += 1
    save_bibtex(resolved_bib, existing, sort_keys=True)
    single_key = keys[0] if len(keys) == 1 else ""
    single_entry = entries_valid[0] if len(entries_valid) == 1 else None
    return {
        "ok": True,
        "created": bool(created_count > 0 and updated_count == 0),
        "key": single_key,
        "keys": keys,
        "created_count": int(created_count),
        "updated_count": int(updated_count),
        "bib_file": str(resolved_bib),
        "entry_type": str(getattr(single_entry, "entry_type", "") or ""),
        "entry_count": len(keys),
    }


def list_citations_tool(
    *,
    bib_file: str,
    query: str,
    limit: int,
    resolve_bib_path: Callable[[str], Path],
    load_bibtex: Callable[[Path], Any],
    search_entries: Callable[[Any, str, int], Any],
    entry_to_dict: Callable[[Any], Dict[str, Any]],
) -> Dict[str, Any]:
    resolved_bib = resolve_bib_path(str(bib_file or ""))
    if not resolved_bib.exists():
        return {
            "ok": True,
            "count": 0,
            "entries": [],
            "bib_file": str(resolved_bib),
        }
    entries = load_bibtex(resolved_bib)
    q = str(query or "").strip()
    if q:
        entries = search_entries(entries, q, limit=max(1, int(limit or 20)))
    total = len(entries)
    capped = entries[: max(1, int(limit or 20))]
    return {
        "ok": True,
        "count": total,
        "entries": [entry_to_dict(e) for e in capped],
        "bib_file": str(resolved_bib),
        "query": q,
    }
