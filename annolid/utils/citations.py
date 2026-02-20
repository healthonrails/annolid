from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import difflib
import html
import re
from pathlib import Path
from urllib.parse import quote, urljoin
import urllib.request
import json
from typing import Iterable


@dataclass
class BibEntry:
    entry_type: str
    key: str
    fields: dict[str, str] = field(default_factory=dict)


def parse_bibtex(text: str) -> list[BibEntry]:
    entries: list[BibEntry] = []
    i = 0
    n = len(text)
    while i < n:
        at = text.find("@", i)
        if at < 0:
            break
        i = at + 1
        while i < n and text[i].isspace():
            i += 1
        type_start = i
        while i < n and (text[i].isalnum() or text[i] in ("_", "-")):
            i += 1
        entry_type = text[type_start:i].strip().lower()
        if not entry_type:
            continue
        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] not in "{(":
            continue
        open_char = text[i]
        close_char = "}" if open_char == "{" else ")"
        close_idx = _find_matching(text, i, open_char, close_char)
        if close_idx < 0:
            break
        payload = text[i + 1 : close_idx].strip()
        key, fields_text = _split_top_level_once(payload, ",")
        key = (key or "").strip()
        if key:
            fields = _parse_fields(fields_text or "")
            entries.append(BibEntry(entry_type=entry_type, key=key, fields=fields))
        i = close_idx + 1
    return entries


def load_bibtex(path: str | Path) -> list[BibEntry]:
    bib_path = Path(path)
    if not bib_path.exists():
        return []
    return parse_bibtex(bib_path.read_text(encoding="utf-8"))


def serialize_bibtex(entries: Iterable[BibEntry], *, sort_keys: bool = True) -> str:
    ordered = list(entries)
    if sort_keys:
        ordered.sort(key=lambda e: e.key.lower())
    blocks: list[str] = []
    for entry in ordered:
        blocks.append(_serialize_entry(entry))
    text = "\n\n".join(blocks).rstrip()
    return f"{text}\n" if text else ""


def save_bibtex(
    path: str | Path, entries: Iterable[BibEntry], *, sort_keys: bool = True
) -> None:
    bib_path = Path(path)
    bib_path.parent.mkdir(parents=True, exist_ok=True)
    bib_path.write_text(
        serialize_bibtex(entries, sort_keys=sort_keys), encoding="utf-8"
    )


def upsert_entry(
    entries: list[BibEntry], new_entry: BibEntry
) -> tuple[list[BibEntry], bool]:
    key_norm = new_entry.key.strip().lower()
    for idx, entry in enumerate(entries):
        if entry.key.strip().lower() == key_norm:
            entries[idx] = new_entry
            return entries, False
    entries.append(new_entry)
    return entries, True


def remove_entry(entries: list[BibEntry], key: str) -> tuple[list[BibEntry], bool]:
    key_norm = key.strip().lower()
    out = [e for e in entries if e.key.strip().lower() != key_norm]
    return out, len(out) != len(entries)


def search_entries(
    entries: list[BibEntry],
    query: str,
    *,
    field: str | None = None,
    limit: int | None = None,
) -> list[BibEntry]:
    q = query.strip().lower()
    if not q:
        return list(entries[: limit or len(entries)])
    filtered: list[BibEntry] = []
    field_norm = field.strip().lower() if field else None
    for entry in entries:
        if field_norm:
            value = entry.fields.get(field_norm, "")
            haystack = value.lower()
        else:
            parts = [entry.key, entry.entry_type, *entry.fields.values()]
            haystack = " ".join(parts).lower()
        if q in haystack:
            filtered.append(entry)
            if limit and len(filtered) >= limit:
                break
    return filtered


def entry_to_dict(entry: BibEntry) -> dict[str, object]:
    return {
        "entry_type": entry.entry_type,
        "key": entry.key,
        "fields": dict(entry.fields),
        "title": entry.fields.get("title"),
        "author": entry.fields.get("author"),
        "year": entry.fields.get("year"),
    }


def validate_citation_metadata(
    fields: dict[str, str], *, timeout_s: float = 2.0
) -> dict[str, object]:
    title = str(fields.get("title") or "").strip()
    doi = str(fields.get("doi") or "").strip()
    year = str(fields.get("year") or "").strip()
    scholar = _google_scholar_lookup(title=title, doi=doi, timeout_s=timeout_s)
    if scholar.get("ok"):
        candidate = dict(scholar.get("candidate") or {})
        score = _score_candidate(
            title=title,
            year=year,
            doi=doi,
            candidate=candidate,
        )
        return {
            "checked": True,
            "verified": bool(score >= 0.7),
            "provider": "google_scholar",
            "score": score,
            "message": (
                "Validated via Google Scholar."
                if score >= 0.7
                else "Google Scholar match was weak."
            ),
            "candidate": candidate,
        }
    if doi:
        crossref = _crossref_lookup_doi(doi, timeout_s=timeout_s)
        if crossref.get("ok"):
            candidate = dict(crossref.get("candidate") or {})
            score = _score_candidate(
                title=title,
                year=year,
                doi=doi,
                candidate=candidate,
            )
            return {
                "checked": True,
                "verified": bool(score >= 0.7),
                "provider": "crossref",
                "score": score,
                "message": (
                    "Validated via DOI lookup."
                    if score >= 0.7
                    else "DOI lookup did not match strongly."
                ),
                "candidate": candidate,
            }
    if not title:
        return {
            "checked": False,
            "verified": False,
            "provider": "",
            "score": 0.0,
            "message": "No title/doi available for online validation.",
            "candidate": {},
        }
    candidates: list[tuple[str, dict[str, str]]] = []
    candidates.extend(_crossref_search_title(title, timeout_s=timeout_s))
    candidates.extend(_openalex_search_title(title, timeout_s=timeout_s))
    if not candidates:
        return {
            "checked": True,
            "verified": False,
            "provider": "",
            "score": 0.0,
            "message": "No candidate match returned by citation APIs.",
            "candidate": {},
        }
    best_provider = ""
    best_candidate: dict[str, str] = {}
    best_score = -1.0
    for provider, candidate in candidates:
        score = _score_candidate(title=title, year=year, doi=doi, candidate=candidate)
        if score > best_score:
            best_score = score
            best_provider = provider
            best_candidate = candidate
    return {
        "checked": True,
        "verified": bool(best_score >= 0.72),
        "provider": best_provider,
        "score": float(max(0.0, best_score)),
        "message": (
            "Citation metadata validated."
            if best_score >= 0.72
            else "Citation metadata weakly matched external APIs."
        ),
        "candidate": best_candidate,
    }


def merge_validated_fields(
    fields: dict[str, str],
    validation: dict[str, object],
    *,
    replace_when_confident: bool = True,
) -> dict[str, str]:
    out = dict(fields)
    candidate = dict(validation.get("candidate") or {})
    score = float(validation.get("score") or 0.0)
    provider = str(validation.get("provider") or "").strip().lower()
    existing_doi = str(out.get("doi") or "").strip().lower()
    candidate_doi = str(candidate.get("doi") or "").strip().lower()
    doi_exact = bool(existing_doi and candidate_doi and existing_doi == candidate_doi)
    replace = bool(replace_when_confident and (score >= 0.85 or doi_exact))
    scholar_prefer = provider == "google_scholar" and doi_exact
    keys = list(candidate.keys()) or [
        "title",
        "author",
        "year",
        "doi",
        "url",
        "journal",
    ]
    for key in keys:
        if str(key).startswith("__"):
            continue
        value = str(candidate.get(key) or "").strip()
        if not value:
            continue
        existing = str(out.get(key) or "").strip()
        if not existing or replace or scholar_prefer:
            out[key] = value
    return out


def validate_basic_citation_fields(fields: dict[str, str]) -> list[str]:
    errors: list[str] = []
    year = str(fields.get("year") or "").strip()
    if year:
        if not year.isdigit():
            errors.append("year must be numeric (YYYY).")
        else:
            year_value = int(year)
            current_year = int(datetime.now().year)
            if year_value < 1900 or year_value > current_year + 1:
                errors.append(f"year must be between 1900 and {current_year + 1}.")
    doi = str(fields.get("doi") or "").strip()
    if doi:
        if not _looks_like_doi(doi):
            errors.append("doi format looks invalid.")
    key = str(fields.get("__key__") or "").strip()
    if key and any(ch.isspace() for ch in key):
        errors.append("citation key should not contain whitespace.")
    return errors


def _serialize_entry(entry: BibEntry) -> str:
    lines = [f"@{entry.entry_type}{{{entry.key},"]
    for field_name, value in entry.fields.items():
        clean_name = field_name.strip().lower()
        clean_value = _strip_wrappers(value.strip())
        lines.append(f"  {clean_name} = {{{clean_value}}},")
    lines.append("}")
    return "\n".join(lines)


def _looks_like_doi(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    import re

    return re.fullmatch(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", text) is not None


def _normalize_text(value: str) -> str:
    import re

    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").lower()))


def _normalize_title_for_matching(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(
        r"\s*[-:|]\s*(pmc|pubmed|arxiv|bioarxiv|medrxiv)\b.*$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    return text


def _score_candidate(
    *, title: str, year: str, doi: str, candidate: dict[str, str]
) -> float:
    title_norm = _normalize_text(_normalize_title_for_matching(title))
    cand_title_norm = _normalize_text(
        _normalize_title_for_matching(candidate.get("title", ""))
    )
    title_score = (
        difflib.SequenceMatcher(None, title_norm, cand_title_norm).ratio()
        if title_norm and cand_title_norm
        else 0.0
    )
    year_score = 0.0
    if year and candidate.get("year"):
        year_score = 1.0 if str(year) == str(candidate.get("year")) else 0.0
    doi_score = 0.0
    cand_doi = str(candidate.get("doi") or "").lower()
    if doi and cand_doi:
        doi_score = 1.0 if str(doi).lower() == cand_doi else 0.0
    author_score = 0.0
    if candidate.get("author"):
        author_score = 0.6
    return (
        (0.65 * title_score)
        + (0.2 * year_score)
        + (0.1 * doi_score)
        + (0.05 * author_score)
    )


def _http_json(
    url: str, *, timeout_s: float, email: str | None = None
) -> dict[str, object]:
    headers = {
        "User-Agent": "annolid-citation-validator/1.0 (https://github.com/annolid)",
        "Accept": "application/json",
    }
    if email:
        headers["User-Agent"] = f"annolid-citation-validator/1.0 (mailto:{email})"

    req = urllib.request.Request(
        url=url,
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=max(0.5, float(timeout_s))) as response:
        payload = response.read()
    decoded = payload.decode("utf-8", errors="replace")
    data = json.loads(decoded)
    return data if isinstance(data, dict) else {}


def _http_text(url: str, *, timeout_s: float) -> str:
    req = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
            ),
            "Accept": "text/html,*/*;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=max(0.5, float(timeout_s))) as response:
        payload = response.read()
    return payload.decode("utf-8", errors="replace")


def _google_scholar_lookup(
    *, title: str, doi: str, timeout_s: float
) -> dict[str, object]:
    query_candidates = [doi, title]
    for query in query_candidates:
        q = str(query or "").strip()
        if not q:
            continue
        result = _google_scholar_lookup_query(query=q, timeout_s=timeout_s)
        if result.get("ok"):
            return result
    return {"ok": False, "candidate": {}}


def _google_scholar_lookup_query(*, query: str, timeout_s: float) -> dict[str, object]:
    try:
        base = "https://scholar.google.com"
        search_url = f"{base}/scholar?hl=en&q={quote(query)}"
        search_html = _http_text(search_url, timeout_s=timeout_s)
        cite_rel = _extract_google_scholar_cite_link(search_html)
        if not cite_rel:
            return {"ok": False, "candidate": {}}
        cite_url = urljoin(base, cite_rel)
        cite_html = _http_text(cite_url, timeout_s=max(0.5, timeout_s))
        bib_rel = _extract_google_scholar_bib_link(cite_html)
        if not bib_rel:
            return {"ok": False, "candidate": {}}
        bib_url = urljoin(base, bib_rel)
        bib_text = _http_text(bib_url, timeout_s=max(0.5, timeout_s))
        candidate = _candidate_from_bibtex(bib_text)
        return {"ok": bool(candidate), "candidate": candidate}
    except Exception:
        return {"ok": False, "candidate": {}}


def _extract_google_scholar_cite_link(text: str) -> str:
    decoded = html.unescape(str(text or ""))
    patterns = [
        r'href="(/scholar\?[^"]*output=cite[^"]*)"',
        r"href='(/scholar\?[^']*output=cite[^']*)'",
    ]
    for pattern in patterns:
        match = re.search(pattern, decoded, flags=re.IGNORECASE)
        if match:
            return str(match.group(1) or "").strip()
    return ""


def _extract_google_scholar_bib_link(text: str) -> str:
    decoded = html.unescape(str(text or ""))
    patterns = [
        r'href="([^"]*scholar\.bib[^"]*)"',
        r"href='([^']*scholar\.bib[^']*)'",
    ]
    for pattern in patterns:
        match = re.search(pattern, decoded, flags=re.IGNORECASE)
        if match:
            return str(match.group(1) or "").strip()
    return ""


def _candidate_from_bibtex(text: str) -> dict[str, str]:
    entries = parse_bibtex(str(text or ""))
    if not entries:
        return {}
    first = entries[0]
    fields = dict(first.fields)
    candidate = {
        "__bibkey__": str(first.key or "").strip(),
        "title": str(fields.get("title") or "").strip(),
        "author": str(fields.get("author") or "").strip(),
        "year": str(fields.get("year") or "").strip(),
        "doi": str(fields.get("doi") or "").strip(),
        "url": str(fields.get("url") or "").strip(),
        "journal": str(fields.get("journal") or fields.get("booktitle") or "").strip(),
        "volume": str(fields.get("volume") or "").strip(),
        "number": str(fields.get("number") or "").strip(),
        "pages": str(fields.get("pages") or "").strip(),
        "publisher": str(fields.get("publisher") or "").strip(),
    }
    return {k: v for k, v in candidate.items() if str(v).strip()}


def _crossref_lookup_doi(doi: str, *, timeout_s: float) -> dict[str, object]:
    try:
        data = _http_json(
            f"https://api.crossref.org/works/{quote(doi, safe='')}",
            timeout_s=timeout_s,
            email="support@annolid.com",
        )
        message = dict(data.get("message") or {})
        candidate = _crossref_message_to_candidate(message)
        return {"ok": bool(candidate), "candidate": candidate}
    except Exception:
        return {"ok": False, "candidate": {}}


def _crossref_search_title(
    title: str, *, timeout_s: float
) -> list[tuple[str, dict[str, str]]]:
    try:
        data = _http_json(
            f"https://api.crossref.org/works?query.bibliographic={quote(title)}&rows=5",
            timeout_s=timeout_s,
            email="support@annolid.com",
        )
        message = dict(data.get("message") or {})
        items = list(message.get("items") or [])
        out: list[tuple[str, dict[str, str]]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            candidate = _crossref_message_to_candidate(item)
            if candidate:
                out.append(("crossref", candidate))
        return out
    except Exception:
        return []


def _crossref_message_to_candidate(message: dict[str, object]) -> dict[str, str]:
    title_values = list(message.get("title") or [])
    title = str(title_values[0] if title_values else "").strip()
    issued = dict(message.get("issued") or {})
    date_parts = list(issued.get("date-parts") or [])
    year = ""
    if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
        year = str(date_parts[0][0])
    author_values = list(message.get("author") or [])
    author_names: list[str] = []
    for person in author_values:
        if not isinstance(person, dict):
            continue
        family = str(person.get("family") or "").strip()
        given = str(person.get("given") or "").strip()
        literal = str(person.get("name") or "").strip()
        if family or given:
            author_names.append(f"{family}, {given}".strip(", "))
        elif literal:
            author_names.append(literal)
    author = " and ".join([name for name in author_names if name])
    doi = str(message.get("DOI") or "").strip()
    url = str(message.get("URL") or "").strip()
    container = list(message.get("container-title") or [])
    journal = str(container[0] if container else "").strip()
    volume = str(message.get("volume") or "").strip()
    number = str(message.get("issue") or "").strip()
    pages = str(message.get("page") or "").strip()
    publisher = str(message.get("publisher") or "").strip()
    candidate = {
        "title": title,
        "year": year,
        "author": author,
        "doi": doi,
        "url": url,
        "journal": journal,
        "volume": volume,
        "number": number,
        "pages": pages,
        "publisher": publisher,
    }
    return {k: v for k, v in candidate.items() if str(v).strip()}


def _openalex_search_title(
    title: str, *, timeout_s: float
) -> list[tuple[str, dict[str, str]]]:
    try:
        data = _http_json(
            f"https://api.openalex.org/works?search={quote(title)}&per-page=5",
            timeout_s=timeout_s,
        )
        results = list(data.get("results") or [])
        out: list[tuple[str, dict[str, str]]] = []
        for item in results:
            if not isinstance(item, dict):
                continue
            title_value = str(item.get("title") or "").strip()
            year = str(item.get("publication_year") or "").strip()
            doi_url = str(item.get("doi") or "").strip()
            doi = doi_url.split("doi.org/")[-1] if "doi.org/" in doi_url else doi_url
            source = dict(item.get("primary_location") or {}).get("source")
            journal = ""
            if isinstance(source, dict):
                journal = str(source.get("display_name") or "").strip()
            author_names: list[str] = []
            authorships = list(item.get("authorships") or [])
            for authorship in authorships:
                if not isinstance(authorship, dict):
                    continue
                author_obj = dict(authorship.get("author") or {})
                display = str(author_obj.get("display_name") or "").strip()
                if display:
                    author_names.append(display)
            author = " and ".join([name for name in author_names if name])
            biblio = dict(item.get("biblio") or {})
            candidate = {
                "title": title_value,
                "year": year,
                "doi": doi,
                "url": doi_url,
                "journal": journal,
                "author": author,
                "volume": str(biblio.get("volume") or "").strip(),
                "number": str(biblio.get("issue") or "").strip(),
                "pages": (
                    "--".join(
                        [
                            str(biblio.get("first_page") or "").strip(),
                            str(biblio.get("last_page") or "").strip(),
                        ]
                    ).strip("-")
                ),
            }
            cleaned = {k: v for k, v in candidate.items() if str(v).strip()}
            if cleaned:
                out.append(("openalex", cleaned))
        return out
    except Exception:
        return []


def _parse_fields(text: str) -> dict[str, str]:
    parts = _split_top_level(text, ",")
    fields: dict[str, str] = {}
    for part in parts:
        if not part.strip():
            continue
        name, value = _split_top_level_once(part, "=")
        if name is None or value is None:
            continue
        field_name = name.strip().lower()
        if not field_name:
            continue
        fields[field_name] = _strip_wrappers(value.strip())
    return fields


def _strip_wrappers(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] == "{" and v[-1] == "}" and _is_balanced_braces(v):
        return v[1:-1].strip()
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        return v[1:-1].strip()
    return v


def _is_balanced_braces(value: str) -> bool:
    depth = 0
    in_quote = False
    escape = False
    for ch in value:
        if in_quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_quote = False
            continue
        if ch == '"':
            in_quote = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and not in_quote


def _split_top_level_once(text: str, separator: str) -> tuple[str | None, str | None]:
    depth = 0
    in_quote = False
    escape = False
    for idx, ch in enumerate(text):
        if in_quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_quote = False
            continue
        if ch == '"':
            in_quote = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth = max(0, depth - 1)
            continue
        if ch == separator and depth == 0:
            return text[:idx], text[idx + 1 :]
    return text, None


def _split_top_level(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    in_quote = False
    escape = False
    for idx, ch in enumerate(text):
        if in_quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_quote = False
            continue
        if ch == '"':
            in_quote = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth = max(0, depth - 1)
            continue
        if ch == separator and depth == 0:
            parts.append(text[start:idx])
            start = idx + 1
    parts.append(text[start:])
    return parts


def _find_matching(text: str, start: int, open_char: str, close_char: str) -> int:
    depth = 0
    in_quote = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_quote = False
            continue
        if ch == '"':
            in_quote = True
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return idx
    return -1
