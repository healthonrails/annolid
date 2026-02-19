from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from annolid.core.agent.tools.pdf import DownloadPdfTool


async def safe_run_arxiv_search(
    *,
    query: str,
    run_arxiv_search: Callable[..., Awaitable[Dict[str, Any]]],
    emit_progress: Callable[[str], None],
    log_error: Callable[[str], None],
) -> None:
    try:
        result = await run_arxiv_search(query=query)
        if not result.get("ok"):
            error_msg = result.get("error", "Unknown error")
            emit_progress(f"arXiv download failed: {error_msg}")
        else:
            open_result = result.get("open_result", {})
            if not open_result.get("ok"):
                emit_progress(
                    f"Downloaded but failed to open: {open_result.get('error')}"
                )
            else:
                emit_progress("Opened PDF successfully.")
    except Exception as exc:
        log_error(str(exc))
        emit_progress(f"Error during arXiv operation: {str(exc)}")


def list_local_pdfs(
    *,
    workspace: Path,
    query: Optional[str] = None,
    max_results: int = 20,
) -> Dict[str, Any]:
    search_dirs = [workspace, workspace / "downloads"]
    found_files: List[Path] = []
    for item in search_dirs:
        if item.exists() and item.is_dir():
            found_files.extend(list(item.glob("*.pdf")))

    unique_files = {str(f.absolute()): f for f in found_files}
    pdf_files = list(unique_files.values())
    if query:
        q = str(query).lower().strip()
        pdf_files = [f for f in pdf_files if q in f.name.lower()]
    pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    max_items = max(1, int(max_results or 20))
    truncated = len(pdf_files) > max_items
    pdf_files = pdf_files[:max_items]

    rel_paths = []
    for item in pdf_files:
        try:
            rel_paths.append(str(item.relative_to(workspace)))
        except ValueError:
            rel_paths.append(str(item))

    return {
        "ok": True,
        "files": rel_paths,
        "count": len(found_files),
        "showing": len(rel_paths),
        "truncated": truncated,
    }


async def arxiv_search_tool(
    *,
    query: str,
    max_results: int,
    workspace: Path,
    emit_progress: Callable[[str], None],
    open_pdf: Callable[[str], Awaitable[Dict[str, Any]]],
) -> Dict[str, Any]:
    try:
        emit_progress(f"Searching arXiv for '{query}'...")
        base_url = "http://export.arxiv.org/api/query"
        final_query = query if query.startswith("id:") else f"all:{query}"
        safe_query = urllib.parse.quote(final_query)
        url = f"{base_url}?search_query={safe_query}&start=0&max_results={max_results}"
        emit_progress(f"arXiv: {query}")
        loop = asyncio.get_running_loop()

        def fetch_feed():
            with urllib.request.urlopen(url, timeout=10) as response:
                return response.read()

        xml_data = await loop.run_in_executor(None, fetch_feed)
        emit_progress("Metadata received. Parsing...")
        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is None:
            return {
                "ok": False,
                "error": f"No papers found for query: {query}",
                "query": query,
            }

        title_elem = entry.find("atom:title", ns)
        title = (title_elem.text or "Untitled").strip().replace("\n", " ")
        id_elem = entry.find("atom:id", ns)
        id_url = (id_elem.text or "").strip()
        pdf_link = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_link = link.attrib.get("href")
                break
        if not pdf_link and id_url:
            arxiv_id = id_url.split("/")[-1]
            pdf_link = f"http://arxiv.org/pdf/{arxiv_id}.pdf"
        if not pdf_link:
            return {
                "ok": False,
                "error": "Could not find PDF link for paper.",
                "title": title,
            }

        emit_progress(f"Found: {title}")
        emit_progress("Downloading PDF...")
        downloads_dir = workspace / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c for c in title if c.isalnum() or c in " ._-").strip()
        if len(safe_title) > 100:
            safe_title = safe_title[:100].rstrip(" ._-")
        filename = f"{safe_title}.pdf"
        output_path = str(downloads_dir / filename)

        downloader = DownloadPdfTool(allowed_dir=workspace)
        dl_result_json = await downloader.execute(
            url=pdf_link, output_path=output_path, overwrite=True
        )
        try:
            dl_result = json.loads(dl_result_json)
        except Exception:
            dl_result = {"error": "Invalid download response"}
        if dl_result.get("error"):
            return {
                "ok": False,
                "error": f"Download failed: {dl_result['error']}",
                "url": pdf_link,
            }

        final_path = dl_result.get("output_path", output_path)
        emit_progress("Opening PDF...")
        open_res = await open_pdf(final_path)
        return {"ok": True, "title": title, "path": final_path, "open_result": open_res}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
