---
name: web-scraping
description: Extract and summarize content from the currently opened web view with robust fallback.
metadata: '{"annolid":{"always":true}}'
---

Use this skill when the user asks about the currently opened web page, says "this page/current page", or asks for web extraction/summarization.

Primary goal: return grounded page content quickly, with source links and minimal hallucination.

## Workflow

1. Check active page state first.
- Use `gui_web_get_state()`.
- If no active page, run a web lookup flow (`gui_web_run_steps` or `web_search`) and then continue.

2. Extract page text from the active web view.
- Use `gui_web_get_dom_text(max_chars=9000)` first.
- Prefer `gui_web_extract_structured(fields=[...])` for schema-shaped answers.
- When field-specific regex is brittle, provide `selector_hints` (CSS-like or keyword hints) and set `extraction_mode="hint"` for resilient line-level extraction.
- If content is too short/noisy, use `gui_web_run_steps` with `get_text` after a short `wait` and optional `scroll`.

3. Fallback to scraper/search tools when needed.
- Use `web_search(query, count)` (Scrapling-first backend).
- Use `web_fetch(url, extractMode="text")` when a target URL is known.
- Prefer returning the best available grounded result rather than refusal text.

4. Response style.
- Return concise extracted facts.
- Include source URL(s).
- If data is time-sensitive (weather, prices, news), clearly label it as current lookup output.

## Reliability rules

- If model output is empty or uncertain, execute tool fallback immediately.
- Do not claim browsing is unavailable before attempting available web tools.
- Prioritize the currently open page when the user references "this/current/open page".
