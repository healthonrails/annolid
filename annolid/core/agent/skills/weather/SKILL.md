---
name: weather
description: Get weather conditions and short forecasts.
metadata: '{"annolid":{"requires":{"bins":["curl"]}}}'
---

Use this skill for weather lookup tasks.

Guidelines:

1. Use explicit locations (city/state/country) when possible.
2. Prefer concise forecast summaries (temperature, precipitation, wind).
3. Include date/time context and units.
4. Do not answer with a promise to check later. Use the lookup tools immediately, then return the weather result.
5. Lookup order for reliability:
   - First use `web_search` (DDGS with hardened DuckDuckGo HTML and optional Brave fallbacks).
   - If search is unavailable, try the embedded browser workflow (`gui_web_run_steps`).
   - If a direct weather URL is available, use `web_fetch` for extraction.
6. If model output is empty or says it cannot browse, immediately run a web lookup tool instead of returning a refusal.
