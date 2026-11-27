---
name: weather
description: Look up current weather conditions, temperatures, precipitation, and short forecasts for any location.
metadata: '{"annolid":{"always":false,"tags":["weather","forecast","temperature","rain","snow","wind","climate"]}}'
---

# Weather Lookup

Use this skill for weather and forecast requests.

## Guidelines

1. Use explicit locations (city, state/region, country) when possible.
2. Return concise forecast summaries (temperature with units, precipitation chance/type, wind, and local date/time context).
3. Do not reply with a promise to check later—execute a lookup tool immediately and return the retrieved forecast.
4. Lookup order for reliability:
   - First use `web_search` (DDGS with hardened DuckDuckGo HTML and optional Brave fallbacks).
   - If a direct weather URL is returned or known, use `web_fetch(url, extractMode="text")` to extract details.
   - If search is unavailable in a GUI session, use the embedded browser workflow (`gui_web_run_steps`).
5. If an initial lookup is sparse or ambiguous, refine the location query and retry once rather than refusing.
