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
4. Lookup order for reliability:
   - First try embedded browser weather lookup (`gui_web_run_steps`).
   - If unavailable, use `web_search` (Scrapling-first backend).
   - If a direct weather URL is available, use `web_fetch` for extraction.
5. If model output is empty or says it cannot browse, immediately run a web lookup tool instead of returning a refusal.
