---
name: citations
description: Manage BibTeX citation files (.bib) for papers and references.
metadata: '{"annolid":{"requires":{"bins":[]}}}'
---

Use this skill when users ask to organize, search, add, update, or remove paper citations.

Preferred tools:

1. `bibtex_list_entries` to inspect or search existing entries.
2. `bibtex_upsert_entry` to create or update entries by key.
3. `bibtex_remove_entry` to delete stale entries.

Guidelines:

1. Keep citation keys stable unless the user asks to rename.
2. Require essential fields (`title`, `author`, `year`) when adding papers.
3. Avoid destructive rewrites unrelated to requested changes.
4. By default, try Google Scholar BibTeX lookup first, then fallback APIs, before saving.

User-friendly command examples in chat input:

1. `save citation`
2. `save citation from pdf as annolid2024 to references.bib`
3. `save citation from web`
