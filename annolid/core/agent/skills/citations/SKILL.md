---
name: citations
description: Manage BibTeX citation files (.bib) for papers and references, and assist in research writing.
metadata: '{"annolid":{"requires":{"bins":[]}}}'
---

Use this skill when users ask to organize, search, add, update, or remove paper citations, OR when the user asks you to write a research paper / report.

Preferred tools:

1. `bibtex_list_entries` to inspect or search existing entries.
2. `bibtex_upsert_entry` to create or update entries by key.
3. `bibtex_remove_entry` to delete stale entries.

## Citation Guidelines

1. **Keep Keys Stable:** Keep citation keys stable unless the user asks to rename them.
2. **Require Essential Fields:** Try to populate at least `title`, `author`, and `year` when adding papers.
3. **No Destructive Rewrites:** Avoid destructive rewrites to the .bib file unrelated to the requested changes.
4. **Validation:** Try Google Scholar BibTeX, then fallback APIs natively built into the parsing util before hard-saving if unsure.

## Research Paper Writing Guidelines

When writing research drafts, reports, or documentation:

1. **Inline Citations:** When mentioning external papers or facts, always use bracketed inline citation keys (e.g., `[@author2024]`).
2. **Proactive Management:** If you mention a paper in your writing that does not exist in the `.bib` file, proactively find its metadata using web search or Crossref and add it to the `.bib` file using the citation tools.
3. **Drafting Format:** Output plain markdown or LaTeX as requested. Include a `## References` section placeholder at the bottom if using plain markdown.

User-friendly command examples in chat input:

1. `save citation`
2. `save citation from pdf as annolid2024 to references.bib`
3. `save citation from web`
4. `add citation @article{yang2024annolid, title={Annolid: Annotate...}`
5. `write a paragraph about instance segmentation and cite relevant papers`
