---
name: citations
description: Manage BibTeX citation files (.bib), verify references against scholarly metadata, and cite literature accurately in papers and reports.
metadata: '{"annolid":{"always":false,"tags":["citations","bibtex","bib","references","bibliography","arxiv","crossref","doi","verify citations","paper"]}}'
---

# Citations & BibTeX Management

Use this skill when users ask to organize, search, add, update, verify, or remove paper citations in `.bib` files, or manage references while writing papers and reports.

## Preferred Tools

1. **Direct `.bib` file management**:
   - `bibtex_list_entries(path, query?, field?, limit?)`: inspect or search existing entries.
   - `bibtex_upsert_entry(path, key, entry_type?, fields, sort_keys?)`: create or update entries by citation key.
   - `bibtex_remove_entry(path, key, sort_keys?)`: delete stale entries.
2. **GUI-assisted capture & verification**:
   - `gui_save_citation(key?, bib_file?, source?, entry_type?, validate_before_save?, strict_validation?, verify_after_save?)`: extract and save citations from the open PDF, open web page, or chat context (`source="pdf"|"web"|"auto"`).
   - `gui_verify_citations(bib_file?, limit?)`: audit `.bib` entries for missing fields, hallucinated titles/DOIs, or mismatched metadata.
3. **Literature lookup**:
   - `search_literature(query, max_results?)`: query scholarly metadata before inventing or guessing citation details.
   - `gui_arxiv_search(query, max_results?)`: search arXiv directly from the GUI workflow.

## Citation Guidelines

1. **Keep Keys Stable:** Keep citation keys stable unless the user asks to rename them.
2. **Require Essential Fields:** Populate at least `title`, `author`, and `year` (plus `journal`/`booktitle`, `doi`, or `url` when available).
3. **No Destructive Rewrites:** Avoid unrelated edits to other entries in the `.bib` file.
4. **Verify Before Finalizing:** Never fabricate authors, DOIs, or publication years. Validate with `search_literature` or `gui_verify_citations` before declaring references ready.

## Research Writing Citation Conventions

1. **Inline Citations:** Cite external claims with bracketed keys (e.g., `[@author2024]`) or LaTeX `\cite{author2024}` as requested.
2. **Proactive `.bib` Sync:** If a cited paper is missing from the target `.bib` file, look up verified metadata first and upsert it via `bibtex_upsert_entry` or `gui_save_citation`.
3. **Drafting Format:** Match the user's format (Markdown with a `## References` section, or LaTeX). For full multi-section paper drafting and paper-readiness gates, pair with the `research-writing` skill.

## User-Friendly Chat Examples

1. `save citation`
2. `save citation from pdf as annolid2024 to references.bib`
3. `save citation from web`
4. `verify citations in references.bib`
5. `add citation @article{yang2024annolid, title={Annolid: Annotate...}}`
