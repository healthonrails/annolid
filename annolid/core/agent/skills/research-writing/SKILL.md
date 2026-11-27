---
name: research-writing
description: Conduct literature searches, build research packets, run novelty checks, draft structured manuscripts with verified citations, and generate paper-ready evaluation gates.
metadata: '{"annolid":{"always":false,"tags":["research","paper","manuscript","literature","literature review","novelty","draft paper","paper ready","arxiv","peer review","assess draft","academic writing"]}}'
---

# Research & Manuscript Writing

Use this skill when the user wants to perform a literature review, check novelty of a research idea, assemble a research packet, draft a scientific manuscript or methods/results section, assess draft quality, or validate paper readiness.

## End-to-End Tool Pipeline

1. **Literature Discovery**:
   - `search_literature(query, max_results?)`: query scholarly literature cleanly.
   - `gui_arxiv_search(query, max_results?)`: search arXiv directly from the GUI.
   - `gui_pdf_summarize()` / `extract_pdf_text(path, ...)`: read and summarize local PDFs or open papers.
2. **Research Packet & Novelty Audit**:
   - `build_research_packet(topic, idea_summary?, related_work?, max_results?)`: synthesize structured related-work context and gaps.
   - `annolid_novelty_check(idea_title, idea_summary, related_work?, related_work_json_path?, top_k?)`: measure overlap against prior work and classify differentiate vs abort thresholds.
3. **Structured Drafting & Critique**:
   - `draft_paper_swarm(topic, idea_summary?, related_work?, output_dir?, report_basename?, allow_mutation=true)`: multi-role drafting workflow producing structured manuscript artifacts.
   - `assess_paper_draft(paper_path?, paper_text?, references?, bib_path?)`: critique clarity, claims-to-evidence grounding, citation coverage, and structural gaps.
4. **Citation & Paper-Readiness Gates**:
   - Manage and verify `.bib` references with `bibtex_list_entries`, `bibtex_upsert_entry`, `gui_save_citation`, and `gui_verify_citations` (see the `citations` skill).
   - `annolid_paper_run_report(eval_report_json_path?, citation_report_path?, novelty_report_path?, draft_quality_report_path?, paper_draft_path?, bib_path?, paper_ready_gate=true, allow_mutation=true)`: produce a consolidated readiness report enforcing citation integrity and novelty coverage floors.

## Scientific Writing Guardrails

1. **Zero Fabricated Citations or Numbers**: Every metric must trace to a real evaluation/analysis artifact; every cited paper must come from verified search/BibTeX metadata.
2. **Separate Empirical Results from Claims**: Keep observed metrics, statistical caveats (QC warnings, confidence intervals, multiple-testing $q$-values), and limitations transparent.
3. **Explicit File Outputs**: Pass `allow_mutation=true` only when writing drafts/reports to disk, and always return the exact output paths (`.md`, `.bib`, `.json`).
