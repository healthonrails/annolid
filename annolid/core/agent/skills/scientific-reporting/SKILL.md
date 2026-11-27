---
name: scientific-reporting
description: Write behavior-analysis summaries and publication-grade reports in a concise scientific style with methods, findings, QC caveats, and artifact-backed conclusions.
metadata: '{"annolid":{"always":false,"tags":["scientific report","report","analysis report","findings","methods","caveats","qc","track qc","behavior qc","cohort","irr","publication","manifest"]}}'
---

# Scientific Reporting

Use this skill when the user wants an interpretable, publication-grade analysis summary backed by reproducible Annolid artifacts.

## Built-In Annolid Reporting & QC Tools

Prefer calling Annolid's deterministic tools (or GUI `/report`, `/qc`, `/stats` slash commands) before drafting narrative text:

- `annolid_generate_report` (`annolid-run report <source> [--events-csv ...] [--zone-file ...] [--px-per-cm ...]`): builds a self-contained `analysis_report.html` (with inline SVG trajectory, freezing budget, zone occupancy, and ethogram raster charts), `analysis_report.md`, and `analysis_report.json` (`unified_analysis_report/v1`).
- `annolid_tracking_stats` / `gui_analyze_tracking_stats`: summarizes tracking datasets, frame coverage, instance counts, and trajectory metrics.
- `annolid_track_qc` (`annolid-run track-qc <input>`): audits tracking coverage, velocity spikes, identity swaps, and prioritized manual review bouts (`track_qc/v1`).
- `annolid_behavior_qc` (`annolid-run behavior-qc <events_csv>`): checks unpaired state intervals and mutual-exclusion violations (`behavior_qc_report/v1`).
- `annolid_behavior_cohort` (`annolid-run behavior-cohort --manifest ... --unit recording|subject`): computes group summaries, Welch/ANOVA & Mann-Whitney/Kruskal-Wallis tests, Hedges' $g$, Cliff's $\delta$, Kaplan-Meier censored latency + log-rank $p$, and Benjamini-Hochberg FDR $q$-values (`behavior_cohort/v1`).
- `annolid_behavior_irr` (`annolid-run behavior-irr rater1.csv rater2.csv`): computes Cohen's $\kappa$ (with bootstrap 95% CI), Fleiss' $\kappa$, Krippendorff's $\alpha$, bout F1, and disagreement windows (`behavior_irr_report/v1`).
- `annolid_eval_report` & `annolid_paper_run_report`: package model evaluation metrics, citation integrity, and paper-readiness gates for manuscript drafts.

## Structure

- **Assay & Recording Context**: frame count, FPS, spatial calibration (`px_per_cm` vs uncalibrated pixels), and overall Quality Gate (`PASS` / `WARN` / `FAIL`).
- **Methods & Statistical Unit**: exact denominator (session duration vs union of intervals), experimental unit (`subject` vs `recording`), and multiple-testing correction (`q_value_bh`).
- **Key Findings**: report both parametric (mean ± SD/SEM, Hedges' $g$) and non-parametric (median [IQR], Cliff's $\delta$) summaries alongside 95% CIs.
- **Caveats & Data Quality**: surface tracking gaps, unresolved bad-shape events, censored latencies, or pseudoreplication warnings explicitly.
- **Reproducible Artifacts**: cite the exact generated `.html`, `.md`, `.csv`, and `.json` paths.

## Rules

1. Ground every numeric claim in tool outputs or stored artifacts; never invent numbers.
2. Keep tone scientific, objective, and concise.
3. Separate empirical observations from biological interpretation.
4. Never conceal QC warnings, missing frames, or uncalibrated pixel units.
