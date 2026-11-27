---
name: experimental-design
description: Keep behavior analysis aligned with assay objectives, controls, blinding, pseudoreplication guards, and reproducible cohort statistics instead of ad hoc summaries.
metadata: '{"annolid":{"always":false,"tags":["experimental design","experiment","cohort","blinding","pseudoreplication","irr","fdr","statistics","hypothesis","controls","survival","log-rank"]}}'
---

# Experimental Design

Use this skill when connecting Annolid tracking and behavior outputs to rigorous experimental hypotheses, blinding/unblinding workflows, and multi-animal cohort comparisons.

## Statistical & Design Guardrails in Annolid

1. **Prevent Dyadic Pseudoreplication (`--unit recording`)**:
   - When multiple interacting animals share the same arena/cage (`resident` + `intruder`, dyads, triads), subject-level rows within a recording are non-independent.
   - Use `annolid_behavior_cohort` with `analysis_unit="recording"` (or `annolid-run behavior-cohort ... --unit recording`) so recordings—not co-housed animals—serve as the independent experimental unit.
2. **Blinded Scoring & Manifest Unblinding**:
   - Score videos blinded to treatment group, validate scorer consistency via `annolid_behavior_irr` (`annolid-run behavior-irr` — target Cohen's $\kappa \ge 0.70$ or Krippendorff's $\alpha \ge 0.67$), and unblind only at cohort aggregation via `--manifest` or `--metadata-csv`.
3. **Right-Censored Latencies & Zero-Occurrence Denominators**:
   - Animals that never perform a behavior during a session must contribute `0` occurrences/duration to group denominators (never be dropped!) and have right-censored first-bout latency analyzed via Kaplan-Meier median survival and Mantel-Cox log-rank tests (`CohortComparisonRow.logrank_p_value`).
4. **Multiple-Testing Control & Effect Sizes**:
   - Report Benjamini-Hochberg FDR-adjusted $q$-values (`q_value_bh`) across behaviors alongside standardized effect sizes (Hedges' $g$, Cliff's $\delta$, and bootstrap 95% CIs).

## Guidelines

1. Prefer pre-specified primary endpoints (e.g., % time in open arms, social preference index, attack latency) over post-hoc fishing.
2. Require identical `fps`, `session_duration`, and `px_per_cm` calibration across treatment and control arms.
3. Run `annolid_track_qc` and `annolid_behavior_qc` before running inferential group tests.
