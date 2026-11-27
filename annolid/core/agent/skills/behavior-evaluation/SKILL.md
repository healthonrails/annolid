---
name: behavior-evaluation
description: Evaluate behavior classifiers, pose/detection models, tracking outputs, and ethograms using Annolid's deterministic eval, QC, and inter-rater reliability tools.
metadata: '{"annolid":{"always":false,"tags":["evaluation","eval","behavior evaluation","metrics","macro f1","confusion matrix","track qc","behavior qc","qc","irr","cohen kappa","krippendorff","tracking stats","paper report"]}}'
---

# Behavior & Model Evaluation

Use this skill when the user asks to evaluate a trained behavior classifier, audit tracking or ethogram quality, measure inter-rater reliability (IRR), or generate an evaluation report.

## Choose the Right Evaluation Tool

1. **Trained Model Evaluation (`annolid_eval_start` + `annolid_eval_report`)**:
   - Start evaluation (mutating — set `allow_mutation=true`; use `background=true` for long runs):
     - `annolid_eval_start(model_family="behavior_classifier"|"dino_kpseg"|"yolo", weights="...", data="...", split="val"|"test", allow_mutation=true)`
     - For `behavior_classifier`, supply `video_folder` / `checkpoint_path` (plus `feature_backbone`, `val_ratio`, `random_seed`, `plot_dir` as needed).
   - Build bootstrap confidence intervals and formatted Markdown/JSON evaluation summaries:
     - `annolid_eval_report(path="...", model_family="...", dataset_name="...", model_name="...", split="test", allow_mutation=true)`
2. **Tracking & Trajectory Quality Control (`annolid_track_qc`, `annolid_tracking_stats`)**:
   - `annolid_track_qc(input_path="...", expected_instances?, fps?, top_k_review?, out_json?, issues_csv?, review_csv?)`: detects frame gaps, implausible velocity jumps, and likely ID swaps, returning prioritized manual review bouts.
   - `annolid_tracking_stats(root_dir="...", output_dir?, include_plots?)` or GUI `gui_analyze_tracking_stats(...)`: coverage and trajectory summary across recordings.
3. **Ethogram Quality & Inter-Rater Reliability (`annolid_behavior_qc`, `annolid_behavior_irr`)**:
   - `annolid_behavior_qc(events_csv="...", schema_path?, strict_schema?)`: validates `START`/`STOP` pairing, timestamps, and mutual-exclusion rules.
   - `annolid_behavior_irr(events_csvs=[...], rater_names?, bin_seconds?, onset_tolerance_seconds?, pool_subjects?, schema_path?)`: computes Cohen's $\kappa$ (with bootstrap 95% CI), Fleiss' $\kappa$, Krippendorff's $\alpha$, bout-level F1, and top disagreement windows.
4. **Publication / Paper-Ready Gate (`annolid_paper_run_report`)**:
   - `annolid_paper_run_report(eval_report_json_path="...", citation_report_path?, novelty_report_path?, paper_ready_gate=true, allow_mutation=true)`

## Evaluation Rules

1. **Evaluate on Held-Out Recordings**: Never report training-split accuracy as model performance. For polygon/TCN classifiers, report held-out `macro_f1`, `weighted_f1`, per-class precision/recall, and confusion matrix alongside class support counts.
2. **Check Imbalance & Rare Behaviors**: Accuracy alone hides failures on sparse behaviors; always include macro-F1 and per-class recall.
3. **Surface Quality Gates Explicitly**: Always state `PASS` / `WARN` / `FAIL` outcomes, missing frames, unpaired events, and calibration status (`fps`, `px_per_cm`).
4. **Preserve Provenance**: Cite exact checkpoint paths, dataset splits, random seeds, and generated report files.
