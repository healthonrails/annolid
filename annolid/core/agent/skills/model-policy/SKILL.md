---
name: model-policy
description: Choose behavior-analysis models and runtimes with explicit tradeoffs around reproducibility, cost, privacy, speed, and artifact compatibility.
metadata: '{"annolid":{"always":false,"tags":["model policy","llm policy","local model","ollama","hosted","privacy","offline","cost","reproducibility","runtime"]}}'
---

# Model Policy

Use this skill when selecting between hosted reasoning models, local/offline models (e.g., Ollama), existing Annolid perception backends, or heavier optional pipelines.

## Canonical Annolid Policies

- Default hybrid policy: `hosted_reasoning_local_tracking_v1` (local Annolid trackers/analytics for deterministic geometry + hosted VLM/LLM for high-level planning and interpretation).
- Configurable via `gui_score_aggression_bouts(model_policy=...)`, `annolid-run agent-behavior --model-policy ...`, or `llm_profile` in `gui_label_behavior_segments` / `sam3_agent_video_track`.
- Local-only / privacy-preserving runs: keep all perception, kinematics, zone analysis, and reporting local; use local Ollama/vLLM endpoints when unpublished lab footage cannot leave the machine.

## Rules

1. Prefer existing deterministic Annolid perception and analytics backends for default execution.
2. Prefer typed, replayable outputs over one-off model calls.
3. Use hosted reasoning for planning and interpretation when it materially helps and privacy permits.
4. Keep privacy-sensitive or offline workflows compatible with local fallback.
5. Do not silently switch to a heavier stack or hosted API without explaining why.

## Output

State:

- selected policy (`model_policy` / backend),
- why it fits the task,
- what evidence/artifacts it is expected to produce,
- any cost, privacy, runtime, or reproducibility tradeoff.
