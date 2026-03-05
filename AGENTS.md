# AGENTS.md (Annolid)

This document defines the engineering operating standard for coding agents and contributors in this repository.

## Mission

Build and maintain Annolid as a reliable system for:

- annotation,
- segmentation,
- tracking,
- behavior analysis,
- practical model workflows in real research/lab conditions.

Quality order:

1. Correctness
2. Reproducibility
3. Usability (GUI and CLI)
4. Performance optimization

## Non-Negotiable Rules

1. Make minimal, surgical changes.
2. Preserve backward compatibility unless the user explicitly requests a behavior change.
3. Do not break GUI behavior to satisfy CLI assumptions, or vice versa.
4. Treat on-disk formats and user data as contracts.
5. Never commit secrets, credentials, or machine-local absolute paths.
6. Never claim checks/tests passed unless they were actually run.
7. Use `.venv` for local validation and testing.

## Engineering Principles (World-Class Baseline)

Apply these principles by default:

1. Make the correct thing easy and the wrong thing hard.
2. Optimize for long-term maintainability, not short-term cleverness.
3. Keep interfaces stable and internals evolvable.
4. Prefer explicit behavior over hidden magic.
5. Fail fast with actionable errors; recover gracefully where possible.
6. Measure before optimizing performance.
7. Design for observability: logs, metrics, and debuggable state transitions.

## Architecture and Design Practices

Before coding non-trivial changes:

1. Define invariants and contracts (input/output, side effects, persisted schema).
2. Identify blast radius (GUI, CLI, data format, workflows, release).
3. Choose the simplest design that preserves extension points.
4. Prefer composition over deep inheritance and global state.
5. Document tradeoffs when selecting one approach over alternatives.

Design checklist:

- Single responsibility per module/function.
- Clear boundaries between UI, domain logic, and IO.
- Idempotent operations where feasible.
- Deterministic behavior for CI-critical paths.

## Execution Loop (Required)

For every task:

1. Restate the user-visible goal in one sentence.
2. Find the true code path causing the behavior.
3. Apply the smallest correct fix.
4. Run targeted validation for touched surfaces.
5. Update documentation if behavior/flags/workflows changed.
6. Verify no unrelated files were edited.
7. Summarize changes, validations, and residual risks.

## Requirements and Acceptance Criteria

When the request is ambiguous, derive concrete acceptance criteria:

- user-visible behavior change,
- expected input/output examples,
- compatibility constraints,
- validation plan.

Do not ship changes without a testable acceptance target.

## Definition of Done

A task is done only when all are true:

- Behavior matches the request.
- Relevant checks were run, or skipped with explicit reason.
- User-facing changes are documented.
- No unrelated regressions were introduced.
- Final handoff includes changed files and impact.

## Code Quality Practices

Keep code reviewable and robust:

- Use small functions with explicit inputs/outputs.
- Avoid hidden state mutation.
- Keep naming precise and domain-specific.
- Remove dead code and outdated comments during touched edits.
- Prefer typed, validated data flow where type hints already exist.
- Avoid TODO debt unless tracked with context and owner.

## API and Contract Practices

For CLI flags, config keys, JSON schemas, and public functions:

- Treat them as compatibility contracts.
- Use additive changes first; avoid renames/removals.
- If deprecation is required: document, warn, and provide migration path.
- Keep default behavior predictable and backward-safe.

## Repository Context

High-value areas:

- `annolid/gui`: Qt GUI workflows, canvas, dialogs, managers.
- `annolid/engine`: CLI orchestration and `annolid-run` entrypoints.
- `annolid/annotation`: format conversion and serialization.
- `annolid/tracking`, `annolid/tracker`: tracking integrations and runtime behavior.
- `annolid/datasets`: indexing/spec/splits.
- `pyproject.toml`, `annolid/version.py`, `uv.lock`: dependency and release integrity.
- `.github/workflows`: CI, docs, release, deployment.

## Change Scope Policy

- Prefer additive evolution over breaking changes.
- Avoid broad refactors unless required by the task.
- Keep public APIs stable unless a breaking change is requested and documented.
- If a breaking change is unavoidable, include migration notes in the same change set.

## Data and Annotation Integrity

- Preserve LabelMe-compatible shape semantics and defaults.
- Avoid destructive rewrites of user annotations.
- For serialization changes, ensure round-trip compatibility.
- Missing optional fields must map to safe, documented defaults.

## Reliability, Errors, and Observability

- Surface errors with root-cause context and remediation steps.
- Do not swallow exceptions silently.
- Keep retry logic bounded and explicit.
- Add or improve logging around critical state transitions.
- Ensure failures are diagnosable from logs/CI output.

## Security and Privacy Practices

- Never log secrets, tokens, or sensitive user data.
- Validate and sanitize external inputs (paths, URLs, config values).
- Use least-privilege assumptions in scripts and workflow tokens.
- Prefer secure defaults; require explicit opt-in for risky behavior.

## GUI Engineering Standard

- Keep Qt signal/slot interactions safe; avoid feedback loops.
- Treat `dirty`, selection, and visibility state as correctness-critical.
- Validate load/save/file-switch flows when touching GUI state logic:
  - checkbox state,
  - canvas visibility,
  - unsaved-change prompts,
  - file-switch behavior.
- Avoid long blocking operations on the UI thread.

## Performance Engineering Practices

- Establish baseline before optimization.
- Optimize hot paths only after correctness is proven.
- Preserve deterministic behavior while improving speed.
- Document meaningful performance tradeoffs in PR summary.

## Validation Matrix

Run the smallest meaningful validation first, then expand by risk.

Core checks:

- Python tests: `source .venv/bin/activate && pytest`
- Hooks/lint: `source .venv/bin/activate && pre-commit run --all-files` (if configured)

When relevant:

- Lockfile changes: `source .venv/bin/activate && uv lock --check`
- Packaging/release changes:
  - `source .venv/bin/activate && python -m build`
  - `source .venv/bin/activate && python -m twine check dist/*`
- Docs/workflow changes:
  - `source .venv/bin/activate && mkdocs build --strict --clean --config-file mkdocs.yml`

Test strategy expectations:

- Add regression tests for bug fixes.
- Prefer focused unit tests for logic and targeted integration tests for workflows.
- Keep tests deterministic (fixed seeds, stable fixtures).
- Avoid flaky timing/network dependencies in default CI paths.

## CI and Release Discipline

- Keep CI definitions coherent across `.github/workflows`.
- Prefer explicit path filters and deterministic change detection.
- Avoid ad-hoc release metadata edits when scripted release flow exists.
- Maintain tag/version consistency (`vX.Y.Z` vs package version).
- Keep workflows debuggable: explicit path filters and changed-file logs.

Release commands:

- `make release-patch`
- `make release-minor`
- `make release-major`
- `make release VERSION=X.Y.Z`

## Documentation Standard

When behavior changes, update docs in the same change set:

- `README.md` for user-facing quick guidance.
- `docs/` for canonical docs.
- `RELEASING.md` for release-process updates.

Docs must be concrete, command-driven, and platform-aware when applicable.

## Documentation Quality Practices

- Document behavior, not implementation trivia.
- Include exact commands and expected outcomes for operational steps.
- Keep canonical source singular (`docs/`), avoid duplicate guidance.
- Update migration/redirect notes when links or paths change.

## Review Standard

Prioritize findings in this order:

1. Correctness and behavioral regressions
2. Data integrity and compatibility risk
3. Missing or insufficient validation
4. Maintainability and clarity

Review notes should include concrete file/line references when possible.

Review checklist:

- Is the change minimal and coherent?
- Are contracts and compatibility preserved?
- Are failure modes handled explicitly?
- Are tests meaningful for the changed behavior?
- Are docs updated for user-visible changes?

## Commit Quality

Use scoped, intent-clear messages:

- `fix(gui): default missing shape visibility to checked`
- `docs(install): add one-line installer choices guide`
- `fix(lockfile): regenerate uv.lock as valid TOML`
- `ci(docs): tighten docs trigger paths and debug logging`

Keep unrelated changes out of the same commit.

Commit hygiene:

- One logical change per commit.
- Subject line explains intent and scope.
- Body explains why, not just what, for non-trivial changes.

## What to Avoid

- Mass-formatting unrelated files.
- Editing vendored third-party code unless explicitly required.
- Adding speculative dependencies.
- Suppressing errors that should be surfaced to users.
- “Passing CI by workaround” instead of fixing root cause.
- Merging behavior changes without validation evidence.
- Mixing refactor + feature + docs + workflow churn in one opaque commit.

## Communication Contract

- Be concise, direct, and technically specific.
- State assumptions explicitly.
- Call out unresolved risks.
- Ask for clarification only when truly blocking; otherwise proceed.
