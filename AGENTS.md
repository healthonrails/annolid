# AGENTS.md (Annolid)

This file defines how coding agents and contributors should work in this repository.
It is intentionally opinionated and Annolid-specific.

## Quick Agent Checklist

Use this before every final response or handoff:

1. Confirm the user-visible goal in one sentence.
2. Change only the minimum necessary files.
3. Preserve backward compatibility unless change is explicitly requested.
4. Run the smallest relevant validation (`pytest`, targeted checks, or lock/build checks as needed).
5. Update docs when behavior, flags, commands, or workflows changed.
6. Verify no accidental edits in unrelated files.
7. Summarize what changed, why, and any remaining risk or follow-up.

## Mission

Build and maintain Annolid as a reliable toolkit for:
- annotation,
- segmentation,
- tracking,
- behavior analysis,
- and practical model workflows in real lab/research conditions.

Primary quality bar:
- correctness first,
- reproducibility second,
- usability in the GUI/CLI third,
- performance optimization after correctness and reproducibility are solid.

## Project context

Annolid is not a toy demo repo. It includes:
- a Qt GUI (`annolid/gui`),
- a CLI runtime (`annolid/engine/cli.py`, exposed as `annolid-run`),
- model/tracking pipelines (Cutie, SAM-family, DINO/YOLO integrations),
- dataset conversion/indexing utilities,
- optional extras (e.g. `sam3`, `image_editing`, `text_to_speech`, `qwen3_embedding`).

The repository contains large, optional, and vendor-style subtrees. Be careful when changing shared interfaces.

## Source map (high-value areas)

- `annolid/gui`: GUI workflows, canvas, labeling UX, dialogs, managers.
- `annolid/engine`: CLI entrypoints and command orchestration.
- `annolid/annotation`: label formats, conversion, serialization.
- `annolid/tracking` and `annolid/tracker`: tracking logic and integrations.
- `annolid/datasets`: dataset indexing, specs, split builders.
- `annolid/version.py` + `pyproject.toml`: package/runtime version sync.
- `.github/workflows`: CI, release, publishing.
- `scripts/release.sh`, `RELEASING.md`: release automation and policy.

## Non-negotiable engineering rules

1. Prefer minimal, surgical changes.
2. Keep behavior backward-compatible unless explicitly changing behavior.
3. Do not silently break GUI workflows to satisfy CLI assumptions, or vice versa.
4. Keep file formats stable (`labelme` JSON semantics, index schemas, lockfiles).
5. Never commit secrets, credentials, or local absolute paths.
6. Never introduce fake/mock confidence for end-to-end behavior claims.

## Definition of Done

A task is done only when all are true:
- Implementation matches the requested behavior.
- Relevant checks were run (or an explicit reason is provided if not run).
- Documentation is updated for user-facing changes.
- No unrelated regressions were introduced.
- Final summary includes changed files and impact.

## Environment and dependency policy

- Python support target is defined by `pyproject.toml` (`>=3.10`).
- Use project-managed dependencies from `pyproject.toml`.
- Optional features must be guarded with graceful fallback when extras are missing.
- For dependency lock updates, ensure `uv.lock` remains valid TOML and parseable by `uv`.

## Coding guidelines

- Follow existing project style and naming before introducing new patterns.
- Keep functions cohesive and explicit; avoid hidden side effects.
- Add comments only where logic is non-obvious.
- Avoid broad refactors unless required for the task.
- Preserve existing public APIs unless task explicitly requires change.
- Favor robust error messages with actionable remediation.

## Compatibility policy

- Treat on-disk formats and stable interfaces as contracts.
- If a breaking change is unavoidable, document migration steps in the same change set.
- Prefer additive evolution (new optional fields/flags) over destructive schema changes.

## GUI-specific guidelines

- Maintain Qt signal/slot safety and avoid signal feedback loops.
- Treat GUI state flags (`dirty`, selection state, visibility state) as correctness-critical.
- When changing load/save behavior, validate:
  - checkbox state,
  - canvas visibility,
  - unsaved-change prompts,
  - and file-switch behavior.
- Avoid expensive blocking work on UI thread when avoidable.

## Data and annotation integrity

- Preserve LabelMe-compatible shape fields and defaults.
- Missing optional fields should resolve to safe, documented defaults.
- Avoid destructive rewrites of user annotations.
- If changing serialization behavior, confirm round-trip compatibility.

## Performance and reliability

- Prioritize deterministic, reliable behavior over micro-optimizations.
- Avoid introducing background-thread/UI-thread races.
- For heavy operations, prefer progressive feedback and cancellation-safe patterns.

## Testing expectations

Before finishing substantial changes, run targeted checks that match the touched area:

- Python tests: `pytest`
- Lint/format hooks: `pre-commit run --all-files` (if available)
- For packaging/release changes:
  - `python -m build`
  - `python -m twine check dist/*`
- For lockfile changes:
  - `uv lock --check`

Use the smallest meaningful test surface first, then broaden if risk is high.

When changing:
- GUI state logic: validate file switching, dirty state, shape visibility, save/load.
- Locking/dependencies: validate `uv.lock` with `uv lock --check`.
- Release flow: validate scripts/docs/workflow assumptions remain aligned.

## CI and release expectations

- CI definitions live in `.github/workflows/CI.yml`.
- Release workflows must keep tag/version consistency (`vX.Y.Z` vs package version).
- Use release automation:
  - `make release-patch|release-minor|release-major`
  - or `make release VERSION=X.Y.Z`
- Do not manually mutate release metadata in ad-hoc ways if scripted flow exists.

## PR and review hygiene

- Keep PRs focused and reviewable; split unrelated work.
- Include a concise change summary and verification notes.
- Prefer explicit file/line references in review discussions.
- Respond to review comments with concrete fixes or technical rationale.

## Documentation expectations

When behavior changes, update docs in the same PR/commit set:
- `README.md` for user-facing workflow changes.
- `docs/` for detailed guidance.
- `RELEASING.md` for release-process updates.

Documentation must be concrete, command-driven, and platform-aware (Linux/macOS/Windows where relevant).

## Commit quality

Use clear commit messages with scope and intent, e.g.:
- `fix(gui): default missing shape visibility to checked`
- `docs(install): add one-line installer choices guide`
- `fix(lockfile): regenerate uv.lock as valid TOML`
- `chore(release): v1.5.3`

Keep unrelated changes out of the same commit.

## What to avoid

- Do not mass-format unrelated files.
- Do not rewrite vendored third-party code unless explicitly requested.
- Do not add speculative dependencies.
- Do not claim tests passed if they were not run.
- Do not suppress errors that should be surfaced to users.

## Communication contract (for agents)

- Be concise, direct, and technically specific.
- Do not claim unrun tests as passed.
- Call out assumptions and unresolved risks explicitly.
- Ask for clarification only when truly blocking; otherwise proceed.

## Agent operating checklist

For each task:
1. Understand the exact user-visible issue or goal.
2. Locate the true source of behavior in code.
3. Implement the smallest correct fix.
4. Validate with relevant tests/checks.
5. Update docs if user-facing behavior changed.
6. Summarize what changed, why, and any remaining risk.

If blocked by missing context, ask one focused question that unblocks execution.
