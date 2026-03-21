# AGENTS.md — Annolid Engineering Standard

This document defines the required engineering standard for coding agents and contributors working in the Annolid repository.

Annolid supports real research and lab workflows. Every change must protect correctness, reproducibility, data integrity, and user trust across GUI, CLI, annotation, tracking, and model pipelines.

---

## 1. Mission

Build and maintain Annolid as a dependable system for:

- annotation,
- segmentation,
- tracking,
- behavior analysis,
- practical model workflows under real research conditions.

Engineering priority order:

1. Correctness
2. Reproducibility
3. Data integrity
4. Reliability
5. Usability
6. Maintainability
7. Performance

Performance matters, but never ahead of correctness, stable behavior, or user data safety.

---

## 2. Engineering Doctrine

The goal is not merely to make code pass.
The goal is to make the system correct, stable, diagnosable, and evolvable.

Contributors must optimize for:

- preserving user trust,
- protecting workflow continuity,
- keeping contracts stable,
- enabling future change without unnecessary rewrite,
- and reducing the long-term cost of ownership.

A good change is not the smallest patch.
A good change is the **minimum necessary change set that correctly resolves the problem at the right boundary**.

---

## 3. Non-Negotiable Rules

1. Fix the problem at its true boundary, not just at its symptom.
2. Keep change scope proportional to the task and justified by the root cause.
3. Preserve established user and developer contracts unless an intentional contract change is required.
4. Treat saved data, annotations, configs, exported artifacts, and user workflows as compatibility-sensitive surfaces.
5. Never commit secrets, credentials, tokens, or machine-local absolute paths.
6. Never claim tests, checks, or builds passed unless they were actually run.
7. Use `.venv` for local validation unless the task explicitly requires another environment.
8. Do not mix unrelated fixes, refactors, formatting churn, and workflow edits in the same change.
9. Do not conceal skipped validation, uncertainty, or residual risk.
10. Leave the touched code clearer, safer, and easier to evolve.

---

## 4. Scope and Change Control

### 4.1 Scope discipline

Prefer the **minimum necessary scope**, not reflexively the smallest diff.

That means:

- do not broaden a patch without reason,
- do not preserve bad structure merely to keep the diff small,
- do not use a local workaround where a slightly broader but cleaner fix is the correct one.

A larger change is justified when it materially improves one or more of:

- correctness,
- contract clarity,
- data safety,
- debuggability,
- testability,
- maintainability of the touched path.

### 4.2 Allowed scope expansion

It is acceptable to widen a change when needed to:

- remove the root cause rather than patch symptoms,
- add focused regression coverage,
- clarify or harden a contract,
- eliminate fragile duplication on the directly affected path,
- improve observability for a critical failure mode,
- update required user-facing documentation.

### 4.3 Disallowed scope expansion

Do not widen scope for:

- opportunistic cleanup,
- aesthetic refactors unrelated to the task,
- speculative abstractions,
- repo-wide formatting,
- dependency churn without direct need,
- “while here” edits with no acceptance target.

---

## 5. Compatibility and Contract Management

Backward compatibility is not a vague preference. It is a deliberate contract discipline.

### 5.1 Protect by default

The following are compatibility-sensitive unless explicitly designated otherwise:

- CLI flags and command behavior,
- config keys and defaults,
- file formats and serialization behavior,
- annotation semantics,
- exported schemas and metadata,
- public Python APIs,
- saved project behavior,
- GUI workflows users rely on,
- documented operational commands.

### 5.2 Internal freedom vs external stability

Staff-level engineering requires distinguishing between:

- **external contracts**, which should remain stable by default,
- **internal implementations**, which may change freely if contracts are preserved.

Do not freeze internals unnecessarily.
Do not destabilize external behavior casually.

### 5.3 If a contract must change

A contract change is allowed only when it is:

- intentional,
- necessary,
- explicitly described,
- validated against affected workflows,
- documented with migration guidance when applicable.

A breaking change must include, as appropriate:

- rationale,
- affected surfaces,
- migration path,
- warning or deprecation strategy,
- documentation updates,
- release-note visibility.

### 5.4 Preferred order of change

When evolving a contract, prefer this order:

1. additive support,
2. compatibility shim,
3. deprecation with warning,
4. removal only when justified and documented.

Do not silently reinterpret existing user data or prior inputs.

---

## 6. Required Design Questions Before Coding

For any non-trivial change, establish these first:

### 6.1 User-visible goal
State the intended outcome in one sentence.

### 6.2 Root cause
Identify the actual code path and failure boundary.

### 6.3 Invariants
Define what must remain true, including:

- persisted data expectations,
- UI state guarantees,
- CLI/config behavior,
- serialization behavior,
- error handling guarantees,
- determinism requirements.

### 6.4 Blast radius
Identify affected surfaces:

- GUI
- CLI
- IO / serialization
- training / inference / tracking runtime
- docs
- packaging
- CI / release

### 6.5 Acceptance target
Define a testable success condition before implementing.

No patch should ship on intent alone.

---

## 7. Required Execution Loop

For every task:

1. Restate the user-visible goal.
2. Trace the real code path.
3. Identify the root cause.
4. Choose the minimum necessary change set at the correct boundary.
5. Add or update focused validation.
6. Update documentation when behavior, expectations, or workflows changed.
7. Confirm no unrelated files were edited.
8. Summarize:
   - changed files,
   - behavioral impact,
   - validation performed,
   - unresolved risks.

---

## 8. Architecture Standard

Choose the simplest design that preserves extension points and keeps contracts clear.

Required practices:

- single clear responsibility per module/function,
- explicit boundaries between UI, domain logic, and IO,
- explicit state ownership,
- deterministic behavior for CI-critical paths,
- typed and validated data flow where type hints already exist,
- composition preferred over deep inheritance,
- adapters preferred over scattered special cases.

Avoid:

- hidden mutation,
- cross-layer leakage,
- speculative abstraction,
- deep implicit coupling,
- uncontrolled global state.

If selecting one design over another non-obvious alternative, document the tradeoff.

---

## 9. Data Integrity Standard

User data is a contract.

Must preserve:

- LabelMe-compatible semantics where promised,
- annotation meaning,
- round-trip safety for serialization changes,
- safe defaults for missing optional fields,
- stable interpretation of saved data.

Must avoid:

- destructive rewrites,
- silent field loss,
- backward-incompatible schema drift without documentation,
- lossy conversions without explicit user intent.

For serialization changes, think in terms of:

- forward compatibility,
- backward compatibility,
- downgrade behavior,
- recovery behavior after partial failure.

---

## 10. GUI Engineering Standard

GUI behavior is correctness-critical.

When touching GUI code, validate:

- dirty state,
- selection state,
- visibility state,
- checkbox state,
- file-switch behavior,
- unsaved-change prompts,
- load/save transitions,
- long-running task responsiveness.

Rules:

- keep signal/slot behavior explicit,
- avoid feedback loops and re-entrant corruption,
- avoid blocking the UI thread,
- preserve coherent state on partial failure,
- surface actionable errors.

Do not accept a CLI-clean solution that destabilizes GUI workflows.

---

## 11. Reliability and Observability Standard

Failures must be diagnosable by a user, maintainer, or CI log reader.

Required:

- actionable error messages,
- root-cause context,
- bounded retry behavior,
- logs around critical state transitions,
- debuggable failure output.

Forbidden:

- swallowed exceptions,
- broad exception handling without reason,
- vague error text,
- hidden retries,
- silent fallback that masks corruption or data loss.

Prefer explicit failure over misleading success.

---

## 12. Performance Standard

Performance changes must be evidence-driven.

Rules:

- baseline before optimizing,
- optimize verified hot paths,
- preserve correctness and determinism,
- document meaningful tradeoffs,
- avoid hidden cache invalidation or concurrency complexity unless justified.

Do not trade observability or reproducibility for speed without explicit justification.

---

## 13. Validation Standard

Run the smallest meaningful validation first, then expand according to risk.

Core commands:

- `source .venv/bin/activate && pytest`
- `source .venv/bin/activate && pre-commit run --all-files`

When relevant:

- `source .venv/bin/activate && uv lock --check`
- `source .venv/bin/activate && python -m build`
- `source .venv/bin/activate && python -m twine check dist/*`
- `source .venv/bin/activate && mkdocs build --strict --clean --config-file mkdocs.yml`

Validation expectations:

- add regression coverage for bug fixes where practical,
- prefer focused tests over broad incidental coverage,
- keep tests deterministic,
- avoid flaky timing or network dependence in default CI paths.

If validation was not run, state exactly what was skipped and why.

---

## 14. Documentation Standard

If behavior changes, documentation changes in the same change set.

Canonical locations:

- `README.md` for quick-start user guidance,
- `docs/` for canonical operational documentation,
- `RELEASING.md` for release-process changes.

Documentation must be:

- concrete,
- command-driven where appropriate,
- platform-aware when needed,
- free of duplicate or conflicting guidance.

Document behavior, not implementation trivia.

---

## 15. CI and Release Discipline

CI and release integrity are product-quality concerns.

Rules:

- keep workflow intent explicit,
- prefer deterministic path filters and changed-file detection,
- keep version/tag semantics coherent,
- avoid manual metadata drift where scripted release flow exists,
- make failures diagnosable from logs.

Do not make CI “pass” by bypassing the real issue.

---

## 16. Definition of Done

A task is done only when all are true:

- behavior matches the request,
- the root cause is addressed at the right boundary,
- affected contracts are preserved or explicitly changed,
- relevant validation was run, or skipped with explicit reason,
- user-facing changes are documented,
- unrelated regressions were not introduced,
- final handoff identifies impact, changed files, and residual risks.

---

## 17. Review Standard

Review priorities:

1. Correctness and regression risk
2. Data integrity and compatibility risk
3. Missing or weak validation
4. Reliability and diagnosability
5. Maintainability and clarity

Review checklist:

- Is the root cause actually addressed?
- Is the chosen scope justified?
- Are contracts preserved or explicitly evolved?
- Are failure modes visible and explicit?
- Are tests meaningful?
- Are docs updated?
- Are unrelated edits absent?

---

## 18. Commit Standard

Use scoped, intent-clear commit messages.

Examples:

- `fix(gui): restore shape visibility defaults on reload`
- `fix(io): preserve optional label fields during round-trip serialization`
- `docs(install): clarify installer choices and expected outcomes`
- `ci(docs): tighten path filters and add changed-file logging`

Commit hygiene:

- one logical change per commit,
- subject explains scope and intent,
- body explains why for non-trivial changes.

---

## 19. Communication Contract

Be concise, direct, and technically precise.

Always:

- state assumptions,
- separate facts from inference,
- report actual validation,
- identify residual risk,
- avoid overstating certainty.

Ask for clarification only when truly blocking.
Otherwise proceed with clearly stated assumptions.

---

## 20. What to Avoid

Do not:

- optimize for tiny diffs at the expense of correct design,
- preserve broken structure merely for backward appearance,
- silently break workflows or contracts,
- mass-format unrelated files,
- add speculative dependencies,
- suppress important failures,
- mix refactor + feature + docs + workflow churn in one opaque patch,
- claim validation not actually performed,
- trade data safety for convenience.

---

## 21. Final Standard

Annolid engineering work must be:

- correct,
- reproducible,
- compatibility-aware,
- data-safe,
- diagnosable,
- maintainable,
- narrowly scoped but not artificially constrained,
- validated,
- documented.

When forced to choose, protect correctness, user data, and workflow continuity over implementation speed or patch smallness.
