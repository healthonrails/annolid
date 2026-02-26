---
name: github
description: Work with GitHub repositories and pull requests using gh CLI.
metadata: '{"annolid":{"requires":{"bins":["gh"]}}}'
---

Use this skill when tasks involve GitHub issues, PRs, checks, releases, or comments.
Prefer dedicated VCS tools over generic shell execution.

Workflow:

1. Verify repository context and auth first:
   - `git_status(short=true)`
   - `gh_cli(args=["auth","status"])`
2. Inspect context before changing anything:
   - `github_pr_status()`
   - `github_pr_checks()`
   - `gh_cli(args=["pr","view"])`
   - Optional detail:
     - `git_diff(cached=true)` for staged changes
     - `git_log(max_count=20, oneline=true)` for recent commits
3. For mutating operations, require explicit intent and set `allow_mutation=true`:
   - Example: `gh_cli(args=["pr","comment","123","--body","..."], allow_mutation=true)`
4. Prefer targeted actions over broad automation.
5. Summarize outcome with concrete IDs/URLs and next actions.

Guardrails:

- Do not claim git/gh access is unavailable without checking tool output first.
- If `gh` is missing or unauthenticated, report the exact tool error and next action.
