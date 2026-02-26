---
name: github
description: Work with GitHub repositories and pull requests using gh CLI.
metadata: '{"annolid":{"requires":{"bins":["gh"]}}}'
---

Use this skill when tasks involve GitHub issues, PRs, checks, releases, or comments.

Workflow:

1. Verify repository context and auth first:
   - `git_cli(args=["status","--short","--branch"])`
   - `gh_cli(args=["auth","status"])`
2. Inspect context before changing anything:
   - `gh_cli(args=["pr","status"])`
   - `gh_cli(args=["pr","checks"])`
   - `gh_cli(args=["pr","view"])`
3. For mutating operations, require explicit intent and set `allow_mutation=true`:
   - Example: `gh_cli(args=["pr","comment","123","--body","..."], allow_mutation=true)`
4. Prefer targeted actions over broad automation.
5. Summarize outcome with concrete IDs/URLs and next actions.
