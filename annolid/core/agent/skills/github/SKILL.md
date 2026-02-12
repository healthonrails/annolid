---
name: github
description: Work with GitHub repositories and pull requests using gh CLI.
metadata: '{"annolid":{"requires":{"bins":["gh"]}}}'
---

Use this skill when tasks involve GitHub issues, PRs, checks, releases, or comments.

Workflow:

1. Verify `gh auth status` before mutating operations.
2. Inspect repo/PR context (`gh pr view`, `gh pr checks`, `gh issue view`).
3. Prefer targeted actions over broad automation.
4. Summarize outcome with concrete IDs/URLs and next actions.
