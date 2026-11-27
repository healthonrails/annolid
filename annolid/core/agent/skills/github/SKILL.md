---
name: github
description: Work with Git repositories, GitHub pull requests, issues, CI checks, and releases using dedicated git and gh tools.
metadata: '{"annolid":{"always":false,"tags":["github","git","gh","pull request","pr","issues","ci","checks","diff","commit","release"],"requires":{"bins":["gh"]}}}'
---

# Git & GitHub Workflows

Use this skill when tasks involve Git status/diffs/history or GitHub issues, PRs, CI checks, releases, or comments.
Prefer dedicated VCS tools (or GUI `/git` and `/gh` slash shortcuts) over generic shell execution.

## Workflow

1. Verify repository context and auth first:
   - `git_status(short=true)`
   - `gh_cli(args=["auth", "status"])`
2. Inspect context before changing anything:
   - `github_pr_status()`
   - `github_pr_checks()`
   - `gh_cli(args=["pr", "view"])`
   - Optional detail:
     - `git_diff(cached=true)` for staged changes (or `cached=false` for unstaged)
     - `git_log(max_count=20, oneline=true)` for recent commits
3. For mutating operations, require explicit user intent and pass `allow_mutation=true`:
   - Git: `git_cli(args=["commit", "-m", "..."], allow_mutation=true)`
   - GitHub: `gh_cli(args=["pr", "comment", "123", "--body", "..."], allow_mutation=true)`
4. Prefer targeted actions over broad automation.
5. Summarize outcome with concrete commit SHAs, PR/issue numbers, check states, URLs, and next actions.

## Guardrails

- Never push, force-push, merge, or delete branches/PRs without explicit user instruction.
- Do not claim git/gh access is unavailable without checking tool output first.
- If `gh` is missing or unauthenticated, report the exact tool error and next action.
