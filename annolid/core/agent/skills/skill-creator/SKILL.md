---
name: skill-creator
description: Create, validate, debug, or update Annolid-compatible agent skills with proper frontmatter, tags, gating, and registry refresh.
metadata: '{"annolid":{"always":false,"tags":["skill","create skill","author skill","skill.md","frontmatter","skill registry","shadow","metadata","annolid skills"]}}'
---

# Skill Creator

Use this skill when users ask to create, update, validate, or debug an Annolid Bot skill.

## Directory & Precedence

Annolid discovers `SKILL.md` directories in priority order (highest wins on name collision):

1. `<workspace>/skills/<slug>/SKILL.md` (workspace-local custom skills)
2. `~/.annolid/skills/<slug>/SKILL.md` (managed/shared user skills)
3. `annolid/core/agent/skills/<slug>/SKILL.md` (built-in Annolid skills)
4. Extra dirs via `ANNOLID_SKILLS_EXTRA_DIRS` or `skills.load.extraDirs`

## Required `SKILL.md` Structure

```markdown
---
name: my-skill-slug
description: Concise one-sentence summary of what the skill does and when to trigger it.
metadata: '{"annolid":{"always":false,"tags":["keyword1","keyword2","domain phrase"],"requires":{"bins":[]}}}'
---

# My Skill Title

Use this skill when ...
```

### Frontmatter Contract

- `name`: **must match the directory slug** (`my-skill-slug`).
- `description`: concise, action-oriented summary used in skill listings and task-hint matching.
- `metadata` (JSON string or object under `"annolid"`):
  - `always`: `false` by default; reserve `true` only for universal guardrails.
  - `tags` (or `keywords`): list of retrieval keywords/phrases to improve task matching.
  - `os`: optional list from `["darwin", "linux", "win32"]`.
  - `requires`: optional gating (`bins`, `anyBins`, `env`, `config`); only declare binaries the skill actually runs.
- Optional top-level invocation keys:
  - `user-invocable` (default `true`)
  - `disable-model-invocation` (default `false`; hides skill from model auto-suggestion and model context)
  - `command-dispatch`, `command-tool`, `command-arg-mode`
  - Production/signed skills: `signature`, `signature_alg` (`hmac-sha256` / `ed25519`).

## Authoring Rules

1. Reference **only real registered tool names** and verified CLI flags (check `TOOLS.md` or `--help`).
2. Keep instructions concise, command-oriented, and explicit about safe defaults (`dry_run=true`, `allow_mutation=true`, allowed roots).
3. Add `scripts/` or `assets/` only when deterministic code materially improves reliability.
4. Never embed secrets, credentials, or machine-local absolute paths.

## Validation & Refresh

- Refresh in-session: `admin_skills_refresh()`
- Inspect/debug from CLI:
  - `annolid-run agent skills inspect`
  - `annolid-run agent skills refresh`
  - `annolid-run agent skills shadow` (check overridden skills)
  - `annolid-run agent-skill-pool --task-hint "<sample user prompt>"` (verify retrieval ranking)
