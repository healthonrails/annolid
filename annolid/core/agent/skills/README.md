# Annolid Built-in Skills

This directory contains built-in skills for the Annolid Bot runtime (`annolid/core/agent/skills/<slug>/SKILL.md`).

## Format

Each skill is a folder containing `SKILL.md` with YAML frontmatter and concise, tool-grounded markdown instructions:

- `name`: **must match the directory slug**
- `description`: concise one-sentence summary used in skill menus and retrieval ranking
- `metadata`: JSON string or object using the `annolid` namespace (legacy `openclaw` keys are still accepted for compatibility)

Supported metadata & invocation keys:

- `always`: load this skill automatically in every turn's context (keep `false` unless universally needed)
- `tags` (or `keywords`): list of domain keywords/phrases indexed by both lexical and embedding skill retrieval
- `user-invocable` (default `true`)
- `disable-model-invocation` (default `false`; excludes the skill from model auto-suggestion and context injection)
- `command-dispatch`, `command-tool`, `command-arg-mode`
- `os`: allowed operating systems (`darwin`, `linux`, `win32`)
- `requires.bins`: required command-line binaries (declare only binaries the skill actually runs)
- `requires.anyBins`: at least one CLI required
- `requires.env`: required environment variables
- `requires.config`: required config paths (dot notation)

Example:

```yaml
---
name: weather
description: Get current weather and forecasts using web search or web fetch.
metadata: '{"annolid":{"always":false,"tags":["weather","forecast","temperature","rain"]}}'
---
```

## Discovery & Precedence

Skills are discovered from four roots (highest precedence wins on name conflict):

1. `<workspace>/skills` (workspace-local overrides & custom skills)
2. `~/.annolid/skills` (managed/shared user skills)
3. `annolid/core/agent/skills` (built-in skills bundled with Annolid)
4. Extra dirs from `ANNOLID_SKILLS_EXTRA_DIRS` or `config.json` `skills.load.extraDirs` (lowest)

## CLI & Runtime Inspection

- `annolid-run agent skills inspect` — list discovered skills, eligibility, and manifest validity
- `annolid-run agent skills refresh` (or tool `admin_skills_refresh()`) — reload skills without restarting
- `annolid-run agent skills shadow` — inspect which skills shadow lower-precedence definitions
- `annolid-run agent-skill-pool --task-hint "<prompt>"` — test ranked skill retrieval for a query

### Hot Reload

Enable mtime-based refresh checks between turns:

- Env: `ANNOLID_SKILLS_LOAD_WATCH=1` (or `ANNOLID_SKILLS_WATCH=1`), optional `ANNOLID_SKILLS_WATCH_POLL_SECONDS=0.5`
- Config (`~/.annolid/config.json`): `skills.load.watch: true`, `skills.load.pollSeconds: 1.0`

## Built-in Skill Catalog (41 Skills)

- **Behavior Analysis & Scientific Rigor**: `behavior-assay-taxonomy`, `behavior-feature-selection`, `behavior-segmentation`, `behavior-vlm-labeling`, `behavior-evaluation`, `timeline-reasoning`, `vision-language-analysis`, `perception-routing`, `model-policy`, `metrics-derivation`, `experimental-design`, `sandboxed-analysis`, `scientific-reporting`, `provenance`
- **Tracking, Video, Shapes & 3D**: `sam3-agent-video`, `video-ffmpeg`, `camera-realtime`, `shape-workflows`, `shape-annotation-store`, `threejs`, `gui`
- **Datasets, Training, Colab & Sim2Real**: `dataset-create`, `dataset-convert`, `model-training`, `polygon-classifier-training`, `google-colab`, `sim2real`
- **Research, Citations & Web**: `research-writing`, `citations`, `summarize`, `web-scraping`, `mcp-search`, `weather`
- **Cloud, Automation, DevOps & Meta-Skills**: `cloud-workspace`, `cron`, `github`, `logs`, `tmux`, `multi-agent-orchestration`, `clawhub`, `skill-creator`
