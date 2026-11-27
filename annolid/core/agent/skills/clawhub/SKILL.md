---
name: clawhub
description: Search and install agent skills from ClawHub, the public skill registry, and refresh the local skill index.
homepage: https://clawhub.ai
metadata: '{"annolid":{"always":false,"tags":["clawhub","install skill","search skills","skill registry","marketplace","refresh skills"]}}'
---

# ClawHub

Use ClawHub to discover and install skills for the Annolid agent workspace.

## When to use

- User asks to search for community or third-party skills.
- User asks to install or update a skill from ClawHub.
- User asks what additional skills are available for a workflow.

## Preferred tools

- `clawhub_search_skills(query, limit?)`: search the public ClawHub registry.
- `clawhub_install_skill(slug)`: install a skill into `<workspace>/skills/<slug>/`.
- `admin_skills_refresh(workspace?)`: force an immediate registry reload after installing or editing skills (CLI equivalent: `annolid-run agent skills refresh`).

These tools are preferred over raw shell calls because they:

- use a pure Python implementation (no Node.js/npm dependency),
- enforce workspace-aware install paths, and
- return structured payloads.

## Notes

- Installing a skill writes under `<workspace>/skills/`, which has highest precedence over managed and built-in skills.
- After install, call `admin_skills_refresh()` (or enable hot reload with `ANNOLID_SKILLS_LOAD_WATCH=1`) or start a new session so newly installed instructions are active.
- Review third-party `SKILL.md` instructions before trusting commands that mutate files or access credentials.
