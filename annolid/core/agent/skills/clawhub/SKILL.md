---
name: clawhub
description: Search and install agent skills from ClawHub, the public skill registry.
homepage: https://clawhub.ai
---

# ClawHub

Use ClawHub to discover and install skills for the Annolid agent workspace.

## When to use

- User asks to search for skills.
- User asks to install/update a skill.
- User asks what skills are available for a task.

## Preferred tools

- `clawhub_search_skills(query, limit?)`
- `clawhub_install_skill(slug)`

These tools are preferred over raw shell calls because they:

- use a pure Python implementation (no Node.js/npm dependency),
- enforce workspace-aware install paths, and
- return structured payloads.

## Notes

- Installing a skill writes under workspace `skills/`.
- After install, start a new session to ensure newly installed skill instructions are loaded.
