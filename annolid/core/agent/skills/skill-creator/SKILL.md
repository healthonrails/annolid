---
name: skill-creator
description: Create or update Annolid-compatible skills.
metadata: '{"annolid":{"always":false,"requires":{"bins":[]}}}'
---

Use this skill when users ask to create a new skill.

Checklist:

1. Create folder under `skills/<name>/`.
2. Add `SKILL.md` with frontmatter:
   - `name`
   - `description`
   - `metadata` JSON (`annolid` section)
3. Keep instructions concrete and command-oriented.
4. Add scripts/assets only if needed by the workflow.
5. Ensure content is concise and safe for autonomous execution.
