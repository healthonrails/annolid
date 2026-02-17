# Annolid Bot ClawHub Skills Tutorial

This tutorial shows how to search and install agent skills from ClawHub in Annolid.

## 1. Prerequisites

No Node.js or npm packages are required.
Annolid uses Python HTTP APIs to search/download skills from ClawHub.

## 2. Open Annolid Bot

1. Start Annolid.
2. Open **AI & Models → Annolid Bot…**
3. Pick your chat provider/model as usual.

## 3. Search for skills (natural language)

In Annolid Bot, ask:

- `Search ClawHub skills for behavior analysis`
- `Find ClawHub skills for web scraping`
- `Find skills on ClawHub for PDF summarization`

Annolid Bot will call the `clawhub_search_skills` tool and return results.

## 4. Install a skill by slug

After you identify a skill slug, ask:

- `Install skill <slug> from ClawHub`

Example:

- `Install skill weather-pro from ClawHub`

Annolid will call `clawhub_install_skill` and install into your Annolid agent workspace `skills/` directory.

## 5. Restart behavior

After installing a new skill, start a new Annolid Bot session so the newly installed skill instructions are loaded into context.

## 6. Troubleshooting

- `ClawHub CLI is not available`
  - Not applicable in current implementation (no CLI dependency).
  - If search/install fails, check internet access and firewall/proxy settings.
- `Invalid skill slug format`
  - Use the exact slug from search results (letters/numbers/`-`/`_`/`.`).
- `ClawHub command timed out`
  - Retry and check internet connection.

## 7. What tools are used

This feature is powered by core agent tools:

- `clawhub_search_skills(query, limit?)`
- `clawhub_install_skill(slug)`

These are core tools (not GUI-only wrappers), so the feature is available in agent tool workflows beyond the UI layer.
