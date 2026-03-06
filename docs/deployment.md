# Deployment

Annolid publishes multiple documentation surfaces from this repository.

## Source-of-truth Layout

- `website/`: landing page content for the site root
- `docs/`: MkDocs source for the docs portal and mirrored docs routes
- `docs/tutorials/`: notebook and markdown tutorial assets tracked in the repo

## Current GitHub Actions Pipelines

- `docs-quality.yml`
  Builds the MkDocs site in strict mode and lints the primary docs pages.
- `docs-pages.yml`
  Builds the MkDocs site and uploads the HTML artifact for review/validation.
- `healthonrails-site-sync.yml`
  Syncs generated output to `healthonrails/healthonrails.github.io` for the public site, gated to the appropriate branch/event conditions.
- `book-pages.yml`
  Builds the notebook/book surface that is published separately.
- `CI.yml`
  Includes broader repository validation and may run docs checks depending on changes.

## Published Targets

- `/`: landing page from `website/`
- `/portal/`: MkDocs docs homepage
- root docs routes such as `/installation`, `/workflows`, and `/reference` for compatibility
- `/book/`: book/notebook surface

## Current MkDocs Scope

The active MkDocs nav now includes:

- home
- installation
- uv setup
- one-line installer guide
- workflows
- tutorials
- MCP
- SAM 3D
- deployment
- migration
- redirects
- reference

That matters because README links should point to docs that are actually published, not pages excluded from the site build.

## Deployment Principles

- keep `docs/` as the canonical user-facing documentation source,
- build MkDocs in strict mode,
- keep publication reproducible and debuggable,
- preserve compatibility routes while evolving the docs structure,
- prefer ephemeral generated site output over committed build artifacts.

## Operational Note

If you change docs pages, `mkdocs.yml`, or related sync scripts, validate with:

```bash
source .venv/bin/activate
mkdocs build --strict --clean --config-file mkdocs.yml
```
