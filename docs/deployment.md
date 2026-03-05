# Deployment

Annolid documentation uses a unified MkDocs source under `docs/`.

## Pipelines

- `docs-quality.yml`: validates combined site/doc sources and enforces markdown + strict MkDocs checks.
- `docs-pages.yml`: builds docs and uploads the generated HTML artifact for validation.
- `healthonrails-site-sync.yml`: deploys the combined public site to `annolid.com`.
- `book-pages.yml`: builds Jupyter Book and uploads the generated HTML artifact for validation.
- `CI.yml`: runs core tests plus a conditional strict docs build verification.

## Deployment Targets

- External site repo: `healthonrails/healthonrails.github.io`
- External site branch: `gh-pages`
- Published paths:
  - root (`/`) for docs
  - `/portal` compatibility mirror
  - `/book` Jupyter Book

## Design Principles

- Keep source-of-truth docs in `docs/`.
- Publish docs at root and preserve `/portal` for backward compatibility.
- Use path filters and conditional build logic to reduce unnecessary runs.
- Use deterministic sync (`rsync --delete`) for reproducible output.
- Fail fast when external deploy credentials are missing.
- Bootstrap the external `gh-pages` branch automatically on first deploy.
