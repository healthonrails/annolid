# Deployment

Annolid currently uses automated GitHub Actions pipelines for website, book, and documentation publishing.

## Pipelines

- `docs-quality.yml`: validates combined site/doc sources and enforces markdown + strict portal build checks.
- `book-pages.yml`: builds and deploys Jupyter Book to `/book`.
- `CI.yml`: builds/tests and publishes Sphinx docs in this repository.
- `docs-portal-pages.yml`: builds and deploys this MkDocs portal to `gh-pages/portal`.
- `healthonrails-site-sync.yml`: deploys the combined public site to `annolid.com`.

## Deployment Targets

- This repository `gh-pages` branch (`/portal` for MkDocs artifact publishing)
- Optional external site repo: `healthonrails/healthonrails.github.io`

## Design Principles

- Keep sources in this repository.
- Publish the portal as the root site (`/`) and keep `/portal` as a compatibility path.
- Use path filters so workflows run only on relevant changes.
- Use deterministic sync (`rsync --delete`) for reproducible outputs.
