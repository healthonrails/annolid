# Deployment

Annolid documentation uses a unified MkDocs source under `docs/`.

## Pipelines

- `docs-quality.yml`: validates combined site/doc sources and enforces markdown + strict MkDocs checks.
- `docs-pages.yml`: builds docs and publishes to repository `gh-pages` (root + `/portal` compatibility path).
- `healthonrails-site-sync.yml`: deploys the combined public site to `annolid.com`.
- `book-pages.yml`: builds and deploys Jupyter Book to `/book`.
- `CI.yml`: runs core tests plus a conditional strict docs build verification.

## Deployment Targets

- Repository `gh-pages` branch:
  - root (`/`) for docs
  - `/portal` compatibility mirror
  - `/book` Jupyter Book
- External site repo: `healthonrails/healthonrails.github.io`

## Design Principles

- Keep source-of-truth docs in `docs/`.
- Publish docs at root and preserve `/portal` for backward compatibility.
- Use path filters and conditional build logic to reduce unnecessary runs.
- Use deterministic sync (`rsync --delete`) for reproducible output.
