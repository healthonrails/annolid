# Deployment

Annolid currently uses automated GitHub Actions pipelines for website, book, and documentation publishing.

## Pipelines

- `website-pages.yml`: deploys `website/` updates.
- `book-pages.yml`: builds and deploys Jupyter Book to `/book`.
- `CI.yml`: builds and publishes Sphinx docs.
- `docs-portal-pages.yml`: builds and deploys this MkDocs portal to `/portal`.

## Deployment Targets

- This repository `gh-pages` branch
- Optional external site repo: `healthonrails/healthonrails.github.io`

## Design Principles

- Keep sources in this repository.
- Publish by path (`/`, `/book`, `/portal`) to avoid duplication.
- Use path filters so workflows run only on relevant changes.
- Use deterministic sync (`rsync --delete`) for reproducible outputs.
