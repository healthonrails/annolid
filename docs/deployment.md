# Deployment

Annolid documentation uses a unified MkDocs source under `docs/`.

## Pipelines

- `docs-quality.yml`: validates combined site/doc sources and enforces markdown + strict MkDocs checks.
- `docs-pages.yml`: builds docs and uploads the generated HTML artifact for validation.
- `healthonrails-site-sync.yml`: on every push to `main`, syncs the combined public site to `healthonrails/healthonrails.github.io`.
- `book-pages.yml`: builds Jupyter Book and uploads the generated HTML artifact for validation.
- `CI.yml`: runs core tests plus a conditional strict docs build verification.

## Deployment Targets

- External site repo: `healthonrails/healthonrails.github.io`
- External site branch: `gh-pages`
- Published paths:
  - root (`/`) for the landing page from `website/`
  - `/website-assets` for landing-page-only static assets
  - `/portal` for the MkDocs docs homepage
  - root docs routes (`/installation`, `/reference`, etc.) mirrored for compatibility
  - `/book` Jupyter Book

## Design Principles

- Keep source-of-truth docs in `docs/`.
- Publish docs at root and preserve `/portal` for backward compatibility.
- Attempt external site sync on every push to `main`; no-op cleanly when generated output does not change.
- Use deterministic sync (`rsync --delete`) for reproducible output.
- Fail fast when external deploy credentials are missing.
- Bootstrap the external `gh-pages` branch automatically on first deploy.
- Keep generated MkDocs HTML ephemeral instead of relying on a repo-local `site_docs/` directory.
- Keep the landing page in `website/` and the MkDocs homepage in `docs/index.md`.
