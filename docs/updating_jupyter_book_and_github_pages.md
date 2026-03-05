# Updating Annolid Jupyter Book + GitHub Pages

This repository contains two documentation surfaces:

- Sphinx docs from `docs/` (published at site root).
- Jupyter Book from `book/` (published under `/book/`).
- MkDocs portal from `docs_portal/` (published under `/portal/`).
- Landing page source in `website/` (`index.html` + `assets/`) for `annolid.com`.

They are deployed by separate workflows:

- `.github/workflows/CI.yml`: Sphinx docs (`docs/`) to this repo `gh-pages` root.
- `.github/workflows/book-pages.yml`: Jupyter Book (`book/`) to:
  - this repo `gh-pages/book/`
  - `healthonrails/healthonrails.github.io` (optional)
- `.github/workflows/docs-portal-pages.yml`: MkDocs portal (`docs_portal/`) to:
  - this repo `gh-pages/portal/`
  - `healthonrails/healthonrails.github.io/portal/` (optional)
- `.github/workflows/website-pages.yml`: landing page (`website/`) to `healthonrails/healthonrails.github.io`.
- `.github/workflows/docs-quality.yml`: quality gate for portal docs (strict MkDocs build + markdown lint).

## Local workflow

Use the project virtual environment:

```bash
source .venv/bin/activate
```

Install or refresh docs dependencies:

```bash
pip install -r docs/requirements.txt
pip install -r docs/requirements_mkdocs.txt
pip install -r book/requirements.txt
```

Build Sphinx docs:

```bash
make -C docs html
```

Build Jupyter Book:

```bash
jupyter-book build book
```

Build MkDocs portal:

```bash
mkdocs build --clean --config-file mkdocs.yml
```

Preview the Jupyter Book locally:

```bash
open book/_build/html/index.html
```

## GitHub Pages deployment

Sphinx deployment (`CI.yml`):

1. Build `docs/` HTML.
2. Update `gh-pages` branch root.
3. Ensure `.nojekyll` exists.

Book deployment (`book-pages.yml`):

1. Build `book/` HTML.
2. Update `gh-pages/book/`.
3. Ensure `.nojekyll` exists.
4. Optionally sync to `healthonrails/healthonrails.github.io` when `HEALTHONRAILS_GHIO_TOKEN` is set.

Website deployment (`website-pages.yml`):

1. Trigger on any `website/**` change.
2. Validate website files.
3. Sync `website/index.html` + `website/assets/` to the target site root.

Portal deployment (`docs-portal-pages.yml`):

1. Build `docs_portal/` using MkDocs Material.
2. Update `gh-pages/portal/`.
3. Optionally sync built output to external site repo at `/portal/`.

No manual branch switching or rsync steps are required.

## Deploy to `healthonrails.github.io` (custom-domain flow)

If you publish `annolid.com` from `healthonrails/healthonrails.github.io`:

1. Add secret `HEALTHONRAILS_GHIO_TOKEN` in this repo.
2. Push updates under `book/**` for book content or `website/**` for landing page.
3. The workflows sync output into the target repo root while preserving:
   - `index.html`
   - `assets/`
   - `CNAME`

It uses:

- `scripts/sync_book_build_to_site_root.sh`
- `scripts/sync_landing_page_to_site_root.sh`
- `scripts/sync_docs_portal_to_site_root.sh`
- `scripts/write_legacy_redirects.sh`

to make sync behavior explicit and reproducible.

## Source of truth

- `book/` is the only source of truth for Jupyter Book content.
- `website/` is the source of truth for the public landing page (`annolid.com` root).
- `docs_portal/` is the source of truth for the unified docs portal (`/portal`).
- Redirect/deprecation mapping is maintained in `docs_portal/redirects.md`.
- Local mirror folders like `book/healthonrails.github.io/` are deprecated and ignored by git to prevent divergence.

## Published URLs

- Main docs: `https://<org-or-user>.github.io/<repo>/` (or custom domain root).
- Annolid Book: `https://<org-or-user>.github.io/<repo>/book/`.
- Annolid Docs Portal: `https://<org-or-user>.github.io/<repo>/portal/`.

If this repo is mapped to `annolid.com`, the book URL is `https://annolid.com/book/`.
If this repo is mapped to `annolid.com`, the portal URL is `https://annolid.com/portal/`.
