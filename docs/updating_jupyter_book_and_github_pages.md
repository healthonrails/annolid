# Updating Annolid Jupyter Book + GitHub Pages

This repository contains two documentation surfaces:

- Sphinx docs from `docs/` (published at site root).
- Jupyter Book from `book/` (published under `/book/`).
- Landing page source in `website/` (`index.html` + `assets/`) for `annolid.com`.

They are deployed by separate workflows:

- `.github/workflows/CI.yml` deploys Sphinx docs.
- `.github/workflows/book-pages.yml` deploys Jupyter Book to:
  - this repo `gh-pages/book/`
  - `healthonrails/healthonrails.github.io` (optional, when token is configured)
- `.github/workflows/website-pages.yml` deploys landing page updates from `website/` to `healthonrails/healthonrails.github.io`.

## Local workflow

Use the project virtual environment:

```bash
source .venv/bin/activate
```

Install or refresh docs dependencies:

```bash
pip install -r docs/requirements.txt
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
4. Optionally sync to `healthonrails/healthonrails.github.io` if `HEALTHONRAILS_GHIO_TOKEN` is present.

Website deployment (`website-pages.yml`):

1. Trigger on any `website/**` change.
2. Validate website files.
3. Sync `website/index.html` + `website/assets/` to the target site root.

No manual branch switching or rsync steps are required.

## Deploy to `healthonrails.github.io` (custom-domain flow)

If you still publish `annolid.com` from `healthonrails/healthonrails.github.io`:

1. Add secret `HEALTHONRAILS_GHIO_TOKEN` in this repo.
2. Push updates under `book/**` for book content or `website/**` for landing page.
3. The workflows sync output into the target repo root while preserving:
   - `index.html`
   - `assets/`
   - `CNAME`

It uses `scripts/sync_book_build_to_site_root.sh` to make the sync behavior explicit and reproducible.
Landing updates use `scripts/sync_landing_page_to_site_root.sh`.

## Source of truth

- `book/` is the only source of truth for Jupyter Book content.
- `website/` is the source of truth for the public landing page (`annolid.com` root).
- Local mirror folders like `book/healthonrails.github.io/` are deprecated and ignored by git to prevent divergence.

## Published URLs

- Main docs: `https://<org-or-user>.github.io/<repo>/` (or custom domain root).
- Annolid Book: `https://<org-or-user>.github.io/<repo>/book/`.

If this repo is mapped to `annolid.com`, the book URL is `https://annolid.com/book/`.
