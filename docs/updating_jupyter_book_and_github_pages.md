# Updating Annolid Docs, Book, and GitHub Pages

This repository uses a unified documentation source plus legacy references:

- MkDocs docs from `docs/` (canonical for `annolid.com`, mirrored at `/portal/` for compatibility).
- Jupyter Book from `book/` (published under `/book/`).
- `website/` retained as a legacy compatibility surface.

## Active Workflows

- `.github/workflows/docs-quality.yml`: source validation, strict MkDocs build, markdown lint.
- `.github/workflows/docs-pages.yml`: build and validate the MkDocs site, then upload the HTML artifact.
- `.github/workflows/book-pages.yml`: build and validate the Jupyter Book output, then upload the HTML artifact.
- `.github/workflows/healthonrails-site-sync.yml`: sync `annolid.com` output to `healthonrails/healthonrails.github.io`.
- `.github/workflows/CI.yml`: core tests plus conditional strict docs build verification.

## Local Workflow

Use the project virtual environment:

```bash
source .venv/bin/activate
```

Install docs dependencies:

```bash
uv pip install -r docs/requirements_mkdocs.txt
```

Build docs locally:

```bash
mkdocs build --strict --clean --config-file mkdocs.yml
```

Build book locally:

```bash
uv pip install -r book/requirements.txt
jupyter-book build book
```

## Deployment Behavior

Docs build (`docs-pages.yml`):

1. Build `docs/` with strict MkDocs checks.
2. Upload the generated HTML as a workflow artifact for inspection/debugging.

Book build (`book-pages.yml`):

1. Build `book/` HTML.
2. Upload the generated HTML as a workflow artifact for inspection/debugging.

External site sync (`healthonrails-site-sync.yml`):

1. Detect changed surfaces (`docs`, `book`, legacy `website`).
2. Build only changed surfaces.
3. Sync docs to custom-domain root (`/`) and compatibility path (`/portal`).
4. Sync book output to `/book`.
5. Write legacy redirect stubs.
6. Skip external sync completely if nothing deployable changed.

## Deploy to `healthonrails.github.io`

To publish `annolid.com` from `healthonrails/healthonrails.github.io`:

1. Set repository secret `HEALTHONRAILS_GHIO_TOKEN`.
2. Ensure the target repository exists at `healthonrails/healthonrails.github.io`.
3. Push changes under `docs/**` and/or `book/**`.
4. The sync workflow will create the target `gh-pages` branch on first deploy if it does not exist yet.
5. The workflow fails fast when the deploy token is missing instead of silently skipping publication.

## Source of Truth

- `docs/` is the source of truth for the public docs site.
- `book/` is the source of truth for Jupyter Book content.
- Redirect/deprecation mapping lives in `docs/redirects.md`.

## Published URLs

- Canonical docs: `https://annolid.com/`
- Docs compatibility path: `https://annolid.com/portal/`
- Book: `https://annolid.com/book/`
