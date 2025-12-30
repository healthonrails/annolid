# Updating the Annolid Jupyter Book + GitHub Pages Site

Annolid’s documentation site (`https://annolid.com`) is deployed from a separate repository:

- Website repo: `https://github.com/healthonrails/healthonrails.github.io`

That repo contains both the website landing page and the Jupyter Book output used for docs pages like `https://annolid.com/content/how_to_install.html`.

This guide shows a safe, repeatable workflow:

1. Update book content (Markdown, TOC, images).
2. Build the Jupyter Book locally.
3. Publish the built HTML to GitHub Pages without overwriting the landing page.

## Repo layout (website repo)

The website repo uses two branches:

- `main`: Jupyter Book sources + a checked-in built copy under `html/` (useful for review/diffs).
- `gh-pages`: the published site root (what GitHub Pages serves).

The `gh-pages` branch contains:

- Website landing page: `index.html` + `assets/`
- Jupyter Book output: `content/`, `tutorials/`, `_static/`, `_sources/`, etc.
- Custom domain config: `CNAME`
- A required `.nojekyll` file (so GitHub Pages serves underscore directories like `_static/`)

## One-time setup

Clone the website repo (suggested location inside this repo’s `book/` folder):

```bash
git clone https://github.com/healthonrails/healthonrails.github.io.git book/healthonrails.github.io
```

Create a Python environment and install book build deps:

```bash
cd book/healthonrails.github.io
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Update book content

Edit/add pages in the website repo:

- `content/*.md`
- `tutorials/*.md`
- `_toc.yml` (navigation)
- `_config.yml` (book config)
- `images/` (book images)

Tip: keep filenames stable where possible; published URLs are filename-based.

## 2) Build the Jupyter Book locally

```bash
cd book/healthonrails.github.io
source .venv/bin/activate

git checkout main
jupyter-book build .
```

Preview the build:

```bash
open _build/html/index.html
```

## 3) Update `main` branch build artifacts (`html/`)

The website repo keeps a copy of the generated book in `html/` on `main`.

```bash
cd book/healthonrails.github.io
git checkout main

jupyter-book build .
rsync -a --delete _build/html/ html/

git add -A
git commit -m "Update Jupyter Book sources and rebuild"
git push origin main
```

## 4) Publish to GitHub Pages (`gh-pages`)

Publishing means copying the built HTML into the `gh-pages` branch root.

Important: the `gh-pages` branch also hosts the website landing page. When syncing book output, do **not** overwrite:

- `index.html`
- `assets/`
- `CNAME`

Recommended workflow:

```bash
cd book/healthonrails.github.io

# Build on main (or rebuild if needed)
git checkout main
jupyter-book build .

# Copy build output aside so it survives the branch switch
tmp_dir="$(mktemp -d)"
rsync -a --delete _build/html/ "${tmp_dir}/"

# Switch to the published branch and sync, preserving the landing page
git checkout gh-pages
rsync -a --delete \
  --exclude 'index.html' \
  --exclude 'assets/' \
  --exclude 'CNAME' \
  "${tmp_dir}/" .

# Critical: ensures GitHub Pages serves `_static/`, `_sources/`, etc.
touch .nojekyll

git add -A
git commit -m "Publish Jupyter Book"
git push origin gh-pages

# Return to main for normal work
git checkout main
```

## Verify deployment

After GitHub Pages finishes rebuilding (often ~1–2 minutes), verify that `_static` assets load:

```bash
curl -I https://annolid.com/_static/styles/theme.css
```

If this returns `404`, the most common issue is that `.nojekyll` is missing from the published root (`gh-pages`).

