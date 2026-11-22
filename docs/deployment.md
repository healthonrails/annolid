# Deployment & Docs Authoring Workflow

Annolid publishes multiple documentation surfaces from this repository and ships a local workflow so contributors can edit `website/` and `docs/` side-by-side with live reload, strict link/asset validation, and reproducible showcase generation.

## Source-of-Truth Layout

- `website/`: landing page (`index.html`) and landing assets (`website/assets/` → published as `/website-assets/`)
- `docs/`: canonical MkDocs Material source for `/portal/` and mirrored root compatibility routes
- `docs/imgs/showcase/`: generated screenshots, animated tracking demos, and offline 3D simulation captures
- `docs/assets/reports/`: self-contained interactive example artifacts (such as `mouse083_report.html`)
- `docs/tutorials/`: worked walkthroughs, focused markdown guides, and Jupyter notebooks
- `scripts/site_workflow.py`: unified local staging, live preview/watch server, and strict site validator
- `scripts/generate_docs_showcase.py`: reproducible generator for video/tracking showcase visuals, simulation screenshots, and live HTML5 reports

## Fast Local Authoring Workflow

### 1. Preview Website + Docs Portal Together

Because `website/index.html` references `/website-assets/...` and `/portal/...` (matching production `annolid.com`), use the combined staging preview instead of opening `website/index.html` directly from disk:

```bash
# Stage combined landing page + /portal/ into .site_preview/ and serve at http://127.0.0.1:8080/
make site-preview

# Same preview with automatic live rebuild when website/, docs/, or mkdocs.yml changes
make site-watch
```

You can also invoke the underlying Python tool directly with custom options:

```bash
.venv/bin/python scripts/site_workflow.py preview --watch --port 8080 --open
```

- Editing `website/` syncs instantaneously without waiting for MkDocs.
- Editing `docs/` or `mkdocs.yml` triggers an automatic MkDocs rebuild and re-stage.
- `.site_preview/` is gitignored so local preview artifacts never pollute commits.

### 2. Validate Before Committing

Run one command to validate both `website/` and `docs/` locally:

```bash
make site-check
```

`scripts/site_workflow.py check` verifies:

1. required source files exist (`website/index.html`, `docs/index.md`, `mkdocs.yml`, `docs/requirements_mkdocs.txt`),
2. the production CI title contract in `website/index.html` (`Annolid | Research-Grade Animal Behavior AI`) is preserved,
3. every `website-assets/...` reference resolves in `website/assets/`,
4. every `/portal/...` link in `website/index.html` resolves inside the built MkDocs output,
5. every local image reference across published Markdown docs exists on disk,
6. `mkdocs build --strict --clean` succeeds with zero warnings/errors,
7. no machine-local absolute paths leaked into `docs/`, `website/`, or generated HTML examples, and
8. showcase and website assets stay within size budgets (below the pre-commit 6 MB limit).

### 3. Regenerate Showcase Screenshots, Animations & Simulation Captures

Regenerate offline 3D simulation screenshots (`Two-Mouse`, `13-Keypoint Pose`, `Fly-Brain LIF Connectome`, `FlyBody 3D`) and sync them into `website/assets/images/showcase/`:

```bash
make showcase-sims
```

Regenerate video tracking visuals, animated tracking GIF, behavior storyboard, ethogram/kinematics dashboard, zone analysis chart, offscreen GUI screenshot, and sanitized live HTML5 report from any Annolid-tracked video:

```bash
make showcase-video VIDEO=/path/to/your_tracked_video.mp4
```

Both targets wrap `scripts/generate_docs_showcase.py`, which automatically syncs shared showcase assets from `docs/imgs/showcase/` into `website/assets/images/showcase/` so the website and docs portal stay visually aligned.

## Published Targets

- `/`: landing page from `website/` (`website/assets/` mounted at `/website-assets/`)
- `/portal/`: canonical MkDocs Material documentation portal
- root docs routes such as `/installation/`, `/workflows/`, and `/reference/` for backward compatibility
- `/book/`: notebook-oriented Jupyter Book surface published separately

## GitHub Actions Pipelines

- `docs-quality.yml`: builds MkDocs in strict mode and runs markdownlint across primary docs pages.
- `docs-pages.yml`: builds MkDocs and uploads the HTML artifact for review/validation.
- `healthonrails-site-sync.yml`: syncs landing page, `/portal/`, root compatibility routes, legacy redirects, and `/book/` to `healthonrails/healthonrails.github.io` on `main`.
- `book-pages.yml`: builds the Jupyter Book surface.
- `CI.yml`: runs repository-wide tests and validation checks.

## Deployment Principles

- Keep `docs/` as the canonical user-facing documentation source.
- Always run `make site-check` (or `mkdocs build --strict --clean --config-file mkdocs.yml`) before merging.
- Keep publication reproducible and debuggable with ephemeral local staging (`.site_preview/`).
- Preserve compatibility routes and legacy redirect stubs (`scripts/write_legacy_redirects.sh`).
- Never commit machine-local absolute paths into docs, HTML examples, or scripts.
