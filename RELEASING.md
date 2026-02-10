# Releasing Annolid

This repository publishes:
- **PyPI distributions** from GitHub Releases (`.github/workflows/annolid-publish.yml`).
- **Platform executables** via manual workflow dispatch on a `v*` tag (`.github/workflows/release.yml`).
- **Docs to `gh-pages`** on pushes to `main` (`.github/workflows/CI.yml`).

## Recommended release flow
- `make release-patch` (or `make release-minor`, `make release-major`)
- `make release VERSION=X.Y.Z` for an explicit version
- Add `PUSH=1` to push commit and tag automatically
- Add `SKIP_CHECKS=1` only when intentionally bypassing local checks

Under the hood this runs `scripts/release.sh`, which will:
- verify a clean git worktree
- bump both `pyproject.toml` and `annolid/version.py`
- run `pytest`, `python -m build`, and `python -m twine check dist/*`
- create commit `chore(release): vX.Y.Z`
- create annotated tag `vX.Y.Z`
- optionally push branch + tag

## CI safeguards
- Release workflows now verify that the pushed tag matches package version (`v<pyproject version>`) before publishing.
- PyPI publish runs on `v*` tag pushes (and on manually-published GitHub Releases) and requires `PYPI_TOKEN` in repo secrets.
