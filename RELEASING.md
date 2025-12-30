# Releasing Annolid

This repository publishes:
- **PyPI distributions** from GitHub Releases (`.github/workflows/annolid-publish.yml`).
- **Platform executables** from `v*` tags (`.github/workflows/release.yml`).
- **Docs to `gh-pages`** on pushes to `main` (`.github/workflows/CI.yml`).

## Bump the version
- Update `pyproject.toml` (`[project].version`).
- Update `annolid/gui/app.py` (`__version__`) so the GUI reports the same version.

## Local sanity checks
- `pytest`
- `python -m build`
- `twine check dist/*`

## Cut the release
- Push a tag: `git tag vX.Y.Z && git push origin vX.Y.Z`
- PyPI publish runs on `v*` tag pushes (and on manually-published GitHub Releases) and requires `PYPI_TOKEN` in repo secrets.
