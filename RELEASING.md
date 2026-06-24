# Releasing Annolid

Packaging tiers, artifact policy, signing status, and desktop binary expectations are documented in [Packaging and Distribution](docs/packaging.md).

This repository publishes:
- **PyPI distributions** from GitHub Releases (`.github/workflows/annolid-publish.yml`).
- **Platform executables** from `v*` tag pushes (`.github/workflows/release.yml`).
- **Docs to `gh-pages`** on pushes to `main` (`.github/workflows/CI.yml`).

## Recommended release flow
- `make release-patch` (or `make release-minor`, `make release-major`)
- `make release VERSION=X.Y.Z` for an explicit version
- Add `PUSH=1` to push commit and tag automatically
- Add `SKIP_CHECKS=1` only when intentionally bypassing local checks

Under the hood this runs `scripts/release.sh`, which will:
- verify a clean git worktree
- bump both `pyproject.toml` and `annolid/version.py`
- run `pytest`, `python -m build`, and
  `python -m twine check dist/*.whl dist/*.tar.gz`
- create commit `chore(release): vX.Y.Z`
- create annotated tag `vX.Y.Z`
- optionally push branch + tag

## CI safeguards
- Release workflows now verify that the pushed tag matches package version (`v<pyproject version>`) before publishing.
- PyPI publish runs on `v*` tag pushes (and on manually-published GitHub Releases) and requires `PYPI_TOKEN` in repo secrets.

## Recovering desktop assets

If a bundle-only release check fails after the tag and source package are
published, fix the release tooling on `main` and run:

```bash
gh workflow run release.yml --ref main \
  -f build_binaries=true \
  -f release_tag=vX.Y.Z
```

The recovery workflow refuses to build unless `annolid/`, `annolid.spec`, and
`pyproject.toml` still match the release tag and the tag matches the package
version. This allows release-tooling repairs without moving a published tag.
