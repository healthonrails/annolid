# Migration Plan

## Goal

Move from fragmented documentation sources to a single, maintainable system
with clear ownership of landing, docs, and book surfaces.

## Target Architecture

- `website/` is the source of truth for `https://annolid.com/` (landing page).
- `docs/` is the source of truth for MkDocs content at `https://annolid.com/portal/`.
- Root docs routes (`/installation/`, `/reference/`, etc.) are preserved as
  compatibility mirrors for existing links.
- `book/` is the source of truth for Jupyter Book at `https://annolid.com/book/`.
- Legacy one-off URLs are handled by generated redirect stubs
  (`scripts/write_legacy_redirects.sh`).

## Phased Plan (Updated)

1. Keep legacy sources available while redirects and links are stabilized.
2. Consolidate canonical docs prose in `docs/` and remove duplicate ownership.
3. Preserve compatibility paths during transition (`/portal/` plus root docs mirrors).
4. Publish Jupyter Book only under `/book/` and keep deploy scripts path-safe.
5. Retire duplicated legacy pages after at least one stable release cycle with
   redirect coverage.

## Current Status

- MkDocs source is consolidated in `docs/`.
- Landing page is consolidated in `website/` and synced to site root.
- Docs deploy to `/portal/` and root compatibility routes.
- Book deploy is corrected to `/book/` with bootstrap recovery when missing.
- Redirect map and generated legacy stubs are documented and active.

## Remaining Work

- Audit and replace any remaining internal links that still point to retired
  legacy source locations.
- Add automated docs link validation in CI for public routes (`/`, `/portal/`,
  `/book/`, and legacy redirects).
- Add an explicit deprecation timeline for legacy source files still kept for
  historical context.
- Continue converting notebook-heavy guidance into concise docs-native guides
  with canonical references.

## Exit Criteria

Migration can be considered complete when all are true:

- No user-facing docs route depends on duplicated source ownership.
- Legacy URLs are either redirected or intentionally retired with notice.
- CI validates docs integrity (strict build plus link checks) on docs changes.
- Deployment consistently produces working `/, /portal/, /book/` surfaces.
