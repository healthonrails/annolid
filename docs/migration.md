# Migration Plan

## Goal

Move from fragmented documentation sources to a single, professional, maintainable docs system.

## Phased Plan

1. Keep legacy sources available while redirects and links are stabilized.
2. Use `docs/` as the canonical docs source.
3. Reduce duplicated prose by replacing repeats with concise summaries and canonical links.
4. Migrate high-value guides first (installation, workflows, tutorials, deployment).
5. Retire duplicated legacy pages after at least one stable release cycle.

## Current Status

- MkDocs site source consolidated into `docs/`.
- Automated deployment publishes docs at root and `/portal` compatibility path.
- Installation and workflow guidance consolidated.
- Tutorials and reference index migrated into docs-native pages.

## Next Steps

- Add additional redirects for any remaining external links to retired pages.
- Add link-checking and docs spelling checks.
- Continue improving canonical docs-native guides in `docs/`.
