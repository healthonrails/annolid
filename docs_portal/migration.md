# Migration Plan

## Goal

Move from fragmented documentation sources to a professional, maintainable portal structure.

## Phased Plan

1. Keep existing docs systems live to avoid breaking links.
2. Use `docs_portal/` as the canonical user entry point.
3. Reduce duplicated content by replacing repeated prose with concise summaries and canonical links.
4. Move high-value guides into portal-native pages first (installation, workflow, troubleshooting).
5. Retire legacy duplicated pages only after redirects and link checks are in place.

## Immediate Status

- MkDocs Material portal added.
- Automated deployment added for site root (`/`) with a compatibility mirror at `/portal`.
- Installation and workflow guidance consolidated at portal level.
- Tutorials and reference index migrated into portal-native pages.

## Next Steps

- Add redirects for deprecated duplicated pages.
- Add docs quality checks (link checker, spelling, markdown lint).
- Expand portal with troubleshooting and API reference index pages.
