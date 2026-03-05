# Redirect and Deprecation Map

This page defines migration targets from legacy Annolid URLs to the current
canonical documentation and site paths.

## Migration Policy

- Keep legacy pages available during transition.
- Add canonical links on legacy pages.
- Prefer stable docs URLs for new links and external references.
- Retire duplicated legacy content only after one full release cycle with clear notice.

## Canonical Public Paths

- Landing page: `https://annolid.com/`
- Docs canonical entry: `https://annolid.com/portal/`
- Docs compatibility routes: `https://annolid.com/installation/`,
  `https://annolid.com/reference/`, `https://annolid.com/tutorials/`, etc.
- Book canonical entry: `https://annolid.com/book/`
- Landing assets: `https://annolid.com/website-assets/`

## Legacy to Canonical URL Mapping

| Legacy URL | Canonical URL |
|---|---|
| `https://annolid.com/content/README.html` | `https://annolid.com/` |
| `https://annolid.com/content/how_to_install.html` | `https://annolid.com/installation/` |
| `https://annolid.com/extract_frames.html` | `https://annolid.com/workflows/` |
| `https://annolid.com/install.html` | `https://annolid.com/installation/` |

## Route Ownership and Mirrors

| Public Path Prefix | Source | Notes |
|---|---|---|
| `/` | `website/` + docs compatibility routes | Landing page at root; root docs routes are preserved for compatibility. |
| `/portal/` | `docs/` (MkDocs) | Canonical docs namespace for stable links. |
| `/book/` | `book/` (Jupyter Book) | Dedicated Jupyter Book surface. |
| `/website-assets/` | `website/assets/` | Landing-only static assets. |

## Repository Source Mapping (Legacy Content to Canonical Docs)

| Legacy Source File | Canonical File |
|---|---|
| `book/content/how_to_install.md` | `docs/installation.md` |
| `docs/source/install.md` | `docs/installation.md` |
| `book/content/extract_frames.md` | `docs/workflows.md` |
| `docs/source/extract_frames.md` | `docs/workflows.md` |

## Generated Legacy Redirect Stubs

The following HTML redirect stubs are generated automatically during deploy by
`scripts/write_legacy_redirects.sh`:

- `/content/README.html` -> `/`
- `/content/how_to_install.html` -> `/installation/`
- `/extract_frames.html` -> `/workflows/`
- `/install.html` -> `/installation/`

## Deployment Notes

- External sync workflow: `.github/workflows/healthonrails-site-sync.yml`
- Redirect generation: `scripts/write_legacy_redirects.sh`
- Book path sync target: `/book/`
