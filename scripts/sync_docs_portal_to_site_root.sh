#!/usr/bin/env bash
set -euo pipefail

# Sync a built MkDocs portal into a site repository under /portal.
#
# Defaults:
# - source build dir: site_portal
# - destination repo root: book/healthonrails.github.io

SRC_DIR="${1:-site_portal}"
DEST_DIR="${2:-book/healthonrails.github.io}"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source portal build directory does not exist: $SRC_DIR" >&2
  exit 1
fi

if [[ ! -d "$DEST_DIR" ]]; then
  echo "Destination directory does not exist: $DEST_DIR" >&2
  exit 1
fi

mkdir -p "$DEST_DIR/portal"
rsync -a --delete "$SRC_DIR/" "$DEST_DIR/portal/"

touch "$DEST_DIR/.nojekyll"

# Keep legacy URLs forwarding to canonical portal pages.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/write_legacy_redirects.sh" "$DEST_DIR"

echo "Portal docs synced."
