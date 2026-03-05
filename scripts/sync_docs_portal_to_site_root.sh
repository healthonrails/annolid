#!/usr/bin/env bash
set -euo pipefail

# Sync a built MkDocs portal into a site repository.
#
# Defaults:
# - source build dir: site_portal
# - destination repo root: book/healthonrails.github.io
# - target path in destination: portal (use "/" to sync into site root)

SRC_DIR="${1:-site_portal}"
DEST_DIR="${2:-book/healthonrails.github.io}"
TARGET_PATH="${3:-portal}"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source portal build directory does not exist: $SRC_DIR" >&2
  exit 1
fi

if [[ ! -d "$DEST_DIR" ]]; then
  echo "Destination directory does not exist: $DEST_DIR" >&2
  exit 1
fi

if [[ "$TARGET_PATH" == "/" || "$TARGET_PATH" == "." || "$TARGET_PATH" == "root" ]]; then
  # Root sync powers annolid.com directly; preserve /book and CNAME.
  rsync -a --delete \
    --exclude 'book/' \
    --exclude 'CNAME' \
    "$SRC_DIR/" "$DEST_DIR/"
else
  mkdir -p "$DEST_DIR/$TARGET_PATH"
  rsync -a --delete "$SRC_DIR/" "$DEST_DIR/$TARGET_PATH/"
fi

touch "$DEST_DIR/.nojekyll"

# Keep legacy URLs forwarding to canonical portal pages.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/write_legacy_redirects.sh" "$DEST_DIR"

echo "Portal docs synced."
