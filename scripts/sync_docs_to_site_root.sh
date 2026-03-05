#!/usr/bin/env bash
set -euo pipefail

# Sync a built MkDocs docs site into a target repository.
#
# Defaults:
# - source build dir: site_docs
# - destination repo root: book/healthonrails.github.io
# - target path in destination: portal (use "/" to sync into site root)

SRC_DIR="${1:-site_docs}"
DEST_DIR="${2:-book/healthonrails.github.io}"
TARGET_PATH="${3:-portal}"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source docs build directory does not exist: $SRC_DIR" >&2
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

# Keep legacy URLs forwarding to canonical docs pages.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/write_legacy_redirects.sh" "$DEST_DIR"

echo "Docs site synced."
