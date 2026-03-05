#!/usr/bin/env bash
set -euo pipefail

# Sync landing page assets into a site repo root.
#
# Defaults:
# - source website dir: website
# - destination site repo root: book/healthonrails.github.io

SRC_DIR="${1:-website}"
DEST_DIR="${2:-book/healthonrails.github.io}"

if [[ ! -f "$SRC_DIR/index.html" ]]; then
  echo "Missing landing page source: $SRC_DIR/index.html" >&2
  exit 1
fi

if [[ ! -d "$DEST_DIR" ]]; then
  echo "Destination directory does not exist: $DEST_DIR" >&2
  exit 1
fi

mkdir -p "$DEST_DIR/assets"
rsync -a --delete \
  --exclude '.git/' \
  "$SRC_DIR/assets/" "$DEST_DIR/assets/"
cp "$SRC_DIR/index.html" "$DEST_DIR/index.html"

# Keep legacy URLs forwarding to canonical portal pages.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/write_legacy_redirects.sh" "$DEST_DIR"

echo "Landing page synced."
