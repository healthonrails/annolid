#!/usr/bin/env bash
set -euo pipefail

# Sync a Jupyter Book build output directory into a site repo root while
# preserving a landing page and custom domain files.
#
# Defaults assume:
# - source build: book/_build/html
# - destination repo root: book/healthonrails.github.io

SRC_DIR="${1:-book/_build/html}"
DEST_DIR="${2:-book/healthonrails.github.io}"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "Source build directory does not exist: $SRC_DIR" >&2
  exit 1
fi

if [[ ! -d "$DEST_DIR" ]]; then
  echo "Destination directory does not exist: $DEST_DIR" >&2
  exit 1
fi

echo "Syncing book build from '$SRC_DIR' -> '$DEST_DIR'"
rsync -a --delete \
  --exclude '.git/' \
  --exclude 'index.html' \
  --exclude 'assets/' \
  --exclude 'CNAME' \
  "$SRC_DIR"/ "$DEST_DIR"/

# Required so Pages serves underscore paths like _static.
touch "$DEST_DIR/.nojekyll"

# Keep legacy URLs forwarding to canonical portal pages.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/write_legacy_redirects.sh" "$DEST_DIR"

echo "Sync complete."
