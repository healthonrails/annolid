#!/usr/bin/env bash
set -euo pipefail

# Write legacy URL redirect stubs into a site root directory.
#
# Defaults:
# - destination site root: book/healthonrails.github.io

DEST_DIR="${1:-book/healthonrails.github.io}"

if [[ ! -d "$DEST_DIR" ]]; then
  echo "Destination directory does not exist: $DEST_DIR" >&2
  exit 1
fi

write_redirect() {
  local relative_path="$1"
  local target_url="$2"
  local out_file="$DEST_DIR/$relative_path"
  mkdir -p "$(dirname "$out_file")"
  cat > "$out_file" <<HTML
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="0; url=${target_url}" />
  <link rel="canonical" href="${target_url}" />
  <title>Redirecting...</title>
</head>
<body>
  <p>Redirecting to <a href="${target_url}">${target_url}</a></p>
</body>
</html>
HTML
}

write_redirect "content/README.html" "https://annolid.com/"
write_redirect "content/how_to_install.html" "https://annolid.com/installation/"
write_redirect "extract_frames.html" "https://annolid.com/workflows/"
write_redirect "install.html" "https://annolid.com/installation/"

touch "$DEST_DIR/.nojekyll"

echo "Legacy redirect stubs written to $DEST_DIR"
