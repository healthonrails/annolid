#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/release.sh <patch|minor|major|X.Y.Z> [--push] [--skip-checks]

Examples:
  scripts/release.sh patch
  scripts/release.sh 1.5.3 --push

What it does:
  1) Verifies clean git worktree
  2) Bumps version in pyproject.toml and annolid/version.py
  3) Runs checks (pytest, build, twine check) unless --skip-checks
  4) Commits: "chore(release): vX.Y.Z"
  5) Creates annotated tag: vX.Y.Z
  6) Optionally pushes branch + tag with --push
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

TARGET="$1"
shift

PUSH=0
SKIP_CHECKS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push)
      PUSH=1
      ;;
    --skip-checks)
      SKIP_CHECKS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Git worktree is not clean. Commit/stash changes first." >&2
  exit 1
fi

CURRENT="$(sed -nE 's/^version = "([0-9]+\.[0-9]+\.[0-9]+)"/\1/p' pyproject.toml | head -n1)"
if [[ -z "$CURRENT" ]]; then
  echo "Could not read current version from pyproject.toml" >&2
  exit 1
fi

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

if [[ "$TARGET" == "patch" ]]; then
  NEXT="${MAJOR}.${MINOR}.$((PATCH + 1))"
elif [[ "$TARGET" == "minor" ]]; then
  NEXT="${MAJOR}.$((MINOR + 1)).0"
elif [[ "$TARGET" == "major" ]]; then
  NEXT="$((MAJOR + 1)).0.0"
elif [[ "$TARGET" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  NEXT="$TARGET"
else
  echo "Invalid version target: $TARGET" >&2
  usage
  exit 1
fi

if git rev-parse -q --verify "refs/tags/v${NEXT}" >/dev/null; then
  echo "Tag v${NEXT} already exists." >&2
  exit 1
fi

echo "Bumping version: ${CURRENT} -> ${NEXT}"

sed -i.bak -E "s/^version = \"[0-9]+\.[0-9]+\.[0-9]+\"/version = \"${NEXT}\"/" pyproject.toml
sed -i.bak -E "s/^__version__ = \"[0-9]+\.[0-9]+\.[0-9]+\"/__version__ = \"${NEXT}\"/" annolid/version.py
rm -f pyproject.toml.bak annolid/version.py.bak

if [[ "$SKIP_CHECKS" -eq 0 ]]; then
  echo "Running release checks..."
  python -m pytest
  python -m build
  python -m twine check dist/*
fi

git add pyproject.toml annolid/version.py
git commit -m "chore(release): v${NEXT}"
git tag -a "v${NEXT}" -m "Release v${NEXT}"

if [[ "$PUSH" -eq 1 ]]; then
  BRANCH="$(git branch --show-current)"
  git push origin "${BRANCH}"
  git push origin "v${NEXT}"
  echo "Pushed ${BRANCH} and tag v${NEXT}"
else
  echo "Created commit + tag locally."
  echo "Push when ready:"
  echo "  git push origin $(git branch --show-current)"
  echo "  git push origin v${NEXT}"
fi
