#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <target-pane> <needle> [timeout_seconds]" >&2
  exit 2
fi

pane="$1"
needle="$2"
timeout="${3:-30}"
start="$(date +%s)"

while true; do
  output="$(tmux capture-pane -p -t "$pane" 2>/dev/null || true)"
  if grep -Fq "$needle" <<<"$output"; then
    exit 0
  fi
  now="$(date +%s)"
  if (( now - start >= timeout )); then
    exit 1
  fi
  sleep 1
done
