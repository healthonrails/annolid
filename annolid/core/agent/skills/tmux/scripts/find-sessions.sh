#!/usr/bin/env bash
set -euo pipefail

tmux list-sessions -F '#{session_name}' 2>/dev/null || true
