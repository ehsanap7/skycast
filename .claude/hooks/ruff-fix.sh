#!/usr/bin/env bash
# Lint-fix and format a project Python file after it is edited.
set -euo pipefail

file=$(jq -r '.tool_input.file_path // empty')
[[ "$file" == "$CLAUDE_PROJECT_DIR"/*.py ]] || exit 0

cd "$CLAUDE_PROJECT_DIR"
if ! output=$(uv run --quiet ruff check --fix --quiet "$file" 2>&1); then
  echo "$output" >&2
  exit 2
fi
uv run --quiet ruff format --quiet "$file"
