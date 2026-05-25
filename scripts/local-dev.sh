#!/usr/bin/env bash
# scripts/local-dev.sh — switch to local-editable sibling pd-* deps.
#
# Calls local-setup first to ensure siblings are cloned.
# Then installs editable siblings (Python only — lib repo, no npm), writes marker.
set -euo pipefail

# Repo-specific: Python siblings only (lib repo — no npm siblings).
PY_SIBLINGS=(pd-book-tools)
NPM_SIBLINGS=()

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GIT_COMMON_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir)"
CANONICAL_REPO_ROOT="$(dirname "$GIT_COMMON_DIR")"
WORKSPACE_ROOT="$(dirname "$CANONICAL_REPO_ROOT")"
# Marker lives in the canonical repo's .venv (not the worktree's fake .venv).
MARKER="$CANONICAL_REPO_ROOT/.venv/.pd-local-mode"

say() { echo "[local-dev] $*"; }

# Pre-flight: siblings must exist
make -C "$REPO_ROOT" local-setup

# Python: install editable (run from canonical repo root so project .venv is discovered).
for s in "${PY_SIBLINGS[@]}"; do
  say "→ installing editable: $s"
  (cd "$CANONICAL_REPO_ROOT" && uv pip install --no-deps -e "$WORKSPACE_ROOT/$s")
done

# Write marker
mkdir -p "$(dirname "$MARKER")"
touch "$MARKER"
say "✓ marker written: $MARKER"

say "✓ local-dev mode active. Run 'make local-check' to verify."
