#!/usr/bin/env bash
# scripts/local-upgrade-deps.sh — upgrade deps then restore local-editable.
#
# Refuses if not in local-dev mode (use `make upgrade-deps` for registry mode).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GIT_COMMON_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir)"
CANONICAL_REPO_ROOT="$(dirname "$GIT_COMMON_DIR")"
# Marker lives in the canonical repo's .venv (shared across worktrees).
MARKER="$CANONICAL_REPO_ROOT/.venv/.pd-local-mode"

say() { echo "[local-upgrade-deps] $*"; }

if [[ ! -f "$MARKER" ]]; then
  echo "ERROR: not in local-dev mode (no marker at $MARKER)." >&2
  echo "       Run 'make upgrade-deps' instead." >&2
  exit 1
fi

# Run uv commands from the canonical repo root so the project .venv is found.
say "→ uv lock --upgrade"
(cd "$CANONICAL_REPO_ROOT" && uv lock --upgrade)
say "→ uv sync"
(cd "$CANONICAL_REPO_ROOT" && uv sync)
say "→ uv sync wiped editables; re-running 'make local-dev' to restore"
make -C "$REPO_ROOT" local-dev
say "✓ local mode restored after upgrade."
