#!/usr/bin/env bash
# scripts/local-setup.sh — clone missing pdomain-* sibling repos into the workspace.
#
# Idempotent: skips siblings that already exist.
# Does NOT switch the repo into local-dev mode (use `make local-dev` for that).
set -euo pipefail

# Repo-specific: list of sibling pdomain-* GitHub repo names this repo depends on.
SIBLINGS=(pdomain-book-tools)

# Workspace root = parent of the canonical git repo dir (works in both normal checkouts and worktrees).
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GIT_COMMON_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir)"
CANONICAL_REPO_ROOT="$(dirname "$GIT_COMMON_DIR")"
WORKSPACE_ROOT="$(dirname "$CANONICAL_REPO_ROOT")"

say() { echo "[local-setup] $*"; }

for sibling in "${SIBLINGS[@]}"; do
  if [[ -d "$WORKSPACE_ROOT/$sibling" ]]; then
    say "✓ $sibling already cloned at $WORKSPACE_ROOT/$sibling"
  else
    say "→ cloning $sibling…"
    gh repo clone "ConcaveTrillion/$sibling" "$WORKSPACE_ROOT/$sibling"
    say "✓ $sibling cloned"
  fi
done

say "done. Run 'make local-dev' to install with editable siblings."
