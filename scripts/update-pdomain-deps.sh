#!/usr/bin/env bash
# scripts/update-pdomain-deps.sh — bump all sibling pdomain-* deps to registry latest.
#
# Queries pdomain-index-pip (Python) and pdomain-index-npm (npm) for each sibling,
# updates minimum-version pins in pyproject.toml and frontend/package.json,
# then leaves the diff staged for human review. Does NOT commit.
#
# Auto-flips around local-dev mode if the .venv/.pdomain-local-mode marker is set.
# Idempotent: prints "✓ <name> already at <version>" if already at latest.
#
# Usage: make update-pdomain-deps   (or ./scripts/update-pdomain-deps.sh directly)
set -euo pipefail

# ─── Repo-specific config (edit per-repo during M3–M9 rollout) ───────────────
PY_SIBLINGS=(pdomain-book-tools)
NPM_SIBLINGS=()   # no npm siblings for pdomain-ocr-training
PD_INDEX_PIP="https://pdomain.github.io/pdomain-index-pip"
PD_INDEX_NPM="https://pdomain.github.io/pdomain-index-npm"
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GIT_COMMON_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir)"
CANONICAL_REPO_ROOT="$(dirname "$GIT_COMMON_DIR")"
MARKER="$CANONICAL_REPO_ROOT/.venv/.pdomain-local-mode"

say() { echo "[update-pdomain-deps] $*"; }

# ─── Step 1: Detect local-dev mode ───────────────────────────────────────────
LOCAL_DEV_ACTIVE=false
if [[ -f "$MARKER" ]]; then
  LOCAL_DEV_ACTIVE=true
  say "→ local-dev mode detected (marker: $MARKER)"
fi

# ─── Step 2: Flip out of local-dev mode if needed ────────────────────────────
if [[ "$LOCAL_DEV_ACTIVE" == "true" ]]; then
  say "→ flipping out of local-dev mode (registry resolution needed)"
  (cd "$CANONICAL_REPO_ROOT" && uv sync --group dev)
fi

# ─── Step 3: Query pdomain-index-pip for each Python sibling ──────────────────────
# Track if any Python dep changed (to avoid redundant uv lock+sync).
PY_CHANGED=false

for sibling in "${PY_SIBLINGS[@]}"; do
  index_url="$PD_INDEX_PIP/simple/$sibling/"
  say "→ querying $index_url"

  # Fetch the PEP 503 simple index page; fail loudly if unreachable/404.
  html=$(curl -sSf "$index_url" 2>&1) || {
    echo "ERROR: could not fetch $index_url" >&2
    echo "       Network issue or sibling not yet seeded in pdomain-index-pip." >&2
    echo "       No changes made." >&2
    exit 1
  }

  # Parse: extract all href filenames, pull version segment, find max.
  # Handles both wheel (<name>-<ver>-py3-none-any.whl) and sdist (<name>-<ver>.tar.gz).
  # Strategy: match <name>-<ver> prefix, then strip everything from the first
  # hyphen or extension that follows the version number.
  pkg_norm="${sibling//-/_}"
  latest=$(
    echo "$html" \
      | grep -oP 'href="[^"]*"' \
      | sed 's/href="//;s/"//' \
      | grep -oP "${pkg_norm}-[0-9][^/\"#]*" \
      | sed "s/${pkg_norm}-//" \
      | python3 -c "
import sys, re
vers = []
for line in sys.stdin:
  line = line.strip()
  # strip trailing filename junk: everything from first non-version char
  m = re.match(r'^([0-9]+(?:\.[0-9]+)*(?:[._-]?(?:a|b|rc|alpha|beta|dev)[0-9]*)?).*', line)
  if m:
    vers.append(m.group(1))
if vers:
  from packaging.version import Version
  print(str(max(Version(v) for v in vers)))
"
  )

  if [[ -z "$latest" ]]; then
    echo "ERROR: pdomain-index-pip returned a page for $sibling but no wheel/sdist found." >&2
    echo "       Index may not be seeded yet. No changes made." >&2
    exit 1
  fi

  say "   latest $sibling = $latest"

  # Read current pinned minimum from pyproject.toml, e.g. "pdomain-book-tools>=0.14.1"
  current=$(grep -oP "\"$sibling>=[0-9]+\.[0-9]+[^\"]*\"" "$CANONICAL_REPO_ROOT/pyproject.toml" \
    | grep -oP ">=[0-9]+\.[0-9]+[^\"]+" \
    | sed 's/>=//' \
    || true)

  if [[ "$current" == "$latest" ]]; then
    say "   ✓ $sibling already at $latest — skipping"
    continue
  fi

  say "   pinning $sibling: $current → $latest"
  # Update minimum-version pin in pyproject.toml (handles >=X.Y.Z patterns).
  sed -i "s|\"$sibling>=[^\"]*\"|\"$sibling>=$latest\"|g" \
    "$CANONICAL_REPO_ROOT/pyproject.toml"
  PY_CHANGED=true
done

# Refresh lockfile + venv only if something changed.
if [[ "$PY_CHANGED" == "true" ]]; then
  say "→ running uv lock"
  (cd "$CANONICAL_REPO_ROOT" && uv lock)
  say "→ running uv sync"
  (cd "$CANONICAL_REPO_ROOT" && uv sync --group dev)
fi

# ─── Step 4: Query pdomain-index-npm for each npm sibling ─────────────────────────
FRONTEND_DIR="$REPO_ROOT/frontend"
NPM_CHANGED=false

if [[ -d "$FRONTEND_DIR" ]]; then
  for sibling in "${NPM_SIBLINGS[@]}"; do
    pkg="@pdomain/$sibling"
    index_url="$PD_INDEX_NPM/@pdomain/$sibling/"
    say "→ querying $index_url"

    # Fetch npm-style package metadata JSON; fail loudly if unreachable/404.
    meta=$(curl -sSf "$index_url" 2>&1) || {
      echo "ERROR: could not fetch $index_url" >&2
      echo "       Network issue or sibling not yet seeded in pdomain-index-npm." >&2
      echo "       No changes made." >&2
      exit 1
    }

    # dist-tags.latest from JSON.
    latest=$(echo "$meta" | python3 -c "
import sys, json
try:
  d = json.load(sys.stdin)
  print(d['dist-tags']['latest'])
except Exception as e:
  print('ERROR: ' + str(e), file=sys.stderr)
  sys.exit(1)
" 2>&1) || {
      echo "ERROR: failed to parse dist-tags.latest from $index_url response." >&2
      echo "       Response: $meta" >&2
      echo "       No changes made." >&2
      exit 1
    }

    say "   latest $pkg = $latest"

    # Read current version from frontend/package.json (strips leading ^ or ~).
    current=$(python3 -c "
import sys, json
with open('$FRONTEND_DIR/package.json') as f:
  d = json.load(f)
deps = {**d.get('dependencies', {}), **d.get('devDependencies', {})}
v = deps.get('$pkg', '')
print(v.lstrip('^~'))
" 2>&1) || {
      echo "ERROR: failed to read $pkg version from $FRONTEND_DIR/package.json" >&2
      exit 1
    }

    if [[ "$current" == "$latest" ]]; then
      say "   ✓ $pkg already at $latest — skipping"
      continue
    fi

    say "   pinning $pkg: $current → ^$latest"
    # Update version in package.json (preserves ^ prefix convention).
    python3 -c "
import json, sys

with open('$FRONTEND_DIR/package.json') as f:
  d = json.load(f)

pkg = '$pkg'
new_ver = '^$latest'
for section in ('dependencies', 'devDependencies', 'peerDependencies'):
  if section in d and pkg in d[section]:
    d[section][pkg] = new_ver
    break

with open('$FRONTEND_DIR/package.json', 'w') as f:
  json.dump(d, f, indent=2)
  f.write('\n')
"
    NPM_CHANGED=true
  done

  if [[ "$NPM_CHANGED" == "true" ]]; then
    say "→ running pnpm install --no-frozen-lockfile"
    (cd "$FRONTEND_DIR" && pnpm install --no-frozen-lockfile)
  fi
fi

# ─── Step 5: Restore local-dev mode if we flipped out ────────────────────────
if [[ "$LOCAL_DEV_ACTIVE" == "true" ]]; then
  say "→ restoring local-dev mode"
  make -C "$REPO_ROOT" local-dev
fi

# ─── Step 6: Show staged diff summary ────────────────────────────────────────
say ""
say "── diff summary ──────────────────────────────────────────────"
git -C "$CANONICAL_REPO_ROOT" diff --stat -- \
  pyproject.toml \
  uv.lock \
  "$FRONTEND_DIR/package.json" \
  "$FRONTEND_DIR/pnpm-lock.yaml" \
  2>/dev/null || true
say "──────────────────────────────────────────────────────────────"
say "Review the diff above, then commit:"
say "  git diff -- pyproject.toml uv.lock frontend/package.json frontend/pnpm-lock.yaml"
say "  git add pyproject.toml uv.lock frontend/package.json frontend/pnpm-lock.yaml"
say "  git commit -m 'chore: bump pdomain-* sibling deps to registry latest'"
