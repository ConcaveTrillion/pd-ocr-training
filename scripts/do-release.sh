#!/usr/bin/env bash
# do-release.sh — pre-flight, tag, and push.
#
# Computes the next version from the most recent `v*` tag and the chosen
# bump kind (major / minor / patch), then:
#   1. Verifies repo state (clean tree / on main / up-to-date with origin).
#   2. Runs the full pre-flight (`make ci`).
#   3. Creates a three-component annotated tag (vMAJOR.MINOR.PATCH).
#   4. Pushes main + tag to origin (which fires the GitHub release workflow).
#
# Tag format: always three-component (SemVer, hatch-vcs compatible):
#   patch from v0.1.0 → v0.1.1
#   minor from v0.1.0 → v0.2.0
#   major from v0.1.0 → v1.0.0
#
# Defaults to BUMP=minor.
#
# Escape hatches:
#   FORCE=1     skip the three repo-state guards (dirty tree / branch / origin
#               sync). The pre-flight still runs.
#   SKIP_PUSH=1 create the tag locally but don't push or trigger. Dry-run.
#
# Usage:
#   BUMP=major|minor|patch scripts/do-release.sh

set -eu

BUMP=${BUMP:-minor}
FORCE=${FORCE:-0}
SKIP_PUSH=${SKIP_PUSH:-0}

if [ "$BUMP" != "major" ] && [ "$BUMP" != "minor" ] && [ "$BUMP" != "patch" ]; then
    echo "❌ BUMP must be one of: major, minor, patch (got: $BUMP)" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Repo-state guards (skippable with FORCE=1)
# ---------------------------------------------------------------------------
if [ "$FORCE" != "1" ]; then
    if [ -n "$(git status --porcelain)" ]; then
        echo "❌ Working tree is dirty. Commit or stash changes first." >&2
        echo "   (Set FORCE=1 to override — pre-flight still runs.)" >&2
        exit 1
    fi

    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
    if [ "$CURRENT_BRANCH" != "main" ]; then
        echo "❌ Not on main (current branch: $CURRENT_BRANCH)." >&2
        echo "   Switch to main before tagging. (Set FORCE=1 to override.)" >&2
        exit 1
    fi

    git fetch origin main --quiet
    LOCAL=$(git rev-parse main)
    REMOTE=$(git rev-parse origin/main)
    BASE=$(git merge-base main origin/main)
    if [ "$LOCAL" != "$REMOTE" ]; then
        if [ "$LOCAL" = "$BASE" ]; then
            echo "❌ Local main is behind origin/main. Pull first." >&2
            echo "   (Set FORCE=1 to override.)" >&2
            exit 1
        elif [ "$REMOTE" = "$BASE" ]; then
            echo "  Local main is ahead of origin/main (will be pushed)."
        else
            echo "❌ main and origin/main have diverged." >&2
            exit 1
        fi
    fi
else
    echo "  FORCE=1 — skipping repo-state guards. Pre-flight still runs."
fi

# ---------------------------------------------------------------------------
# Compute next version (always three-component)
# ---------------------------------------------------------------------------
LATEST=$(git tag --list 'v*' --sort=-version:refname | head -1)
if [ -z "$LATEST" ]; then LATEST="v0.0.0"; fi

VER_NO_V=${LATEST#v}
MAJOR=$(echo "$VER_NO_V" | awk -F. '{print ($1 == "" ? 0 : $1)}')
MINOR=$(echo "$VER_NO_V" | awk -F. '{print ($2 == "" ? 0 : $2)}')
PATCH=$(echo "$VER_NO_V" | awk -F. '{print ($3 == "" ? 0 : $3)}')

if [ "$BUMP" = "major" ]; then
    MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0
elif [ "$BUMP" = "minor" ]; then
    MINOR=$((MINOR + 1)); PATCH=0
else
    PATCH=$((PATCH + 1))
fi

VERSION="v$MAJOR.$MINOR.$PATCH"

if git rev-parse -q --verify "refs/tags/$VERSION" >/dev/null; then
    echo "❌ Tag $VERSION already exists. Aborting." >&2
    exit 1
fi

echo "Latest tag: $LATEST"
echo "Next tag:   $VERSION (bump=$BUMP)"

# ---------------------------------------------------------------------------
# Pre-flight (NEVER skipped, even with FORCE=1)
# ---------------------------------------------------------------------------
echo ""
echo "Running pre-flight: make ci"
echo ""
if ! make ci; then
    echo "" >&2
    echo "❌ Pre-flight (make ci) failed. No tag created." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Tag (+ push, unless SKIP_PUSH=1)
# ---------------------------------------------------------------------------
echo ""
echo "Creating annotated tag $VERSION..."
git tag -a "$VERSION" -m "Release $VERSION"

if [ "$SKIP_PUSH" = "1" ]; then
    echo "  SKIP_PUSH=1 — tag created locally but not pushed."
    echo "   To push and release later:"
    echo "     git push origin main --follow-tags"
    exit 0
fi

echo "Pushing main + tag to origin..."
git push origin main --follow-tags

echo ""
echo "Released $VERSION."
echo "   Release page (once workflow finishes): https://github.com/ConcaveTrillion/pd-ocr-training/releases/tag/$VERSION"
