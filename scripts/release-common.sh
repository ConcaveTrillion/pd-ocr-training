#!/usr/bin/env bash
# Shared release driver for pdomain Python repos.
#
# Repo-local scripts/do-release.sh files configure these variables, then call
# pdomain_release_main:
#   RELEASE_REPO              owner/repo, used only for final release URL text
#   RELEASE_BRANCH            release branch, defaults to main
#   RELEASE_PREFLIGHT         command run before tagging, defaults to make ci-slow
#   RELEASE_VERSION_SOURCE    tags or uv, defaults to tags
#   RELEASE_VERSION_FILES     files committed for RELEASE_VERSION_SOURCE=uv

set -eu

pdomain_release_main() {
    BUMP=${BUMP:-minor}
    FORCE=${FORCE:-0}
    SKIP_PUSH=${SKIP_PUSH:-0}
    RELEASE_BRANCH=${RELEASE_BRANCH:-main}
    RELEASE_PREFLIGHT=${RELEASE_PREFLIGHT:-make ci-slow}
    RELEASE_VERSION_SOURCE=${RELEASE_VERSION_SOURCE:-tags}
    RELEASE_VERSION_FILES=${RELEASE_VERSION_FILES:-pyproject.toml uv.lock}

    if [ "$BUMP" != "major" ] && [ "$BUMP" != "minor" ] && [ "$BUMP" != "patch" ]; then
        echo "ERROR: BUMP must be one of: major, minor, patch (got: $BUMP)" >&2
        exit 2
    fi

    if [ "$RELEASE_VERSION_SOURCE" != "tags" ] && [ "$RELEASE_VERSION_SOURCE" != "uv" ]; then
        echo "ERROR: RELEASE_VERSION_SOURCE must be one of: tags, uv" >&2
        exit 2
    fi

    git fetch origin "$RELEASE_BRANCH" --tags --quiet

    if [ "$FORCE" != "1" ]; then
        if [ -n "$(git status --porcelain)" ]; then
            echo "ERROR: Working tree is dirty. Commit or stash changes first." >&2
            echo "       Set FORCE=1 to skip repo-state guards; preflight still runs." >&2
            exit 1
        fi

        CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
        if [ "$CURRENT_BRANCH" != "$RELEASE_BRANCH" ]; then
            echo "ERROR: Not on $RELEASE_BRANCH (current branch: $CURRENT_BRANCH)." >&2
            echo "       Set FORCE=1 to skip this guard." >&2
            exit 1
        fi

        LOCAL=$(git rev-parse "$RELEASE_BRANCH")
        REMOTE=$(git rev-parse "origin/$RELEASE_BRANCH")
        BASE=$(git merge-base "$RELEASE_BRANCH" "origin/$RELEASE_BRANCH")
        if [ "$LOCAL" != "$REMOTE" ]; then
            if [ "$LOCAL" = "$BASE" ]; then
                echo "ERROR: Local $RELEASE_BRANCH is behind origin/$RELEASE_BRANCH. Pull first." >&2
                exit 1
            elif [ "$REMOTE" = "$BASE" ]; then
                echo "INFO: Local $RELEASE_BRANCH is ahead of origin/$RELEASE_BRANCH and will be pushed."
            else
                echo "ERROR: $RELEASE_BRANCH and origin/$RELEASE_BRANCH have diverged." >&2
                exit 1
            fi
        fi
    else
        echo "WARNING: FORCE=1; skipping repo-state guards. Preflight still runs."
    fi

    if [ "$RELEASE_VERSION_SOURCE" = "uv" ]; then
        echo "Bumping project version with uv: $BUMP"
        uv version --bump "$BUMP"
        VERSION="v$(uv version --short)"
    else
        VERSION=$(pdomain_next_tag_from_git_tags "$BUMP")
    fi

    if git rev-parse -q --verify "refs/tags/$VERSION" >/dev/null; then
        echo "ERROR: Tag $VERSION already exists. Aborting." >&2
        exit 1
    fi

    echo "Next tag: $VERSION (bump=$BUMP)"
    echo ""
    echo "Running preflight: $RELEASE_PREFLIGHT"
    echo ""
    if ! sh -c "$RELEASE_PREFLIGHT"; then
        echo "" >&2
        echo "ERROR: Preflight failed. No tag created." >&2
        exit 1
    fi

    if [ "$RELEASE_VERSION_SOURCE" = "uv" ]; then
        git add $RELEASE_VERSION_FILES
        if git diff --cached --quiet; then
            echo "ERROR: uv version did not change release files: $RELEASE_VERSION_FILES" >&2
            exit 1
        fi
        git commit -m "chore: release $VERSION"
    fi

    echo ""
    echo "Creating annotated tag $VERSION..."
    git tag -a "$VERSION" -m "Release $VERSION"

    if [ "$SKIP_PUSH" = "1" ]; then
        echo "INFO: SKIP_PUSH=1; local commit/tag created but not pushed."
        echo "      To publish later:"
        echo "        git push origin $RELEASE_BRANCH"
        echo "        git push origin $VERSION"
        return 0
    fi

    echo "Pushing $RELEASE_BRANCH and exact tag $VERSION to origin..."
    git push origin "$RELEASE_BRANCH"
    git push origin "$VERSION"

    if [ -f ".github/workflows/release.yml" ]; then
        echo "Triggering release workflow for $VERSION..."
        gh workflow run release.yml --ref "$RELEASE_BRANCH" -f "tag=$VERSION"
    fi

    echo ""
    echo "Released $VERSION."
    if [ -n "${RELEASE_REPO:-}" ]; then
        echo "Release page: https://github.com/$RELEASE_REPO/releases/tag/$VERSION"
    fi
}

pdomain_next_tag_from_git_tags() {
    bump=$1
    latest=$(git tag --list 'v[0-9]*' --sort=-version:refname \
        | grep -E '^v[0-9]+(\.[0-9]+){0,2}$' \
        | head -1 || true)
    if [ -z "$latest" ]; then
        latest="v0.0.0"
    fi

    echo "Latest stable tag: $latest" >&2

    ver_no_v=${latest#v}
    major=$(echo "$ver_no_v" | awk -F. '{print ($1 == "" ? 0 : $1)}')
    minor=$(echo "$ver_no_v" | awk -F. '{print ($2 == "" ? 0 : $2)}')
    patch=$(echo "$ver_no_v" | awk -F. '{print ($3 == "" ? 0 : $3)}')

    if [ "$bump" = "major" ]; then
        major=$((major + 1)); minor=0; patch=0
    elif [ "$bump" = "minor" ]; then
        minor=$((minor + 1)); patch=0
    else
        patch=$((patch + 1))
    fi

    echo "v$major.$minor.$patch"
}
