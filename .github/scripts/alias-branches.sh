#!/bin/bash
set -euo pipefail

# Script to create and update version alias branches
# Usage: ./alias-branches.sh <tag> [--dry-run]
#
# Examples:
#   ./alias-branches.sh v1.2.3
#   ./alias-branches.sh v1.2.3 --dry-run

TAG="${1:-}"
DRY_RUN=false

if [[ "${2:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

if [[ -z "$TAG" ]]; then
    echo "Usage: $0 <tag> [--dry-run]" >&2
    exit 1
fi

# Remove 'v' prefix if present
VERSION="${TAG#v}"

# Parse semantic version (stable releases only: v1.2.3, not v0.x.x or v1.2.3-rc0)
if [[ ! $VERSION =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
    echo "Error: Invalid version format: $TAG" >&2
    echo "Expected format: v#.#.# (stable semver only, no prereleases)" >&2
    exit 1
fi

MAJOR="${BASH_REMATCH[1]}"
MINOR="${BASH_REMATCH[2]}"
PATCH="${BASH_REMATCH[3]}"

# Exclude v0.x.x versions (unstable)
if [[ "$MAJOR" == "0" ]]; then
    echo "Skipping v0.x.x version (unstable): $TAG" >&2
    exit 0
fi

echo "Processing tag: $TAG (major: $MAJOR, minor: $MINOR, patch: $PATCH)"

# Function to get latest version matching a pattern
get_latest_version() {
    local pattern="$1"
    git tag -l "$pattern" 2>/dev/null | sed 's/^v//' | sort -V | tail -1
}

# Function to push branch (or simulate in dry-run)
push_branch() {
    local branch="$1"
    local ref="$2"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Would push: $ref -> $branch"
    else
        # Check if origin remote exists
        if git remote get-url origin >/dev/null 2>&1; then
            echo "Pushing: $ref -> $branch"
            git push -f origin "$ref:refs/heads/$branch"
        else
            echo "Creating local branch: $branch -> $ref"
            git branch -f "$branch" "$ref"
        fi
    fi
}

# Check if tag exists
if ! git rev-parse "v$VERSION" >/dev/null 2>&1; then
    echo "Error: Tag v$VERSION does not exist" >&2
    exit 1
fi

# Update v$MAJOR if this is the latest version in this major
echo ""
echo "Checking v$MAJOR branch..."
LATEST_IN_MAJOR=$(get_latest_version "v$MAJOR.*.*")
if [[ "$VERSION" == "$LATEST_IN_MAJOR" ]]; then
    push_branch "v$MAJOR" "v$VERSION"
else
    echo "Skipping v$MAJOR: v$VERSION is not the latest (latest is v$LATEST_IN_MAJOR)"
fi

# Update v$MAJOR.$MINOR if this is the latest patch in this major.minor
echo ""
echo "Checking v$MAJOR.$MINOR branch..."
LATEST_IN_MINOR=$(get_latest_version "v$MAJOR.$MINOR.*")
if [[ "$VERSION" == "$LATEST_IN_MINOR" ]]; then
    push_branch "v$MAJOR.$MINOR" "v$VERSION"
else
    echo "Skipping v$MAJOR.$MINOR: v$VERSION is not the latest (latest is v$LATEST_IN_MINOR)"
fi

# Update stable if this is the latest tag in the greatest major version
echo ""
echo "Checking stable branch..."
MAX_MAJOR=$(git tag -l 'v*.*.*' 2>/dev/null | sed 's/^v//' | cut -d. -f1 | sort -n | tail -1)
if [[ "$MAJOR" == "$MAX_MAJOR" ]]; then
    LATEST_OVERALL=$(get_latest_version "v$MAX_MAJOR.*.*")
    if [[ "$VERSION" == "$LATEST_OVERALL" ]]; then
        push_branch "stable" "v$VERSION"
    else
        echo "Skipping stable: v$VERSION is not the latest in major $MAX_MAJOR (latest is v$LATEST_OVERALL)"
    fi
else
    echo "Skipping stable: major $MAJOR is not the greatest major version (max is $MAX_MAJOR)"
fi

echo ""
echo "Done!"
