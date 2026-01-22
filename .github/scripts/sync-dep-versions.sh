#!/bin/bash
set -e

# Sync workspace member dependency versions in main Cargo.toml
# This script reads versions from crates/*/Cargo.toml and updates
# the dependency references in the root Cargo.toml

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAIN_CARGO_TOML="$PROJECT_ROOT/Cargo.toml"

echo "🔄 Syncing workspace dependency versions..."

# Get the version from a crate's Cargo.toml
get_crate_version() {
    local crate_path="$1"
    local cargo_toml="$crate_path/Cargo.toml"

    if [[ ! -f "$cargo_toml" ]]; then
        echo "❌ Error: $cargo_toml not found"
        exit 1
    fi

    # Extract version from [package] section
    grep '^version = ' "$cargo_toml" | head -1 | sed -E 's/version = "(.*)"/\1/' | tr -d ' '
}

# Update dependency version in main Cargo.toml
update_dep_version() {
    local dep_name="$1"
    local new_version="$2"

    # Create a temporary file
    local temp_file=$(mktemp)

    # Use awk to update the specific dependency line
    awk -v dep="$dep_name" -v ver="$new_version" '
    {
        if ($0 ~ dep " = \\{ path = \"crates/" dep "\", version = ") {
            sub(/version = "[^"]*"/, "version = \"" ver "\"")
        }
        print
    }
    ' "$MAIN_CARGO_TOML" > "$temp_file"

    # Replace original file
    mv "$temp_file" "$MAIN_CARGO_TOML"
}

# Check if dependencies exist in main Cargo.toml
check_dependency_exists() {
    local dep_name="$1"
    if ! grep -q "$dep_name = { path = \"crates/$dep_name\"" "$MAIN_CARGO_TOML"; then
        echo "⚠️  Warning: $dep_name not found in $MAIN_CARGO_TOML"
        return 1
    fi
    return 0
}

# Process metalogue-mlx
if [[ -d "$PROJECT_ROOT/crates/metalogue-mlx" ]]; then
    mlx_version=$(get_crate_version "$PROJECT_ROOT/crates/metalogue-mlx")
    echo "📦 metalogue-mlx version: $mlx_version"

    if check_dependency_exists "metalogue-mlx"; then
        update_dep_version "metalogue-mlx" "$mlx_version"
        echo "✅ Updated metalogue-mlx to version $mlx_version"
    fi
fi

# Process metalogue-traits
if [[ -d "$PROJECT_ROOT/crates/metalogue-traits" ]]; then
    traits_version=$(get_crate_version "$PROJECT_ROOT/crates/metalogue-traits")
    echo "📦 metalogue-traits version: $traits_version"

    if check_dependency_exists "metalogue-traits"; then
        update_dep_version "metalogue-traits" "$traits_version"
        echo "✅ Updated metalogue-traits to version $traits_version"
    fi
fi

echo "🎉 Dependency versions synced successfully!"
