#!/usr/bin/env bash
#
# Package Rename Migration Script
# 
# Usage: ./scripts/rename_package.sh <new-name>
# Example: ./scripts/rename_package.sh surropy

set -e  # Exit on error

NEW_NAME="$1"
OLD_NAME="smlr"

if [ -z "$NEW_NAME" ]; then
    echo "Usage: $0 <new-package-name>"
    echo ""
    echo "Available recommended names:"
    echo "  - surropy (RECOMMENDED)"
    echo "  - polefitter"
    echo "  - responseflow"
    echo "  - spectral-ml"
    echo "  - strengthnet"
    echo ""
    echo "See PACKAGE_NAMES.md for full list"
    exit 1
fi

echo "════════════════════════════════════════════════════════════"
echo "Package Rename: $OLD_NAME → $NEW_NAME"
echo "════════════════════════════════════════════════════════════"

# Confirm with user
read -p "This will rename the package. Continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Step 1: Creating backup..."
git stash
git tag backup-before-rename-$(date +%Y%m%d-%H%M%S)

echo ""
echo "Step 2: Updating pyproject.toml..."
sed -i.bak "s/name = \"$OLD_NAME\"/name = \"$NEW_NAME\"/" pyproject.toml
rm pyproject.toml.bak

echo ""
echo "Step 3: Updating documentation..."
find docs -type f -name "*.md" -exec sed -i.bak "s/\`$OLD_NAME\`/\`$NEW_NAME\`/g" {} \;
find docs -type f -name "*.md.bak" -delete

echo ""
echo "Step 4: Updating README..."
sed -i.bak "s/$OLD_NAME/$NEW_NAME/g" README.md
rm README.md.bak

echo ""
echo "Step 5: Updating workflows..."
find .github/workflows -type f -name "*.yml" -exec sed -i.bak "s/smlr/$NEW_NAME/g" {} \;
find .github/workflows -type f -name "*.yml.bak" -delete

echo ""
echo "Step 6: Checking imports (no changes needed if src structure stays same)..."
# The package imports use src/smlr/ which doesn't need to change
# Only the PyPI package name changes
echo "✓ Source code structure remains: src/smlr/"
echo "  (import smlr will still work)"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "✓ Rename complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Test locally: uv run pytest"
echo "  3. Reserve name on PyPI:"
echo "     https://pypi.org/account/register/"
echo "  4. Update GitHub repo name (Settings → Rename)"
echo "  5. Commit changes:"
echo "     git add -A"
echo "     git commit -m 'Rename package to $NEW_NAME'"
echo "  6. Update remote:"
echo "     git remote set-url origin git@github.com:ascsn/$NEW_NAME.git"
echo ""
echo "To undo: git reset --hard backup-before-rename-*"
