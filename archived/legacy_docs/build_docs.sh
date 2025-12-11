#!/bin/bash
# Build documentation with updated example metrics and plots

set -e

echo "============================================================"
echo "Building SMLR Documentation"
echo "============================================================"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Step 1: Generate example metrics
echo "Step 1: Generating example metrics..."
echo "------------------------------------------------------------"
python scripts/generate_example_metrics.py
echo ""

# Step 2: Build documentation
echo "Step 2: Building documentation with MkDocs..."
echo "------------------------------------------------------------"
mkdocs build --clean
echo ""

echo "============================================================"
echo "✓ Documentation built successfully!"
echo "============================================================"
echo ""
echo "Output location: site/"
echo ""
echo "To serve locally, run:"
echo "  mkdocs serve"
echo ""
