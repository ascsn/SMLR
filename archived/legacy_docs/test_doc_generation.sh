#!/bin/bash
# Quick test of documentation auto-generation system

set -e

echo "============================================================"
echo "Testing Documentation Auto-Generation"
echo "============================================================"
echo ""

cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Use the correct python
PYTHON="${PYTHON:-python}"
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
fi

# Test 1: Generate metrics
echo "Test 1: Generating example metrics..."
echo "------------------------------------------------------------"
$PYTHON scripts/generate_example_metrics.py > /dev/null 2>&1
if [ -f "docs/example_metrics.json" ]; then
    echo "✓ Metrics file generated"
    echo ""
    echo "Metrics summary:"
    cat docs/example_metrics.json | python -m json.tool | head -20
    echo "..."
else
    echo "✗ Failed to generate metrics file"
    exit 1
fi
echo ""

# Test 2: Check metrics are valid JSON
echo "Test 2: Validating JSON..."
echo "------------------------------------------------------------"
if $PYTHON -c "import json; json.load(open('docs/example_metrics.json'))" 2>/dev/null; then
    echo "✓ Valid JSON"
else
    echo "✗ Invalid JSON"
    exit 1
fi
echo ""

# Test 3: Check Jinja templates exist
echo "Test 3: Checking Jinja templates in docs..."
echo "------------------------------------------------------------"
if grep -q "{% if metrics" docs/examples/*.md; then
    echo "✓ Jinja templates found in documentation"
    echo ""
    echo "Template files:"
    grep -l "{% if metrics" docs/examples/*.md
else
    echo "✗ No Jinja templates found"
    exit 1
fi
echo ""

echo "============================================================"
echo "✓ All tests passed!"
echo "============================================================"
echo ""
echo "Ready to build docs with:"
echo "  mkdocs serve    # Live preview with auto-reload"
echo "  mkdocs build    # Static site generation"
echo "  ./scripts/build_docs.sh  # Full build script"
echo ""
