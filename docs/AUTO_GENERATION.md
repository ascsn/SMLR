---
render_macros: false
---

# Documentation Auto-Generation System

This document explains how the SMLR documentation automatically generates and updates performance metrics from running examples.

## Overview

The documentation build process automatically:
1. Runs all example scripts
2. Captures performance metrics from their output
3. Generates plots and saves them to `docs/examples/figs/`
4. Injects metrics into documentation using Jinja templates
5. Builds the final documentation with current, accurate data

This ensures the documentation always reflects the actual performance of the current codebase.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   mkdocs serve/build                    │
│                           ↓                             │
│              ┌────────────────────────┐                 │
│              │  docs/hooks.py         │                 │
│              │  (on_pre_build)        │                 │
│              └────────┬───────────────┘                 │
│                       ↓                                 │
│    ┌─────────────────────────────────────────┐         │
│    │  scripts/generate_example_metrics.py    │         │
│    │  - Runs all example scripts             │         │
│    │  - Parses output with regex             │         │
│    │  - Extracts metrics                     │         │
│    │  - Saves to docs/example_metrics.json   │         │
│    └─────────────┬───────────────────────────┘         │
│                  ↓                                      │
│    ┌─────────────────────────────────────────┐         │
│    │  docs/examples/*.md (Jinja templates)   │         │
│    │  {% if metrics.high_dim_5d %}           │         │
│    │  {{ metrics.high_dim_5d.error }}        │         │
│    │  {% endif %}                            │         │
│    └─────────────┬───────────────────────────┘         │
│                  ↓                                      │
│    ┌─────────────────────────────────────────┐         │
│    │       Final HTML Documentation          │         │
│    │  (with current metrics embedded)        │         │
│    └─────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

## Files and Components

### Core Scripts

- **`scripts/generate_example_metrics.py`**: Main script that runs examples and extracts metrics
  - Runs each example with specific parameters
  - Uses regex to parse output and extract metrics
  - Saves results to JSON file
  - Extensible: easy to add new examples

- **`docs/hooks.py`**: MkDocs hook that runs before build
  - Calls `generate_example_metrics.py`
  - Loads JSON into MkDocs config
  - Provides Jinja filters for formatting (percent, decimal)

- **`scripts/build_docs.sh`**: Convenience script for full build
  - Runs metrics generation
  - Builds documentation
  - Shows output location

### Generated Files

- **`docs/example_metrics.json`**: Auto-generated metrics (gitignored)
  ```json
  {
    "high_dim_5d": {
      "mean_l2_error": 0.2443,
      "min_error": 0.1434,
      "max_error": 0.3751
    },
    "acoustic": { ... },
    "materials": { ... }
  }
  ```

### Documentation Templates

Documentation files use Jinja2 templating with fallbacks:

```markdown
{% if metrics and metrics.high_dim_5d %}
**Current measured performance:**

| Metric | Value |
|--------|-------|
| Mean L² error | {{ metrics.high_dim_5d.mean_l2_error | percent }} |
{% else %}
With default settings:

| Metric | Typical Value |
|--------|---------------|
| Mean L² error | ~24% |
{% endif %}
```

**Key features:**
- Conditional rendering with `{% if metrics %}`
- Automatic fallback to static values if metrics unavailable
- Custom Jinja filters: `percent`, `decimal`
- Always shows accurate, current data when available

## Usage

### Building Documentation

**Standard build (with smart caching):**
```bash
mkdocs serve  # or mkdocs build
```
The system automatically:
- Uses cached metrics if no example files changed
- Regenerates only when `.py` files in `examples/` are modified
- Tracks changes to the generation script itself

**Fast build (skip metrics entirely):**
```bash
SKIP_METRICS=1 mkdocs serve
```
Uses existing metrics file without checking for changes. Fastest option for documentation-only edits.

**Force regeneration:**
```bash
rm docs/example_metrics.json && mkdocs build
```
Delete the cache to force a full regeneration.

**Method 2: Full build script**
```bash
./scripts/build_docs.sh
```
Explicitly shows each step of the process.

**Method 3: Manual control**
```bash
# Generate metrics only
python scripts/generate_example_metrics.py

# Build docs only
mkdocs build
```

### Testing the System

```bash
./scripts/test_doc_generation.sh
```

This validates:
- Metrics generation works
- JSON is valid
- Jinja templates exist
- All components are functional

## Adding New Examples

To add a new example with auto-generated metrics:

### 1. Create the Example Script

```python
# examples/my_new_example.py
def main():
    # ... run emulation ...
    print(f"Mean error: {error:.4f}")
    print(f"Training time: {time:.2f} s")

if __name__ == "__main__":
    main()
```

### 2. Add Metric Extraction

In `scripts/generate_example_metrics.py`:

```python
def extract_my_example_metrics(output: str) -> Dict[str, Any]:
    """Extract metrics from my_new_example.py output."""
    metrics = {}
    
    if match := re.search(r'Mean error: ([\d.]+)', output):
        metrics['mean_error'] = float(match.group(1))
    
    if match := re.search(r'Training time: ([\d.]+)', output):
        metrics['train_time'] = float(match.group(1))
    
    return metrics

# In main():
result = run_example(examples_dir / "my_new_example.py")
if result['success']:
    all_metrics['my_example'] = extract_my_example_metrics(result['output'])
```

### 3. Update Documentation

In `docs/examples/my_new_example.md`:

```markdown
### Performance

{% if metrics and metrics.my_example %}
**Current measured performance:**

| Metric | Value |
|--------|-------|
| Mean error | {{ metrics.my_example.mean_error | percent }} |
| Training time | {{ metrics.my_example.train_time | decimal }} s |
{% else %}
Typical performance:
- Mean error: ~5%
- Training time: ~2 seconds
{% endif %}
```

### 4. Test

```bash
python scripts/generate_example_metrics.py
./scripts/test_doc_generation.sh
mkdocs serve
```

## Jinja Template Reference

### Available Metrics

Check `docs/example_metrics.json` after running generation to see all available metrics.

Current structure:
```python
metrics = {
    'high_dim_5d': {'mean_l2_error': float, 'min_error': float, ...},
    'high_dim_10d': {'mean_l2_error': float, ...},
    'acoustic': {'mean_l2_error': float, 'centroid_error': float, ...},
    'materials': {'best_method': str, 'best_l2_error': float, 'methods': {...}},
}
```

### Custom Filters

- `{{ value | percent }}`: Format as percentage (0.24 → "24.0%")
- `{{ value | percent(2) }}`: Format with 2 decimals (0.2443 → "24.43%")
- `{{ value | decimal }}`: Format as decimal (3.3709 → "3.37")
- `{{ value | decimal(4) }}`: Format with 4 decimals (0.2443 → "0.2443")

### Template Patterns

**Simple metric display:**
```markdown
{{ metrics.example.value | percent }}
```

**Conditional rendering:**
```markdown
{% if metrics and metrics.example %}
  Current: {{ metrics.example.value }}
{% else %}
  Fallback value
{% endif %}
```

**Looping over methods:**
```markdown
{% for method, error in metrics.materials.methods.items() %}
| {{ method.capitalize() }} | {{ error | percent }} |
{% endfor %}
```

## Troubleshooting

### Metrics not appearing

1. Check that `docs/example_metrics.json` exists and is valid JSON
2. Verify the hook is enabled in `mkdocs.yml`
3. Check console output when running `mkdocs serve`
4. Ensure Jinja syntax is correct (no typos in metric names)

### Examples failing

1. Run `python scripts/generate_example_metrics.py` directly
2. Check for errors in the output
3. Verify all dependencies are installed
4. Ensure example scripts run successfully standalone

### Stale metrics

The build system automatically detects changes:
- Compares file modification times of examples vs. metrics
- Regenerates only when needed
- Shows "Using cached metrics" when files haven't changed

To force regeneration:
```bash
rm docs/example_metrics.json && mkdocs build
```

To skip regeneration entirely (fastest):
```bash
SKIP_METRICS=1 mkdocs serve
```

### Build taking too long

The caching system should make rebuilds fast. If you see:
- ✅ `Using cached metrics` → Fast rebuild (metrics cached)
- ⚠️ `Generating example metrics` → Full regeneration (file changed)
- ⚠️ `Skipping metrics` → Using SKIP_METRICS=1

For documentation-only edits, use `SKIP_METRICS=1` to skip all example execution.

## Benefits

✅ **Always accurate**: Metrics come from actual code execution
✅ **No manual updates**: Changes to examples automatically propagate
✅ **Smart caching**: Only regenerates when files actually change
✅ **Fast rebuilds**: Cached results for unchanged examples
✅ **Reproducible**: Anyone can rebuild docs with same metrics
✅ **Scientific integrity**: Can't accidentally claim false performance
✅ **Easy maintenance**: Add new examples by following simple pattern
✅ **Graceful fallback**: Static values shown if generation fails

## Performance

Building docs takes ~2-5 minutes because it runs all examples:
- `high_dim_emulation.py` (5D): ~30 seconds
- `high_dim_emulation.py` (10D): ~60 seconds
- `acoustic_resonance.py`: ~60 seconds
- `materials_spectroscopy.py`: ~45 seconds

For quick iteration during doc writing, you can:
1. Use cached `example_metrics.json` (don't delete it)
2. Comment out slow examples in `generate_example_metrics.py`
3. Use `mkdocs serve` which caches after first build
