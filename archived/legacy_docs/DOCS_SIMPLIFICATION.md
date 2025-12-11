# Documentation Simplification Proposal

## Current Architecture (Complex)

```
examples/*.py  →  scripts/generate_example_metrics.py  →  docs/example_metrics.json
                            ↓                                        ↓
                   (regex parsing)                          (Jinja templating)
                            ↓                                        ↓
                  docs/examples/figs/                    docs/examples/*.md
                            ↓                                        ↓
                      └──────────────────→ mkdocs build ←──────────────┘
```

**Pain points:**
1. **Fragile regex parsing** - Metrics extraction depends on exact output format
2. **Duplication** - Logic exists in both Python scripts and Markdown files
3. **Stale data** - Easy to have metrics/docs out of sync
4. **Complex build process** - Multiple scripts, hooks, caching logic
5. **Maintenance burden** - Adding new examples requires updating 3-4 files

## Proposed Architecture (Simple)

```
docs/notebooks/*.ipynb  →  mkdocs-jupyter (execute=true)  →  Final HTML
```

**Benefits:**
1. **Self-contained** - Code, output, and narrative in one file
2. **Always current** - Notebooks executed at build time
3. **Easy to add** - Just create a notebook, add to nav
4. **Interactive** - Users can download and run themselves
5. **No custom scripts** - Built-in mkdocs-jupyter handles everything

## Migration Plan

### Phase 1: Enable Notebook Execution (Now)

```yaml
# mkdocs.yml
plugins:
  - mkdocs-jupyter:
      execute: true           # Execute notebooks at build time
      include_source: true    # Allow users to download
      execute_ignore:         # Skip heavy notebooks in CI
        - "notebooks/paper_repro_full.ipynb"
```

### Phase 2: Convert Examples to Notebooks

| Current | New |
|---------|-----|
| `examples/acoustic_resonance.py` + `docs/examples/acoustic_resonance.md` | `docs/notebooks/acoustic_resonance.ipynb` |
| `examples/materials_spectroscopy.py` + `docs/examples/materials_spectroscopy.md` | `docs/notebooks/materials_spectroscopy.ipynb` |
| `examples/high_dim_emulation.py` + `docs/examples/nuclear_response.md` | `docs/notebooks/high_dim_emulation.ipynb` |

### Phase 3: Simplify Build

Remove:
- `scripts/generate_example_metrics.py`
- `scripts/generate_example_figs.py`  
- `docs/hooks.py` (or simplify to just load config)
- `docs/example_metrics.json`
- Jinja templating in markdown files

Keep:
- `examples/*.py` - For CLI users who prefer scripts
- `docs/notebooks/*.ipynb` - For documentation (source of truth)

### Phase 4: CI Configuration

```yaml
# .github/workflows/docs.yml
- name: Build docs
  run: |
    uv sync --group docs
    SKIP_METRICS=1 mkdocs build  # Notebooks handle their own execution
```

For expensive notebooks:
```yaml
# docs/notebooks/paper_repro_full.ipynb - metadata
{
  "kernelspec": {...},
  "metadata": {
    "mkdocs-jupyter": {
      "execute": false  # Pre-executed, committed with outputs
    }
  }
}
```

## Hybrid Approach (Recommended)

Keep both for now, with notebooks as the **primary documentation source**:

1. **Tutorials** → Executed notebooks (always fresh)
2. **Examples gallery** → Executed notebooks with outputs
3. **Paper reproduction** → Pre-executed notebook (committed with outputs)
4. **API reference** → Generated from docstrings (mkdocstrings)

This gives you:
- ✅ Fresh, accurate documentation
- ✅ Downloadable examples users can run
- ✅ Simple build process
- ✅ CLI scripts for power users
- ✅ Fallback if notebook execution fails

## Quick Start

To enable notebook execution now:

```bash
# 1. Update mkdocs.yml
sed -i '' 's/execute: false/execute: true/' mkdocs.yml

# 2. Add notebooks to nav
# Already have docs/notebooks/acoustic_resonance.ipynb

# 3. Test locally
uv run mkdocs serve

# 4. For CI, set timeout
# mkdocs-jupyter respects notebook kernel timeout
```

## Comparison

| Aspect | Current (Script+Regex) | Proposed (Notebooks) |
|--------|------------------------|----------------------|
| Lines of custom code | ~500 | ~50 |
| Files to maintain | 8+ | 3-4 |
| Build time | Fast (cached) | Slower (execution) |
| Freshness guarantee | Manual | Automatic |
| User downloadable | No | Yes |
| Debugging | Hard (regex) | Easy (run notebook) |

## Recommendation

**Start small**: 
1. Enable `execute: true` for one notebook
2. Verify it works in CI
3. Gradually migrate examples
4. Keep legacy system until migration complete
5. Remove legacy system after validation

This approach is lower risk and allows incremental improvement.
