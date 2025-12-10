"""MkDocs hook to generate example metrics before building docs.

This hook runs before the docs are built to ensure metrics are up-to-date.

Environment variables:
    SKIP_METRICS: Set to '1' or 'true' to skip metric generation entirely
"""
import json
import os
import subprocess
import sys
from pathlib import Path


def _should_regenerate_metrics(repo_root: Path) -> bool:
    """Check if example files have been modified since last metrics generation."""
    metrics_file = repo_root / "docs" / "example_metrics.json"
    
    # Always regenerate if metrics don't exist
    if not metrics_file.exists():
        return True
    
    metrics_mtime = metrics_file.stat().st_mtime
    
    # Check if any example files are newer than metrics
    examples_dir = repo_root / "examples"
    for example_file in examples_dir.glob("*.py"):
        if example_file.stat().st_mtime > metrics_mtime:
            print(f"✓ Detected change in {example_file.name}")
            return True
    
    # Check if generation script itself changed
    script = repo_root / "scripts" / "generate_example_metrics.py"
    if script.exists() and script.stat().st_mtime > metrics_mtime:
        print(f"✓ Detected change in generation script")
        return True
    
    return False


def on_pre_build(config):
    """Run example metrics generation before building docs."""
    repo_root = Path(__file__).parent.parent
    
    # Check if metrics generation should be skipped entirely
    skip_metrics = os.getenv('SKIP_METRICS', '').lower() in ('1', 'true', 'yes')
    if skip_metrics:
        print("\n" + "="*60)
        print("Skipping metrics generation (SKIP_METRICS=1)")
        print("="*60)
        
        # Try to load existing metrics if available
        metrics_file = repo_root / "docs" / "example_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            config['extra']['metrics'] = metrics
            print("✓ Using existing metrics file")
        else:
            print("⚠ No metrics file found - templates will use fallback values")
        return
    
    # Skip regeneration if no files changed
    if not _should_regenerate_metrics(repo_root):
        print("\n" + "="*60)
        print("Using cached metrics (no example files modified)")
        print("="*60)
        
        # Still load metrics into config
        metrics_file = repo_root / "docs" / "example_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            config['extra']['metrics'] = metrics
            print("✓ Cached metrics loaded")
        return
    
    print("\n" + "="*60)
    print("Generating example metrics...")
    print("="*60)
    
    script = repo_root / "scripts" / "generate_example_metrics.py"
    
    # Run the metrics generation script
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Warning: Failed to generate metrics: {result.stderr}")
        return
    
    print(result.stdout)
    print("✓ Example metrics generated successfully")
    
    # Load and inject metrics into config for Jinja templates
    metrics_file = repo_root / "docs" / "example_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        # Make metrics available to templates
        config['extra']['metrics'] = metrics
        print(f"✓ Metrics loaded and available to templates")


def on_env(env, config, files):
    """Add custom Jinja filters for formatting metrics."""
    
    def format_percent(value, decimals=1):
        """Format a decimal as percentage."""
        return f"{float(value) * 100:.{decimals}f}%"
    
    def format_decimal(value, decimals=2):
        """Format a decimal number."""
        return f"{float(value):.{decimals}f}"
    
    env.filters['percent'] = format_percent
    env.filters['decimal'] = format_decimal
    
    return env
