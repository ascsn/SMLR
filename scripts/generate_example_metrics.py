#!/usr/bin/env python3
"""Generate example metrics by running examples and capturing output.

This script runs all example scripts, captures their performance metrics,
and generates a JSON file that can be used to populate documentation templates.
"""
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


def run_example(script_path: Path, args: list = None) -> Dict[str, Any]:
    """Run an example script and capture metrics from output.
    
    Parameters
    ----------
    script_path : Path
        Path to the example script
    args : list, optional
        Command line arguments to pass to the script
        
    Returns
    -------
    dict
        Extracted metrics and metadata
    """
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )
    
    output = result.stdout + result.stderr
    
    metrics = {
        "script": script_path.name,
        "success": result.returncode == 0,
        "output": output,
    }
    
    return metrics


def extract_high_dim_metrics(output: str) -> Dict[str, Any]:
    """Extract metrics from high_dim_emulation.py output."""
    metrics = {}
    
    # Extract error metrics
    if match := re.search(r'Test set mean normalized L2 error: ([\d.]+)', output):
        metrics['mean_l2_error'] = float(match.group(1))
    
    if match := re.search(r'Min error: ([\d.]+)', output):
        metrics['min_error'] = float(match.group(1))
        
    if match := re.search(r'Max error: ([\d.]+)', output):
        metrics['max_error'] = float(match.group(1))
    
    # Extract parameters
    if match := re.search(r'(\d+)D parameter space', output):
        metrics['n_params'] = int(match.group(1))
        
    if match := re.search(r'Training samples: (\d+)', output):
        metrics['n_train'] = int(match.group(1))
        
    if match := re.search(r'Test samples: (\d+)', output):
        metrics['n_test'] = int(match.group(1))
    
    return metrics


def extract_acoustic_metrics(output: str) -> Dict[str, Any]:
    """Extract metrics from acoustic_resonance.py output."""
    metrics = {}
    
    if match := re.search(r'Mean normalized L2 error: ([\d.]+)', output):
        metrics['mean_l2_error'] = float(match.group(1))
    
    if match := re.search(r'Centroid relative error: ([\d.]+)', output):
        metrics['centroid_error'] = float(match.group(1))
        
    if match := re.search(r'Total power relative error: ([\d.]+)', output):
        metrics['power_error'] = float(match.group(1))
    
    if match := re.search(r'Training on (\d+) samples', output):
        metrics['n_train'] = int(match.group(1))
    
    return metrics


def extract_materials_metrics(output: str) -> Dict[str, Any]:
    """Extract metrics from materials_spectroscopy.py output."""
    metrics = {}
    
    # Extract best method and error
    if match := re.search(r'Best method: (\w+) \(mean L2 = ([\d.]+)\)', output):
        metrics['best_method'] = match.group(1)
        metrics['best_l2_error'] = float(match.group(2))
    
    # Extract individual method results
    methods = {}
    for match in re.finditer(r'Fitting (\w+) emulator.*?Mean L2 error: ([\d.]+)', output, re.DOTALL):
        method = match.group(1)
        error = float(match.group(2))
        methods[method] = error
    
    if methods:
        metrics['methods'] = methods
    
    return metrics


def main():
    """Run all examples and generate metrics JSON."""
    repo_root = Path(__file__).parent.parent
    examples_dir = repo_root / "examples"
    output_file = repo_root / "docs" / "example_metrics.json"
    figs_dir = repo_root / "docs" / "examples" / "figs"
    
    # Ensure figures directory exists
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = {}
    
    # High-dimensional emulation (5D)
    print("\n" + "="*60)
    print("Running high_dim_emulation.py (5D)")
    print("="*60)
    result = run_example(
        examples_dir / "high_dim_emulation.py",
        ["--n-params", "5", "--n-train", "80", "--n-test", "20", "--out", str(figs_dir)]
    )
    if result['success']:
        all_metrics['high_dim_5d'] = extract_high_dim_metrics(result['output'])
        print(f"✓ Captured: {all_metrics['high_dim_5d']}")
    else:
        print(f"✗ Failed: {result['output'][-500:]}")
    
    # High-dimensional emulation (10D)
    print("\n" + "="*60)
    print("Running high_dim_emulation.py (10D)")
    print("="*60)
    result = run_example(
        examples_dir / "high_dim_emulation.py",
        ["--n-params", "10", "--n-train", "150", "--n-test", "20", "--out", str(figs_dir)]
    )
    if result['success']:
        all_metrics['high_dim_10d'] = extract_high_dim_metrics(result['output'])
        print(f"✓ Captured: {all_metrics['high_dim_10d']}")
    else:
        print(f"✗ Failed: {result['output'][-500:]}")
    
    # Acoustic resonance
    print("\n" + "="*60)
    print("Running acoustic_resonance.py")
    print("="*60)
    result = run_example(
        examples_dir / "acoustic_resonance.py",
        ["--out", str(figs_dir)]
    )
    if result['success']:
        all_metrics['acoustic'] = extract_acoustic_metrics(result['output'])
        print(f"✓ Captured: {all_metrics['acoustic']}")
    else:
        print(f"✗ Failed: {result['output'][-500:]}")
    
    # Materials spectroscopy
    print("\n" + "="*60)
    print("Running materials_spectroscopy.py")
    print("="*60)
    result = run_example(
        examples_dir / "materials_spectroscopy.py",
        ["--out", str(figs_dir)]
    )
    if result['success']:
        all_metrics['materials'] = extract_materials_metrics(result['output'])
        print(f"✓ Captured: {all_metrics['materials']}")
    else:
        print(f"✗ Failed: {result['output'][-500:]}")
    
    # Save metrics
    print("\n" + "="*60)
    print(f"Saving metrics to {output_file}")
    print("="*60)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"✓ Metrics saved successfully")
    print(f"\nSummary:")
    print(json.dumps(all_metrics, indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
