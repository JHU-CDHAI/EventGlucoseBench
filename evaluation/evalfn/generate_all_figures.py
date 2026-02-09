"""
Generate All Evaluation Figures
================================

Master script to generate all publication-ready figures from evaluation results.

Runs:
    1. generate_context_ablation_figure.py
    2. generate_demographic_figures.py
    3. generate_clinical_figures.py

Output:
    All figures saved to evaluation/results/figures/
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    """Run a Python script and handle errors."""
    script_path = Path(__file__).parent / script_name
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print(f"{'='*80}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    print("="*80)
    print("GENERATING ALL EVALUATION FIGURES")
    print("="*80)

    scripts = [
        'generate_context_ablation_figure.py',
        'generate_demographic_figures.py',
        'generate_clinical_figures.py',
    ]

    success_count = 0
    for script in scripts:
        if run_script(script):
            success_count += 1

    print(f"\n{'='*80}")
    print(f"COMPLETE: {success_count}/{len(scripts)} figure scripts completed successfully")
    print(f"{'='*80}")

    # Print summary
    repo_root = Path(__file__).resolve().parent.parent.parent
    figures_dir = repo_root / 'evaluation' / 'results' / 'figures'

    if figures_dir.exists():
        figures = sorted(figures_dir.glob('*.png'))
        print(f"\n📊 Generated {len(figures)} figures in {figures_dir}:")
        for fig in figures:
            print(f"   - {fig.name}")
    else:
        print("\n⚠️  Figures directory not found!")

if __name__ == '__main__':
    main()
