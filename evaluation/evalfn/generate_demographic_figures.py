"""
Generate Demographic Analysis Figures
======================================

Creates bar charts showing performance across demographic subgroups.

Output:
    - evaluation/results/figures/demographic_comparison.png
    - evaluation/results/figures/demographic_comparison.pdf
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def parse_demographic_summary(file_path):
    """Parse the demographic summary CSV with section headers."""
    data = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_section = None
    for line in lines:
        line = line.strip()
        if line.startswith('# '):
            current_section = line[2:]
            data[current_section] = []
        elif line and not line.startswith('#') and current_section:
            data[current_section].append(line)

    # Parse each section into DataFrame
    dfs = {}
    for section, rows in data.items():
        if rows:
            from io import StringIO
            dfs[section] = pd.read_csv(StringIO('\n'.join(rows)))

    return dfs

def main():
    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_dir = repo_root / 'evaluation' / 'results'
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load demographic summary
    summary_file = results_dir / 'demographic_summary.csv'
    sections = parse_demographic_summary(summary_file)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Color palette
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    # (a) Diabetes Type
    if 'DIABETES TYPE' in sections:
        df = sections['DIABETES TYPE']
        ax = axes[0]

        x = np.arange(len(df))
        bars = ax.bar(x, df['crps_mean'], yerr=df['crps_std'],
                     color=colors[:2], alpha=0.7, capsize=5, width=0.6)

        ax.set_xlabel('Diabetes Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('CRPS (lower is better)', fontsize=12, fontweight='bold')
        ax.set_title('(a) Diabetes Type', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['diabetes_type'], fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add percentage annotation
        if len(df) == 2:
            pct_diff = ((df.iloc[1]['crps_mean'] - df.iloc[0]['crps_mean']) /
                       df.iloc[0]['crps_mean'] * 100)
            ax.text(0.5, max(df['crps_mean']) * 1.1,
                   f'+{pct_diff:.1f}% harder',
                   ha='center', fontsize=10, color='red', fontweight='bold')

    # (b) Age Groups
    if 'AGE' in sections:
        df = sections['AGE']
        ax = axes[1]

        x = np.arange(len(df))
        bars = ax.bar(x, df['crps_mean'], yerr=df['crps_std'],
                     color=colors[:3], alpha=0.7, capsize=5, width=0.6)

        ax.set_xlabel('Age Group', fontsize=12, fontweight='bold')
        ax.set_ylabel('CRPS (lower is better)', fontsize=12, fontweight='bold')
        ax.set_title('(b) Age Groups', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Age {int(age)}' for age in df['age']], fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add trend annotation
        if len(df) >= 2:
            pct_diff = ((df.iloc[-1]['crps_mean'] - df.iloc[0]['crps_mean']) /
                       df.iloc[0]['crps_mean'] * 100)
            ax.annotate('', xy=(len(df)-1, df.iloc[-1]['crps_mean']),
                       xytext=(0, df.iloc[0]['crps_mean']),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.5))
            ax.text(len(df)/2, max(df['crps_mean']) * 1.1,
                   f'+{pct_diff:.1f}% (18→65)',
                   ha='center', fontsize=10, color='red', fontweight='bold')

    # (c) Event Type
    if 'EVENT TYPE' in sections:
        df = sections['EVENT TYPE']
        ax = axes[2]

        x = np.arange(len(df))
        bars = ax.bar(x, df['crps_mean'], yerr=df['crps_std'],
                     color=colors[2:4], alpha=0.7, capsize=5, width=0.6)

        ax.set_xlabel('Event Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('CRPS (lower is better)', fontsize=12, fontweight='bold')
        ax.set_title('(c) Event Types', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['event_type'], fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Add percentage annotation
        if len(df) == 2:
            pct_diff = ((df.iloc[1]['crps_mean'] - df.iloc[0]['crps_mean']) /
                       df.iloc[0]['crps_mean'] * 100)
            ax.text(0.5, max(df['crps_mean']) * 1.1,
                   f'+{pct_diff:.1f}% harder',
                   ha='center', fontsize=10, color='red', fontweight='bold')

    # Overall title
    fig.suptitle('Demographic Subgroup Analysis: Forecasting Difficulty Gradient',
                fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save figure
    plt.savefig(figures_dir / 'demographic_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'demographic_comparison.pdf', bbox_inches='tight')
    print(f"✅ Saved: {figures_dir / 'demographic_comparison.png'}")
    print(f"✅ Saved: {figures_dir / 'demographic_comparison.pdf'}")

    plt.close()

if __name__ == '__main__':
    main()
