"""
Visualize Metric Comparison for Glucose Forecasting
====================================================

Creates figures showing:
1. Correlation comparison (CRPS vs Glucose-RCRPS vs Weighted term)
2. Scatter plots of metrics vs Clarke A+B
3. Model ranking changes across metrics

Usage:
    python evaluation/eval_fn/visualize_metric_comparison.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Set style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


def create_correlation_barplot(df: pd.DataFrame, output_path: Path):
    """
    Create bar plot showing correlation of different metrics with Clarke A+B.

    Similar to showing how Glucose-RCRPS improves clinical alignment.
    """
    # Compute correlations
    correlations = {}
    for metric in ['crps', 'glucose_rcrps', 'weighted_term', 'mae', 'rmse']:
        if metric in df.columns and 'clarke_ab' in df.columns:
            valid = df[[metric, 'clarke_ab']].dropna()
            corr = valid.corr().iloc[0, 1]
            correlations[metric] = corr

    # Create DataFrame
    corr_df = pd.DataFrame([
        {'Metric': 'Standard\nCRPS', 'Correlation': correlations.get('crps', np.nan), 'order': 3},
        {'Metric': 'Glucose-\nRCRPS', 'Correlation': correlations.get('glucose_rcrps', np.nan), 'order': 2},
        {'Metric': 'Clarke-\nWeighted\nCRPS', 'Correlation': correlations.get('weighted_term', np.nan), 'order': 1},
        {'Metric': 'MAE', 'Correlation': correlations.get('mae', np.nan), 'order': 4},
        {'Metric': 'RMSE', 'Correlation': correlations.get('rmse', np.nan), 'order': 5},
    ]).dropna()

    corr_df = corr_df.sort_values('order')

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#2ecc71' if 'RCRPS' in m or 'Weighted' in m else '#95a5a6'
              for m in corr_df['Metric']]

    bars = ax.barh(corr_df['Metric'], -corr_df['Correlation'], color=colors, alpha=0.8)

    # Add value labels
    for i, (idx, row) in enumerate(corr_df.iterrows()):
        ax.text(-row['Correlation'] + 0.01, i, f"{-row['Correlation']:.3f}",
                va='center', ha='left', fontsize=9)

    ax.set_xlabel('Correlation with Clarke A+B (more negative = better)', fontsize=11)
    ax.set_title('Metric Correlation with Clinical Outcomes', fontsize=13, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)

    # Add improvement annotations
    baseline_corr = abs(correlations.get('crps', 0))
    grcrps_corr = abs(correlations.get('glucose_rcrps', 0))
    weighted_corr = abs(correlations.get('weighted_term', 0))

    grcrps_improvement = (grcrps_corr - baseline_corr) / baseline_corr * 100
    weighted_improvement = (weighted_corr - baseline_corr) / baseline_corr * 100

    ax.text(0.98, 0.95,
            f'Glucose-RCRPS: +{grcrps_improvement:.1f}% vs CRPS\nWeighted CRPS: +{weighted_improvement:.1f}% vs CRPS',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Correlation barplot saved to: {output_path}")
    plt.close()


def create_scatter_comparison(df: pd.DataFrame, output_path: Path):
    """
    Create scatter plots showing CRPS vs Glucose-RCRPS vs Clarke A+B.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = [
        ('crps', 'Standard CRPS'),
        ('glucose_rcrps', 'Glucose-RCRPS'),
        ('weighted_term', 'Clarke-Weighted CRPS')
    ]

    for ax, (metric, title) in zip(axes, metrics):
        valid = df[[metric, 'clarke_ab']].dropna()

        # Scatter plot
        ax.scatter(valid[metric], valid['clarke_ab'], alpha=0.3, s=10, color='steelblue')

        # Regression line
        z = np.polyfit(valid[metric], valid['clarke_ab'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[metric].min(), valid[metric].max(), 100)
        ax.plot(x_line, p(x_line), "r-", alpha=0.8, linewidth=2)

        # Correlation
        corr = valid.corr().iloc[0, 1]

        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('Clarke A+B (%)', fontsize=11)
        ax.set_title(f'r = {corr:.3f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add target zone (Clarke A+B > 90%)
        ax.axhline(90, color='green', linestyle='--', alpha=0.5, label='Clinical target')
        ax.legend(fontsize=8)

    plt.suptitle('Metric Correlation with Clinical Safety (Clarke A+B)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Scatter comparison saved to: {output_path}")
    plt.close()


def create_rank_change_plot(df: pd.DataFrame, output_path: Path):
    """
    Show how model rankings change between CRPS and Glucose-RCRPS.

    Highlights models that improve/worsen with clinical weighting.
    """
    # Aggregate by model
    model_stats = df.groupby('model').agg({
        'crps': 'mean',
        'glucose_rcrps': 'mean',
        'weighted_term': 'mean',
        'clarke_ab': 'mean'
    }).reset_index()

    # Compute ranks
    model_stats['crps_rank'] = model_stats['crps'].rank(method='min')
    model_stats['grcrps_rank'] = model_stats['glucose_rcrps'].rank(method='min')
    model_stats['rank_change'] = model_stats['crps_rank'] - model_stats['grcrps_rank']

    # Sort by rank change (biggest improvers first)
    model_stats = model_stats.sort_values('rank_change', ascending=False).head(15)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(model_stats))

    # Plot rank positions
    for i, (idx, row) in enumerate(model_stats.iterrows()):
        # Line connecting ranks
        ax.plot([row['crps_rank'], row['grcrps_rank']], [i, i],
                color='gray', alpha=0.5, linewidth=1)

        # CRPS rank
        ax.scatter(row['crps_rank'], i, color='steelblue', s=100,
                   label='CRPS' if i == 0 else '', zorder=3)

        # Glucose-RCRPS rank
        ax.scatter(row['grcrps_rank'], i, color='coral', s=100,
                   label='Glucose-RCRPS' if i == 0 else '', zorder=3)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace('_', ' ') for m in model_stats['model']], fontsize=9)
    ax.set_xlabel('Rank (lower is better)', fontsize=11)
    ax.set_title('Model Ranking: CRPS vs Glucose-RCRPS', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.invert_xaxis()  # Lower rank on left
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Rank change plot saved to: {output_path}")
    plt.close()


def create_component_contribution_plot(df: pd.DataFrame, output_path: Path):
    """
    Show contribution of weighted term vs penalty term in Glucose-RCRPS.
    """
    # Aggregate by model
    model_stats = df.groupby('model').agg({
        'weighted_term': 'mean',
        'penalty_term': 'mean',
        'glucose_rcrps': 'mean'
    }).reset_index()

    # Sort by Glucose-RCRPS
    model_stats = model_stats.sort_values('glucose_rcrps').head(15)

    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(model_stats))

    # Stacked bars
    ax.barh(y_pos, model_stats['weighted_term'],
            label='Weighted Term (Clarke zones)', color='#3498db', alpha=0.8)
    ax.barh(y_pos, model_stats['penalty_term'],
            left=model_stats['weighted_term'],
            label='Penalty Term (Constraints)', color='#e74c3c', alpha=0.8)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace('_', ' ') for m in model_stats['model']], fontsize=9)
    ax.set_xlabel('Glucose-RCRPS Score', fontsize=11)
    ax.set_title('Component Contribution to Glucose-RCRPS (Top 15 Models)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    # Add percentage annotations
    for i, (idx, row) in enumerate(model_stats.iterrows()):
        weighted_pct = row['weighted_term'] / row['glucose_rcrps'] * 100
        penalty_pct = row['penalty_term'] / row['glucose_rcrps'] * 100
        ax.text(row['glucose_rcrps'] + 0.01, i,
                f"{weighted_pct:.0f}% / {penalty_pct:.0f}%",
                va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Component contribution plot saved to: {output_path}")
    plt.close()


def main():
    """Generate all visualization figures."""
    print("=" * 80)
    print("CREATING METRIC COMPARISON VISUALIZATIONS")
    print("=" * 80)

    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    csv_path = repo_root / 'evaluation' / 'results' / 'clinical_metrics_detailed.csv'
    output_dir = repo_root / 'evaluation' / 'results' / 'figures'
    output_dir.mkdir(exist_ok=True)

    # Load data
    print(f"\n1. Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    valid = df[df['glucose_rcrps'].notna()].copy()
    print(f"   Loaded {len(valid)} valid rows")

    # Create figures
    print(f"\n2. Creating correlation barplot...")
    create_correlation_barplot(valid, output_dir / 'correlation_barplot.pdf')

    print(f"\n3. Creating scatter comparison...")
    create_scatter_comparison(valid, output_dir / 'scatter_comparison.pdf')

    print(f"\n4. Creating rank change plot...")
    create_rank_change_plot(valid, output_dir / 'rank_change_plot.pdf')

    print(f"\n5. Creating component contribution plot...")
    create_component_contribution_plot(valid, output_dir / 'component_contribution.pdf')

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated figures in: {output_dir}")


if __name__ == '__main__':
    main()
