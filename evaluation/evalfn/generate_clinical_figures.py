"""
Generate Clinical Metrics Figures
==================================

Creates scatter plot (CRPS vs Clarke A+B) and box plots (model families).

Output:
    - evaluation/results/figures/crps_vs_clarke_scatter.png
    - evaluation/results/figures/model_family_comparison.png
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def classify_model_family(model_name):
    """Classify model into families for box plot."""
    if 'chronos' in model_name.lower():
        return 'Chronos'
    elif 'gpt-4o' in model_name.lower():
        return 'GPT-4o'
    elif 'llmp' in model_name.lower():
        return 'LLMP'
    elif 'openrouter' in model_name.lower():
        return 'OpenRouter'
    elif 'moirai' in model_name.lower():
        return 'Moirai'
    elif 'unitime' in model_name.lower() or 'timellm' in model_name.lower():
        return 'Time Foundation'
    elif model_name in ['r-arima', 'r-ets', 'exp-smoothing']:
        return 'Statistical'
    else:
        return 'Other'

def main():
    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_dir = repo_root / 'evaluation' / 'results'
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load clinical metrics summary
    summary_file = results_dir / 'clinical_metrics_summary.csv'
    df = pd.read_csv(summary_file)

    # Exclude oracle and random (outliers)
    df = df[~df['model'].isin(['oracle', 'random'])]

    # Add model family column
    df['family'] = df['model'].apply(classify_model_family)

    # ========================================================================
    # Figure 1: CRPS vs Clarke A+B Scatter Plot
    # ========================================================================

    plt.figure(figsize=(12, 8))

    # Color by family
    families = df['family'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(families)))
    family_colors = dict(zip(families, colors))

    for family in families:
        family_df = df[df['family'] == family]
        if 'clarke_ab_mean' in family_df.columns:
            plt.scatter(family_df['crps_mean'], family_df['clarke_ab_mean'],
                       label=family, s=100, alpha=0.7, color=family_colors[family])

    # Add correlation line
    from scipy.stats import pearsonr
    valid_data = df[['crps_mean', 'clarke_ab_mean']].dropna()
    if len(valid_data) > 2:
        r, p = pearsonr(valid_data['crps_mean'], valid_data['clarke_ab_mean'])

        # Fit line
        z = np.polyfit(valid_data['crps_mean'], valid_data['clarke_ab_mean'], 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(valid_data['crps_mean'].min(), valid_data['crps_mean'].max(), 100)
        plt.plot(x_line, p_line(x_line), "r--", alpha=0.5, linewidth=2,
                label=f'Linear fit (r={r:.3f})')

    # Annotate interesting points
    top_crps = df.nsmallest(1, 'crps_mean')
    top_clarke = df.nlargest(1, 'clarke_ab_mean')  # Higher Clarke A+B is better

    for _, row in top_crps.iterrows():
        plt.annotate(row['model'], xy=(row['crps_mean'], row['clarke_ab_mean']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    for _, row in top_clarke.iterrows():
        if row['model'] not in top_crps['model'].values:  # Don't annotate twice
            plt.annotate(row['model'], xy=(row['crps_mean'], row['clarke_ab_mean']),
                        xytext=(10, -20), textcoords='offset points',
                        fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.xlabel('CRPS (lower is better)', fontsize=14, fontweight='bold')
    plt.ylabel('Clarke A+B % (higher is better)', fontsize=14, fontweight='bold')
    plt.title('CRPS vs Clarke A+B: Moderate Negative Correlation',
             fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save
    plt.savefig(figures_dir / 'crps_vs_clarke_scatter.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'crps_vs_clarke_scatter.pdf', bbox_inches='tight')
    print(f"✅ Saved: {figures_dir / 'crps_vs_clarke_scatter.png'}")
    print(f"✅ Saved: {figures_dir / 'crps_vs_clarke_scatter.pdf'}")

    plt.close()

    # ========================================================================
    # Figure 2: Model Family Comparison (Box Plot)
    # ========================================================================

    # Filter to main families with enough data
    main_families = ['Chronos', 'GPT-4o', 'LLMP', 'Statistical']
    plot_df = df[df['family'].isin(main_families)]

    plt.figure(figsize=(14, 7))

    # Prepare data for box plot
    data_by_family = [plot_df[plot_df['family'] == family]['crps_mean'].values
                     for family in main_families]

    # Box plot
    bp = plt.boxplot(data_by_family, labels=main_families, patch_artist=True,
                    widths=0.6, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=8))

    # Color the boxes
    colors_palette = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    for patch, color in zip(bp['boxes'], colors_palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points (strip plot equivalent)
    for i, family in enumerate(main_families):
        y = plot_df[plot_df['family'] == family]['crps_mean'].values
        x = np.random.normal(i+1, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.5, s=30, color='black')

    plt.xlabel('Model Family', fontsize=14, fontweight='bold')
    plt.ylabel('CRPS (lower is better)', fontsize=14, fontweight='bold')
    plt.title('Model Family Performance Comparison',
             fontsize=16, fontweight='bold', pad=20)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add median annotations
    for i, family in enumerate(main_families):
        family_data = plot_df[plot_df['family'] == family]['crps_mean']
        if len(family_data) > 0:
            median = family_data.median()
            plt.text(i, median, f'  {median:.3f}',
                    ha='left', va='center', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    plt.tight_layout()

    # Save
    plt.savefig(figures_dir / 'model_family_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'model_family_comparison.pdf', bbox_inches='tight')
    print(f"✅ Saved: {figures_dir / 'model_family_comparison.png'}")
    print(f"✅ Saved: {figures_dir / 'model_family_comparison.pdf'}")

    plt.close()

    # ========================================================================
    # Figure 3: Multi-Metric Comparison (Top 10 Models)
    # ========================================================================

    top10 = df.nsmallest(10, 'crps_mean')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # CRPS
    ax = axes[0, 0]
    bars = ax.barh(range(len(top10)), top10['crps_mean'], color='steelblue', alpha=0.7)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10['model'], fontsize=10)
    ax.set_xlabel('CRPS', fontsize=12, fontweight='bold')
    ax.set_title('(a) CRPS Rankings', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Clarke A+B
    ax = axes[0, 1]
    bars = ax.barh(range(len(top10)), top10['clarke_ab_mean'], color='coral', alpha=0.7)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10['model'], fontsize=10)
    ax.set_xlabel('Clarke A+B %', fontsize=12, fontweight='bold')
    ax.set_title('(b) Clarke A+B (Clinical Accuracy)', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # MAE
    ax = axes[1, 0]
    bars = ax.barh(range(len(top10)), top10['mae_mean'], color='seagreen', alpha=0.7)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10['model'], fontsize=10)
    ax.set_xlabel('MAE (mg/dL)', fontsize=12, fontweight='bold')
    ax.set_title('(c) Mean Absolute Error', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # RMSE
    ax = axes[1, 1]
    bars = ax.barh(range(len(top10)), top10['rmse_mean'], color='mediumpurple', alpha=0.7)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10['model'], fontsize=10)
    ax.set_xlabel('RMSE (mg/dL)', fontsize=12, fontweight='bold')
    ax.set_title('(d) Root Mean Squared Error', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    fig.suptitle('Multi-Metric Comparison: Top 10 Models by CRPS',
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    plt.savefig(figures_dir / 'multimetric_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'multimetric_comparison.pdf', bbox_inches='tight')
    print(f"✅ Saved: {figures_dir / 'multimetric_comparison.png'}")
    print(f"✅ Saved: {figures_dir / 'multimetric_comparison.pdf'}")

    plt.close()

if __name__ == '__main__':
    main()
