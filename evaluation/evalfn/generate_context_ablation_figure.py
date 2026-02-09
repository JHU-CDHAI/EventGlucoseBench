"""
Generate Context Ablation Figure
==================================

Creates line plot showing how CRPS changes across context levels.

Output:
    - evaluation/results/figures/context_ablation.png
    - evaluation/results/figures/context_ablation.pdf
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent.parent
    results_dir = repo_root / 'evaluation' / 'results'
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load context ablation summary
    summary_file = results_dir / 'context_ablation_summary.csv'
    df = pd.read_csv(summary_file)

    # For each model, we need to reconstruct NoCtx baseline
    # Delta = Context - NoCtx, so NoCtx = Context - Delta
    # We'll use the detailed CSV to get actual values
    detailed_file = results_dir / 'context_ablation_detailed.csv'
    detailed_df = pd.read_csv(detailed_file)

    # Create figure
    plt.figure(figsize=(12, 7))

    # Plot each model
    context_levels = ['NoCtx', 'Profile', 'MediumEvent', 'DetailedEvent']
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))

    for idx, (_, row) in enumerate(df.iterrows()):
        model = row['model']

        # Get values from detailed CSV for this model
        model_data = detailed_df[detailed_df['model'] == model].iloc[0]

        values = []
        for level in context_levels:
            if level in model_data and pd.notna(model_data[level]):
                values.append(model_data[level])
            else:
                values.append(None)

        # Only plot if we have NoCtx (baseline)
        if values[0] is not None:
            # Plot line
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_values = [values[i] for i in valid_indices]
            valid_levels = [context_levels[i] for i in valid_indices]

            linestyle = '-'
            linewidth = 2
            marker = 'o'
            markersize = 8

            # Highlight special models
            if model == 'llmp-qwen2.5-0.5B-Instruct':
                linewidth = 3
                markersize = 10
                label = f'{model} (+48% 🚀)'
            elif model == 'openrouter-mixtral-8x7b-instruct':
                linewidth = 3
                markersize = 10
                label = f'{model} (-44% 📉)'
            else:
                label = model

            plt.plot(valid_levels, valid_values,
                    marker=marker, markersize=markersize,
                    linewidth=linewidth, linestyle=linestyle,
                    color=colors[idx], label=label, alpha=0.8)

    plt.xlabel('Context Level', fontsize=14, fontweight='bold')
    plt.ylabel('CRPS (lower is better)', fontsize=14, fontweight='bold')
    plt.title('Context Ablation: Impact of Context on Model Performance',
             fontsize=16, fontweight='bold', pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save figure
    plt.savefig(figures_dir / 'context_ablation.png', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'context_ablation.pdf', bbox_inches='tight')
    print(f"✅ Saved: {figures_dir / 'context_ablation.png'}")
    print(f"✅ Saved: {figures_dir / 'context_ablation.pdf'}")

    plt.close()

if __name__ == '__main__':
    main()
