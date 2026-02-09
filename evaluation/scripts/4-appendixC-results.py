"""

This script generates all figures for Appendix C of the paper:
      (NoCtx, Profile, MediumEvent, DetailedEvent, NewMedium, NewDetail)

        {AppendixFigurePath}/ (paper display folder - main metric only)
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from pathlib import Path
import warnings
import os
import sys
import json
import shutil
import yaml

try:
    from adjustText import adjust_text
    HAS_ADJUST_TEXT = True
except ImportError:
    HAS_ADJUST_TEXT = False
    print("Warning: adjustText not installed. Labels may overlap. Install with: pip install adjustText")

TOPIC_NAME = '4-appendixC-results'

warnings.filterwarnings('ignore')
matplotlib.use('Agg')
sns.set_style("whitegrid")

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

def load_config(config_path: Path) -> dict:

    Keys returned:
        pairplot_height
    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  Saved: {filename}.png/.pdf")

    if copy_to_paper and appendix_figure_dir.exists():
        paper_pdf = appendix_figure_dir / f"{filename}.pdf"
        print(f"  Copied to paper: {appendix_figure_dir.name}/{filename}.pdf")

    plt.close(fig)

def save_csv(df, filename):
    if task_name.startswith('EventCGMTask_'):
        return task_name.replace('EventCGMTask_', '')

def get_metric_label(metric, for_title=False):
    original_family = get_model_class(model_name)
    return SIMPLIFIED_FAMILY_MAP.get(original_family, 'Other')

# =============================================================================
# FIGURE GENERATION FUNCTIONS (PER METRIC)
# =============================================================================

def generate_C1_model_task_heatmap(df, metric, is_main_metric):
    print(f"\n  Generating C3 for {metric}...")
    s = get_fig_style('C3')

    metric_display = get_metric_label(metric, for_title=True)

    model_stats = df.groupby('model')[metric].agg(['mean', 'median', 'std'])
    model_order = model_stats.sort_values('median').index.tolist()

    palette = [COLORS_CONTEXT[get_context_status(m)] for m in model_order]

    fig, ax = plt.subplots(figsize=s['figure_size'])
    sns.violinplot(
        data=df,
        x='model',
        y=metric,
        order=model_order,
        palette=palette,
        alpha=0.5,
        ax=ax,
        cut=0
    )

    display_labels = [get_model_display_name(m) for m in model_order]
    ax.set_ylim(0, df[metric].quantile(0.99))
    ax.set_xticklabels(display_labels, rotation=45, ha='right', fontsize=s['x_tick_label'])
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=s['y_tick_label'])
    ax.set_xlabel('Model', fontsize=s['axis_label'])
    ax.set_ylabel(metric_display, fontsize=s['axis_label'])
    ax.set_title(f'{metric_display} Distribution by Model (sorted by median)',
                 fontsize=s['title'], fontweight='bold')

    legend_elements = [Patch(facecolor=c, label=l) for l, c in COLORS_CONTEXT.items()]
    ax.legend(handles=legend_elements, loc='upper right', title='Context Status', fontsize=s['legend'], title_fontsize=s['legend_title'])

    plt.tight_layout()
    save_figure(fig, f"C3-violin_model_{metric}", copy_to_paper=is_main_metric)

def generate_C4_distribution_by_family(df, metric, is_main_metric):
    print(f"\n  Generating C5 for {metric}...")
    s = get_fig_style('C5')

    metric_display = get_metric_label(metric, for_title=True)

    model_means = df.groupby('model')[metric].mean().sort_values()

    colors = [SIMPLIFIED_FAMILY_COLORS.get(get_simple_family(m), '#7f7f7f') for m in model_means.index]

    fig, ax = plt.subplots(figsize=s['figure_size'])
    bars = ax.barh(range(len(model_means)), model_means.values, color=colors, alpha=0.85)

    ax.set_yticks(range(len(model_means)))
    display_labels = [get_model_display_name(m) for m in model_means.index]
    ax.set_yticklabels(display_labels, fontsize=s['y_tick_label'])
    ax.set_xlabel(f'Mean {metric_display} (lower is better)', fontsize=s['axis_label'])
    ax.set_ylabel('Model', fontsize=s['axis_label'])
    ax.set_title(f'Model Ranking by Mean {metric_display}\n(Colored by Model Family)', fontsize=s['title'], fontweight='bold')

    for i, (bar, val) in enumerate(zip(bars, model_means.values)):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=s['annotation'])

    legend_elements = [Patch(facecolor=c, label=f) for f, c in SIMPLIFIED_FAMILY_COLORS.items()
                       if f in df['simple_family'].unique()]
    ax.legend(handles=legend_elements, loc='lower right', title='Model Family', fontsize=s['legend'])

    plt.tight_layout()
    save_figure(fig, f"C5-model_ranking_bar_{metric}", copy_to_paper=is_main_metric)
    save_csv(pd.DataFrame({'model': model_means.index, f'mean_{metric}': model_means.values,
                           'rank': range(1, len(model_means)+1)}), f"C5-model_ranking_{metric}")

def generate_C6_rank_consistency_heatmap(df, metric, is_main_metric):
    print(f"\n  Generating C9 for {metric}...")
    s = get_fig_style('C9')

    metric_display = get_metric_label(metric, for_title=True)

    subgroup_event_values = df.groupby(['subgroup', 'event_type'])[metric].mean().unstack()

    fig, ax = plt.subplots(figsize=s['figure_size'])

    x = np.arange(len(subgroup_event_values))
    width = 0.35

    event_types = [et for et in subgroup_event_values.columns if et in COLORS_EVENT_TYPE]

    for i, event in enumerate(event_types):
        if event in subgroup_event_values.columns:
            offset = (i - len(event_types)/2 + 0.5) * width
            ax.bar(x + offset, subgroup_event_values[event], width,
                   label=event, color=COLORS_EVENT_TYPE.get(event, '#7f7f7f'), alpha=0.8)

    ax.set_xlabel('Demographic Subgroup', fontsize=s['axis_label'])
    ax.set_ylabel(f'Mean {metric_display}', fontsize=s['axis_label'])
    ax.set_title(f'Task Difficulty by Demographic Subgroup and Event Type ({metric_display})\n(Higher = more challenging)',
                 fontsize=s['title'], fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(subgroup_event_values.index, rotation=30, ha='right', fontsize=s['x_tick_label'])
    ax.tick_params(axis='y', labelsize=s['y_tick_label'])
    ax.legend(title='Event Type')

    plt.tight_layout()
    save_figure(fig, f"C9-task_difficulty_bar_{metric}", copy_to_paper=is_main_metric)

    s9b = get_fig_style('C9b')
    subgroup_values = df.groupby('subgroup')[metric].agg(['mean', 'std', 'count'])
    subgroup_values['sem'] = subgroup_values['std'] / np.sqrt(subgroup_values['count'])
    subgroup_values = subgroup_values.sort_values('mean')

    fig, ax = plt.subplots(figsize=s9b['figure_size'])
    colors = [COLORS_SUBGROUP.get(sg, '#7f7f7f') for sg in subgroup_values.index]
    ax.bar(range(len(subgroup_values)), subgroup_values['mean'], yerr=subgroup_values['sem'],
           color=colors, alpha=0.8, capsize=5)
    ax.set_xticks(range(len(subgroup_values)))
    ax.set_xticklabels(subgroup_values.index, rotation=30, ha='right', fontsize=s9b['x_tick_label'])
    ax.tick_params(axis='y', labelsize=s9b['y_tick_label'])
    ax.set_xlabel('Demographic Subgroup', fontsize=s9b['axis_label'])
    ax.set_ylabel(f'Mean {metric_display} (+/- SEM)', fontsize=s9b['axis_label'])
    ax.set_title(f'Overall Task Difficulty by Demographic Subgroup ({metric_display})', fontsize=s9b['title'], fontweight='bold')

    plt.tight_layout()
    save_figure(fig, f"C9b-subgroup_difficulty_overall_{metric}", copy_to_paper=is_main_metric)
    save_csv(subgroup_values, f"C9-subgroup_difficulty_stats_{metric}")

def generate_C10_boxplot_by_family(df, metric, is_main_metric):
    print(f"\n  Generating C11 for {metric}...")
    s = get_fig_style('C11')

    metric_display = get_metric_label(metric, for_title=True)

    context_order = ['NoCtx', 'Profile', 'MediumEvent', 'DetailedEvent', 'NewMedium', 'NewDetail']
    context_order = [c for c in context_order if c in df['context_level'].unique()]

    if len(context_order) == 0:
        print(f"  Skipping C11: no valid context levels found")
        return

    palette = [COLORS_TASK_CONTEXT.get(c, '#7f7f7f') for c in context_order]

    fig, ax = plt.subplots(figsize=s['figure_size'])
    sns.boxplot(
        data=df,
        x='context_level',
        y=metric,
        order=context_order,
        palette=palette,
        ax=ax,
        showfliers=False,
        boxprops=dict(alpha=0.7, linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2),
        medianprops=dict(linewidth=2)
    )

    ax.set_xlabel('Context Level', fontsize=s['axis_label'])
    ax.set_ylabel(metric_display, fontsize=s['axis_label'])
    ax.set_title(f'{metric_display} Distribution by Context Level\n(Ordered by information richness)',
                 fontsize=s['title'], fontweight='bold')
    ax.set_xticklabels(context_order, rotation=30, ha='right', fontsize=s['x_tick_label'])
    ax.tick_params(axis='y', labelsize=s['y_tick_label'])
    ax.set_ylim(0, df[metric].quantile(0.95))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, f"C11-performance_by_context_level_{metric}", copy_to_paper=is_main_metric)

    context_stats = df.groupby('context_level')[metric].agg(['mean', 'median', 'std', 'count'])
    context_stats['sem'] = context_stats['std'] / np.sqrt(context_stats['count'])
    context_stats = context_stats.reindex([c for c in context_order if c in context_stats.index])
    save_csv(context_stats, f"C11-context_level_statistics_{metric}")

def generate_C12_context_level_by_model_family(df, metric, is_main_metric):
    print("\n" + "=" * 80)
    print("C2: METRIC CORRELATION HEATMAP")
    print("=" * 80)
    s = get_fig_style('C2')

    metric_cols = ['crps', 'mae', 'rmse', 'clarke_a', 'clarke_ab', 'clarke_abc']
    if 't_crps' in df.columns:
        metric_cols.append('t_crps')
    if 'c_crps' in df.columns:
        metric_cols.append('c_crps')
    if 'ct_crps' in df.columns:
        metric_cols.append('ct_crps')
    if 'clarke_cde' in df.columns:
        metric_cols.append('clarke_cde')

    metric_cols = [c for c in metric_cols if c in df.columns]
    metrics_df = df[metric_cols].dropna()

    corr_matrix = metrics_df.corr()
    corr_matrix = corr_matrix.rename(index=METRIC_DISPLAY_NAMES, columns=METRIC_DISPLAY_NAMES)

    fig, ax = plt.subplots(figsize=s['figure_size'])
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )
    ax.set_title('Metric Correlation Heatmap', fontsize=s['title'], fontweight='bold')

    plt.tight_layout()
    save_figure(fig, "C2-metric_correlation_heatmap", copy_to_paper=True)
    save_csv(corr_matrix, "C2-metric_correlation_matrix")

def generate_C7_pairplot(df):
    print("\n" + "=" * 80)
    print("C8: METRIC VS STANDARD CRPS SCATTER PLOTS")
    print("=" * 80)
    s = get_fig_style('C8')

    if 'crps' not in df.columns:
        print("  Skipping C8: crps column not found")
        return

    metrics_to_plot = [m for m in available_metrics if m != 'crps']

    for metric in metrics_to_plot:
        print(f"\n  Generating C8 for {metric} vs CRPS...")

        metric_display = get_metric_label(metric, for_title=True)

        model_metrics = df.groupby('model').agg({
            'crps': 'mean',
            metric: 'mean'
        }).dropna()

        fig, ax = plt.subplots(figsize=s['figure_size'])

        texts = []
        for model in model_metrics.index:
            family = get_model_class(model)
            color = COLORS_MODEL_CLASS.get(family, '#7f7f7f')
            ax.scatter(
                s=150, alpha=0.7, color=color
            )
            txt = ax.text(
                get_model_display_name(model),
                fontsize=s['annotation']
            )
            texts.append(txt)

        x_max = model_metrics['crps'].max()
        y_max = model_metrics[metric].max()
        ax.set_xlim(20, 1.2 * x_max)
        ax.set_ylim(20, 1.2 * y_max)

        lims = [
            max(ax.get_xlim()[0], ax.get_ylim()[0]),
            min(ax.get_xlim()[1], ax.get_ylim()[1])
        ]
        if lims[0] < lims[1]:
            ax.plot(lims, lims, 'k--', alpha=0.5, label='y=x')

        ax.set_xlabel('Standard CRPS', fontsize=s['axis_label'])
        ax.set_ylabel(metric_display, fontsize=s['axis_label'])
        ax.tick_params(axis='both', labelsize=s['tick_label'])
        ax.set_title(f'{metric_display} vs Standard CRPS by Model',
                     fontsize=s['title'], fontweight='bold')

        legend_elements = [Patch(facecolor=c, label=f) for f, c in COLORS_MODEL_CLASS.items()
                           if f in df['model_family'].unique()]
        ax.legend(handles=legend_elements, loc='upper left', title='Model Family', fontsize=s['legend'])

        if HAS_ADJUST_TEXT and texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

        plt.tight_layout()
        is_main = (metric == MAIN_METRIC)
        save_figure(fig, f"C8-{metric}_vs_crps", copy_to_paper=is_main)
        save_csv(model_metrics[['crps', metric]], f"C8-{metric}_vs_crps_data")

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_csv(data_path)
print(f"Loaded: {len(df):,} rows")

with open(models_json_path, 'r') as f:
    models_tasks_config = json.load(f)
df = df[df['model'].isin(models_tasks_config['models']) & df['task'].isin(models_tasks_config['tasks'])]

df = filter_excluded_models(df, 'model')
df = filter_excluded_tasks(df, 'task')

print(f"After filtering: {len(df):,} rows")
print(f"Models: {df['model'].nunique()}, Tasks: {df['task'].nunique()}")

df['model_family'] = df['model'].apply(get_model_class)
df['context_status'] = df['model'].apply(get_context_status)
df['subgroup'] = df['task'].apply(parse_instance_subgroup)
df['event_type'] = df['task'].apply(parse_instance_event)
df['context_level'] = df['task'].apply(parse_task_context)
df['simple_family'] = df['model_family'].map(SIMPLIFIED_FAMILY_MAP).fillna('Other')

available_metrics = [m for m in METRICS_LIST if m in df.columns]
print(f"\nAvailable metrics: {available_metrics}")

# =============================================================================
# GENERATE FIGURES
# =============================================================================

generate_C2_metric_correlation(df)
generate_C7_pairplot(df)
generate_C8_metric_vs_crps(df, available_metrics)

for metric in available_metrics:
    is_main = (metric == MAIN_METRIC)

    print("\n" + "=" * 80)
    print(f"GENERATING FIGURES FOR METRIC: {metric.upper()}" + (" (MAIN)" if is_main else ""))
    print("=" * 80)

    generate_C1_model_task_heatmap(df, metric, is_main)
    generate_C3_violin_plot(df, metric, is_main)
    generate_C4_distribution_by_family(df, metric, is_main)
    generate_C5_model_ranking_bar(df, metric, is_main)
    generate_C6_rank_consistency_heatmap(df, metric, is_main)
    generate_C9_task_difficulty(df, metric, is_main)
    generate_C10_boxplot_by_family(df, metric, is_main)
    generate_C11_performance_by_context_level(df, metric, is_main)
    generate_C12_context_level_by_model_family(df, metric, is_main)

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("APPENDIX C GENERATION COMPLETE")
print("=" * 80)

print(f"\n📁 OUTPUT LOCATIONS:")
print(f"   Results directory: {output_dir}")
print(f"   Paper figures directory: {appendix_figure_dir}")

print(f"\n📊 Generated files in results directory:")
for f in sorted(output_dir.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"   {f.name} ({size_kb:.1f} KB)")

if appendix_figure_dir.exists():
    print(f"\n📄 Figures copied to paper directory (main metric only):")
    for f in sorted(appendix_figure_dir.glob("C*")):
        size_kb = f.stat().st_size / 1024
        print(f"   {f.name} ({size_kb:.1f} KB)")

print(f"\n📈 Data summary:")
print(f"   Total instances: {len(df):,}")
print(f"   Models: {df['model'].nunique()}")
print(f"   Tasks: {df['task'].nunique()}")
print(f"   Model families: {df['model_family'].nunique()}")
print(f"   Subgroups: {df['subgroup'].nunique()}")
print(f"   Context levels: {df['context_level'].nunique()} ({', '.join(sorted(df['context_level'].unique(), key=lambda x: TASK_CONTEXT_ORDER.get(x, 99)))})")
print(f"   Metrics processed: {len(available_metrics)}")
print(f"   Main metric: {MAIN_METRIC}")

print("\n" + "=" * 80)

if __name__ == '__main__':
    print("\nScript execution completed!")
