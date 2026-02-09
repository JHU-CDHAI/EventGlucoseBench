

"""
Model-Level Performance Comparison (FigureR2)

This script analyzes performance of filtered models across different task types
and context levels. Generates publication-ready tables showing:
1. Overall model performance (averaged across all tasks)
2. Performance stratified by context type (NoCtx, Profile, MediumEvent, DetailedEvent)
3. Performance grouped by task characteristics (diabetes type, age, event type)
4. Multiple metrics (CRPS, glucose-RCRPS, Clarke AB, MAE, RMSE)

Configuration: Uses 1-config.yaml
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import os
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from shared_config import filter_display_tasks, CONTEXT_DISPLAY_NAMES

TOPIC_NAME = '3-R2-llm-with-without-context-FigureR2'

warnings.filterwarnings('ignore')
matplotlib.use('Agg')
sns.set_style("whitegrid")

def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}

repo_root = Path.cwd()
cfg_candidates = [repo_root / '1-config.yaml']
cfg_path = next((p for p in cfg_candidates if p.exists()), cfg_candidates[0])
config = load_config(cfg_path) if cfg_path.exists() else {}

paper_root = config.get('paper_project_root') if isinstance(config, dict) else None
if paper_root:
    pr = Path(paper_root).expanduser().resolve()
    if pr.exists() and pr.is_dir():
        if Path.cwd().resolve() != pr:
            os.chdir(pr)
            print(f"Changed working directory to paper_project_root: {pr}")
        repo_root = pr

results_base = config.get('evaluation_results_folder', 'evaluation/results') if isinstance(config, dict) else 'evaluation/results'
results_root = Path(results_base)
if not results_root.is_absolute():
    results_root = (repo_root / results_root).resolve()

section = config.get(TOPIC_NAME, {}) if isinstance(config, dict) else {}

MAIN_METRIC = config.get('MainMetrics', 'CRPS') if isinstance(config, dict) else 'CRPS'

data_path = results_root / '0-convert-Result-to-model-task-instance-score' / 'clinical_metrics_detailed.csv'
output_dir = results_root / TOPIC_NAME
if isinstance(section, dict) and section.get('output_dir'):
    out = Path(section['output_dir'])
    output_dir = out if out.is_absolute() else (repo_root / out)
output_dir.mkdir(parents=True, exist_ok=True)

script1_section = config.get('1-describe-model-result-data-quality', {}) if isinstance(config, dict) else {}
filtered_models = []
filtered_tasks = []

if isinstance(script1_section, dict):
    grouped_include = script1_section.get('include_models', {})
    if isinstance(grouped_include, dict):
        for group_key, group_config in grouped_include.items():
            if isinstance(group_config, dict):
                include_list = group_config.get('include_models', [])
                filtered_models.extend(include_list)

    exclude_tasks_set = set(script1_section.get('exclude_tasks', []) or [])

EXCLUDED_MODELS = section.get('excluded_models', []) if isinstance(section, dict) else []
context_levels = section.get('context_levels', ['NoCtx', 'Profile', 'MediumEvent', 'DetailedEvent', 'NewMedium', 'NewDetail']) if isinstance(section, dict) else ['NoCtx', 'Profile', 'MediumEvent', 'DetailedEvent', 'NewMedium', 'NewDetail']
metrics_to_analyze = section.get('metrics', ['crps', 'ct_crps', 'rmse', 'clarke_ab', 'mae', 'clarke_cde']) if isinstance(section, dict) else ['crps', 'ct_crps', 'rmse', 'clarke_ab', 'mae', 'clarke_cde']

fig_size = section.get('figure_size', [14, 7]) if isinstance(section, dict) else [14, 7]
font_sizes = section.get('font_sizes', {}) if isinstance(section, dict) else {}
font_title = font_sizes.get('title', 16)
font_axis = font_sizes.get('axis_label', 13)
font_tick = font_sizes.get('tick_label', 20)
font_legend = font_sizes.get('legend', 16)

figure_path = config.get('FigurePath', '0-display/Figure') if isinstance(config, dict) else '0-display/Figure'
if not Path(figure_path).is_absolute():
    figure_path = (repo_root / figure_path).resolve()
else:
    figure_path = Path(figure_path)

print("Environment setup complete")
print(f"Topic: {TOPIC_NAME}")
print(f"Working directory: {Path.cwd()}")
print(f"Output directory: {output_dir}")
print(f"Figure size: {fig_size}")
print(f"Font sizes: title={font_title}, axis={font_axis}, tick={font_tick}, legend={font_legend}")

def save_figure(fig, filename):
    import shutil

    source_path = Path(source_file)

    if source_path.suffix != '.pdf':
        return

    if not figure_path.exists():
        print(f"⚠️  Display folder not found: {figure_path}")
        return

    dest_path = figure_path / source_path.name
    print(f"📋 Copied to paper: {dest_path}")

df = pd.read_csv(data_path)
print(f"Loaded clinical metrics: {len(df):,} rows")

print(f"Filtered models from config: {len(filtered_models)}")
print(f"Excluded tasks from config: {exclude_tasks_set}")

if filtered_models:
    df = df[df['model'].isin(filtered_models)]
    print(f"After model filtering: {len(df):,} rows")

if exclude_tasks_set:
    df = df[~df['task'].isin(exclude_tasks_set)]
    print(f"After task filtering: {len(df):,} rows")

print(f"\nApplying display context filtering...")
df = filter_display_tasks(df, task_column='task', verbose=True)

print(f"Final filtered data: {len(df):,} rows ({len(df['model'].unique())} models × {len(df['task'].unique())} tasks)")

def parse_task_name(task_name):
    parts = task_name.replace('EventCGMTask_', '').split('_')

    result = {
        'diabetes': None,
        'age': None,
        'event': None,
        'context': None
    }

    for part in parts:
        if part in ['D1', 'D2']:
            result['diabetes'] = part
        elif part.startswith('Age'):
            result['age'] = int(part.replace('Age', ''))
        elif part in ['Diet', 'Exercise']:
            result['event'] = part
        elif part in ['NoCtx', 'Profile', 'MediumEvent', 'DetailedEvent', 'NewMedium', 'NewDetail']:
            result['context'] = part

    return result

task_chars = df['task'].apply(parse_task_name).apply(pd.Series)
df = pd.concat([df, task_chars], axis=1)

print("\nTask characteristics added:")
print(f"  Diabetes types: {df['diabetes'].unique()}")
print(f"  Ages: {sorted(df['age'].unique())}")
print(f"  Event types: {df['event'].unique()}")
print(f"  Context levels: {df['context'].unique()}")

def categorize_model(model_name):
    model_lower = model_name.lower()

    if any(x in model_lower for x in ['gpt', 'claude', 'llama-3-8b']):
        return 'direct_prompt'
    elif 'llmp' in model_lower:
        return 'llmp'
    elif any(x in model_lower for x in ['unitime', 'timellm']):
        return 'multimodal'
    elif any(x in model_lower for x in ['chronos', 'lag-llama', 'moirai', 'timegen']):
        return 'ts_foundation'
    elif any(x in model_lower for x in ['arima', 'ets', 'exp-smoothing']):
        return 'statistical'
    else:
        return 'other'

df['model_category'] = df['model'].apply(categorize_model)

print("\nModel categories:")
for cat in df['model_category'].unique():
    models = df[df['model_category'] == cat]['model'].unique()
    print(f"  {cat}: {len(models)} models")
    for model in sorted(models):
        print(f"    - {model}")

print("=" * 80)
print("OVERALL MODEL PERFORMANCE (Averaged across all tasks)")
print("=" * 80)

overall_stats = df.groupby('model').agg({
    'crps': ['mean', 'std', 'sem', 'count'],
    'ct_crps': ['mean', 'std', 'sem'],
    'clarke_ab': ['mean', 'std', 'sem'],
    'mae': ['mean', 'std', 'sem'],
    'rmse': ['mean', 'std', 'sem']
}).round(4)

overall_stats.columns = ['_'.join(col).strip() for col in overall_stats.columns.values]

overall_stats['crps_rank'] = overall_stats['crps_mean'].rank()

overall_stats['category'] = overall_stats.index.map(categorize_model)

overall_stats = overall_stats.sort_values('crps_mean')

print("\nTop 10 models by CRPS:")
print(overall_stats[['crps_mean', 'crps_sem', 'crps_rank', 'category']].head(10))

overall_csv = output_dir / "1-overall_model_performance.csv"
overall_stats.to_csv(overall_csv)
print(f"\n✓ Saved: {overall_csv.name}")

print("\n" + "=" * 80)
print("PERFORMANCE BY CONTEXT LEVEL")
print("=" * 80)

context_stats = df.groupby(['model', 'context']).agg({
    'crps': ['mean', 'sem', 'count']
}).round(4)

context_stats.columns = ['_'.join(col).strip() for col in context_stats.columns.values]

context_pivot = context_stats.reset_index().pivot(
    index='model',
    columns='context',
    values='crps_mean'
)

context_pivot['Overall'] = df.groupby('model')['crps'].mean()

context_pivot['category'] = context_pivot.index.map(categorize_model)

context_pivot = context_pivot.sort_values('Overall')

print("\nPerformance by context level (CRPS mean):")
print(context_pivot)

context_csv = output_dir / "2-performance_by_context.csv"
context_pivot.to_csv(context_csv)
print(f"\n✓ Saved: {context_csv.name}")

print("\n" + "=" * 80)
print("PERFORMANCE BY DIABETES TYPE")
print("=" * 80)

diabetes_stats = df.groupby(['model', 'diabetes']).agg({
    'crps': ['mean', 'sem', 'count']
}).round(4)

diabetes_stats.columns = ['_'.join(col).strip() for col in diabetes_stats.columns.values]

diabetes_pivot = diabetes_stats.reset_index().pivot(
    index='model',
    columns='diabetes',
    values='crps_mean'
)

diabetes_pivot['Overall'] = df.groupby('model')['crps'].mean()
diabetes_pivot['category'] = diabetes_pivot.index.map(categorize_model)
diabetes_pivot = diabetes_pivot.sort_values('Overall')

print("\nPerformance by diabetes type (CRPS mean):")
print(diabetes_pivot)

diabetes_csv = output_dir / "3-performance_by_diabetes_type.csv"
diabetes_pivot.to_csv(diabetes_csv)
print(f"\n✓ Saved: {diabetes_csv.name}")

print("\n" + "=" * 80)
print("PERFORMANCE BY AGE GROUP")
print("=" * 80)

age_stats = df.groupby(['model', 'age']).agg({
    'crps': ['mean', 'sem', 'count']
}).round(4)

age_stats.columns = ['_'.join(col).strip() for col in age_stats.columns.values]

age_pivot = age_stats.reset_index().pivot(
    index='model',
    columns='age',
    values='crps_mean'
)

age_pivot['Overall'] = df.groupby('model')['crps'].mean()
age_pivot['category'] = age_pivot.index.map(categorize_model)
age_pivot = age_pivot.sort_values('Overall')

print("\nPerformance by age group (CRPS mean):")
print(age_pivot)

age_csv = output_dir / "4-performance_by_age.csv"
age_pivot.to_csv(age_csv)
print(f"\n✓ Saved: {age_csv.name}")

print("\n" + "=" * 80)
print("PERFORMANCE BY EVENT TYPE")
print("=" * 80)

event_stats = df.groupby(['model', 'event']).agg({
    'crps': ['mean', 'sem', 'count']
}).round(4)

event_stats.columns = ['_'.join(col).strip() for col in event_stats.columns.values]

event_pivot = event_stats.reset_index().pivot(
    index='model',
    columns='event',
    values='crps_mean'
)

event_pivot['Overall'] = df.groupby('model')['crps'].mean()
event_pivot['category'] = event_pivot.index.map(categorize_model)
event_pivot = event_pivot.sort_values('Overall')

print("\nPerformance by event type (CRPS mean):")
print(event_pivot)

event_csv = output_dir / "5-performance_by_event_type.csv"
event_pivot.to_csv(event_csv)
print(f"\n✓ Saved: {event_csv.name}")

print("\n" + "=" * 80)
print("PUBLICATION-READY TABLE (Mean ± SEM)")
print("=" * 80)

def format_mean_sem(mean, sem):
    if '-context-' in model:
        base = model.replace('-context-', '-')
    elif '-nocontext-' in model:
        base = model.replace('-nocontext-', '-')
    elif model.endswith('-context'):
        base = model.replace('-context', '')
    elif model.endswith('-nocontext'):
        base = model.replace('-nocontext', '')
    else:
        base = model

    if base.startswith('openrouter-'):
        base = base.replace('openrouter-', '')

    if base.startswith('claude-sdk-'):
        base = base.replace('claude-sdk-', 'claude-')

    base = base.replace('-instruct', '-instr')

    return base

all_models_raw = overall_stats.index.tolist()
base_to_originals = {}  # Maps base model name to list of original model names
for model in all_models_raw:
    base = get_base_model_name(model)
    if base not in base_to_originals:
        base_to_originals[base] = []
    base_to_originals[base].append(model)

base_models = list(base_to_originals.keys())

EXCLUDED_MODELS = [
    'chronos-large',
    'chronos-base',
    'chronos-small',
    'chronos-mini',
    'chronos-tiny',
    'lag-llama',
    'moirai-large',
    'moirai-base',
    'moirai-small',
    'r-arima',
    'r-ets',
    'exp-smoothing',
    'unitime-etth1-pred96',
    'timellm-etth1-pred96',
]

llm_base_models = [model for model in base_models if model not in EXCLUDED_MODELS]

print(f"\n✓ Filtering models using config:")
print(f"  Total base models: {len(base_models)}")
print(f"  Excluded models: {len([m for m in base_models if m in EXCLUDED_MODELS])}")
print(f"  Models to display: {len(llm_base_models)}")
print(f"\n  Display models:")
for model in sorted(llm_base_models):
    print(f"    - {model}")

def create_with_without_context_chart(metric_name, metric_column, lower_is_better=True):
    fig, ax = plt.subplots(figsize=fig_size)

    performance_data = []

    for base_model in llm_base_models:
        original_models = base_to_originals[base_model]

        row = {'model': base_model}

        for context in context_levels:
            values = []
            for orig_model in original_models:
                subset = df[(df['model'] == orig_model) & (df['context'] == context)]
                if len(subset) > 0:
                    values.extend(subset[metric_column].dropna().tolist())

            if len(values) > 0:
                row[f'{context}_mean'] = np.mean(values)
                row[f'{context}_sem'] = np.std(values) / np.sqrt(len(values))
            else:
                row[f'{context}_mean'] = np.nan
                row[f'{context}_sem'] = np.nan

        if any(not np.isnan(row.get(f'{ctx}_mean', np.nan)) for ctx in context_levels):
            performance_data.append(row)

    perf_df = pd.DataFrame(performance_data)

    perf_df['sort_key'] = perf_df[[f'{ctx}_mean' for ctx in context_levels]].mean(axis=1)
    if lower_is_better:
        perf_df = perf_df.sort_values('sort_key')
    else:
        perf_df = perf_df.sort_values('sort_key', ascending=False)

    models = perf_df['model'].tolist()
    x = np.arange(len(models))
    n_contexts = len(context_levels)
    width = 0.8 / n_contexts  # Dynamically set width based on number of contexts

    colors = ['#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#9467bd', '#e377c2']  # Blue, Orange, Red, Teal, Purple, Pink

    labels = [CONTEXT_DISPLAY_NAMES.get(ctx, ctx) for ctx in context_levels]

    n_contexts = len(context_levels)
    for i, context in enumerate(context_levels):
        offset = (i - (n_contexts - 1) / 2) * width
        values = perf_df[f'{context}_mean'].values
        errors = perf_df[f'{context}_sem'].values

        ax.bar(x + offset, values, width, label=labels[i], color=colors[i], alpha=0.8,
               yerr=errors, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})

    ax.set_ylabel(metric_name, fontsize=font_axis, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=font_tick)

    all_values = perf_df[[f'{ctx}_mean' for ctx in context_levels]].values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    if len(all_values) > 0:
        y_max = np.max(all_values)
        y_range = y_max

        if metric_column in ['crps', 'ct_crps', 't_crps', 'c_crps']:
            y_min_start = 0.1
        elif metric_column in ['mae', 'rmse']:
            y_min_start = 10
        elif metric_column == 'clarke_ab':
            y_min_start = 0
        else:
            y_min_start = 0.1  # Default

        ax.set_ylim(y_min_start, y_max + 0.1 * y_range)

    ax.tick_params(axis='y', labelsize=font_tick)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))

    ax.legend(title='Context Level', fontsize=font_legend, loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(-0.5, len(models) - 0.5)

    plt.tight_layout()

    fig_filename = f"FigureR2-context_comparison_{metric_column}"
    save_figure(fig, fig_filename)

    copy_to_paper_display(output_dir / f"{fig_filename}.pdf", display_type='Figure')

    print(f"✓ Generated FigureR2: {metric_name} comparison")

print(f"\nGenerating FigureR2 comparisons for metrics: {metrics_to_analyze}")

metric_info = {
    'crps': ('CRPS', True),
    't_crps': ('Temporal-CRPS', True),
    'c_crps': ('Clarke-CRPS', True),
    'ct_crps': ('Clarke-Temporal-CRPS', True),
    'rmse': ('RMSE (mg/dL)', True),
    'clarke_ab': ('Clarke AB %', False),
    'clarke_cde': ('Clarke CDE %', True),
    'mae': ('MAE (mg/dL)', True)
}

for metric in metrics_to_analyze:
    if metric in metric_info:
        display_name, lower_is_better = metric_info[metric]
        create_with_without_context_chart(display_name, metric, lower_is_better=lower_is_better)
    else:
        print(f"⚠️  Unknown metric: {metric}")

print("\n✓ All FigureR2 comparisons generated successfully!")

# =============================================================================
# CREATE MAIN METRIC COPIES
# =============================================================================

def create_main_metric_copy(file_pattern: str):
    if '-context-' in model:
        base = model.replace('-context-', '-')
    elif '-nocontext-' in model:
        base = model.replace('-nocontext-', '-')
    elif model.endswith('-context'):
        base = model.replace('-context', '')
    elif model.endswith('-nocontext'):
        base = model.replace('-nocontext', '')
    else:
        base = model

    if base.startswith('openrouter-'):
        base = base.replace('openrouter-', '')

    if base.startswith('claude-sdk-'):
        base = base.replace('claude-sdk-', 'claude-')

    base = base.replace('-instruct', '-instr')

    return base

all_models_raw = overall_stats.index.tolist()
base_to_originals = {}  # Maps base model name to list of original model names
for model in all_models_raw:
    base = get_base_model_name(model)
    if base not in base_to_originals:
        base_to_originals[base] = []
    base_to_originals[base].append(model)

base_models = list(base_to_originals.keys())

plot_data = pd.DataFrame(index=context_order, columns=base_models)
plot_sem = pd.DataFrame(index=context_order, columns=base_models)

for base_model in base_models:
    original_models = base_to_originals[base_model]

    for context in context_order:
        subset = pd.DataFrame()

        for orig_model in original_models:
            temp_subset = df[(df['model'] == orig_model) & (df['context'] == context)]
            if len(temp_subset) > 0:
                subset = temp_subset
                break

        if len(subset) > 0:
            plot_data.loc[context, base_model] = subset['crps'].mean()
            plot_sem.loc[context, base_model] = subset['crps'].sem()
        else:
            plot_data.loc[context, base_model] = np.nan
            plot_sem.loc[context, base_model] = np.nan

plot_data = plot_data.astype(float)
plot_sem = plot_sem.astype(float)

column_means = plot_data.mean(axis=0)
sorted_columns = column_means.sort_values().index.tolist()
plot_data = plot_data[sorted_columns]
plot_sem = plot_sem[sorted_columns]

merged_models = plot_data.columns.tolist()
x = np.arange(len(merged_models))
n_contexts = len(context_order)
width = 0.8 / n_contexts  # Dynamically set width based on number of contexts

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2']  # Blue, Orange, Green, Red, Purple, Pink

for i, context in enumerate(context_order):
    offset = (i - (n_contexts - 1) / 2) * width
    values = plot_data.loc[context].values
    errors = plot_sem.loc[context].values
    ax.bar(x + offset, values, width, label=context, color=colors[i], alpha=0.8,
           yerr=errors, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})

ax.set_title('Model Performance by Context Level\n(All Models with Context Variants Merged, Lower CRPS is Better)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('CRPS', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(merged_models, rotation=45, ha='right', fontsize=8)
ax.legend(title='Context Level', loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

ax.set_xlim(-0.5, len(merged_models) - 0.5)

plt.tight_layout()
save_figure(fig, "8-performance_by_context_chart_crps")

def create_metric_chart(metric_name, metric_column, lower_is_better=True, figure_number=9):
    fig, ax = plt.subplots(figsize=(18, 10))

    plot_data_metric = pd.DataFrame(index=context_order, columns=base_models)
    plot_sem_metric = pd.DataFrame(index=context_order, columns=base_models)

    for base_model in base_models:
        original_models = base_to_originals[base_model]

        for context in context_order:
            subset = pd.DataFrame()

            for orig_model in original_models:
                temp_subset = df[(df['model'] == orig_model) & (df['context'] == context)]
                if len(temp_subset) > 0:
                    subset = temp_subset
                    break

            if len(subset) > 0 and metric_column in subset.columns:
                plot_data_metric.loc[context, base_model] = subset[metric_column].mean()
                plot_sem_metric.loc[context, base_model] = subset[metric_column].sem()
            else:
                plot_data_metric.loc[context, base_model] = np.nan
                plot_sem_metric.loc[context, base_model] = np.nan

    plot_data_metric = plot_data_metric.astype(float)
    plot_sem_metric = plot_sem_metric.astype(float)

    column_means = plot_data_metric.mean(axis=0)
    if lower_is_better:
        sorted_columns = column_means.sort_values().index.tolist()
    else:
        sorted_columns = column_means.sort_values(ascending=False).index.tolist()
    plot_data_metric = plot_data_metric[sorted_columns]
    plot_sem_metric = plot_sem_metric[sorted_columns]

    merged_models_metric = plot_data_metric.columns.tolist()
    x = np.arange(len(merged_models_metric))
    n_contexts = len(context_order)
    width = 0.8 / n_contexts  # Dynamically set width based on number of contexts

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2']  # Blue, Orange, Green, Red, Purple, Pink

    for i, context in enumerate(context_order):
        offset = (i - (n_contexts - 1) / 2) * width
        values = plot_data_metric.loc[context].values
        errors = plot_sem_metric.loc[context].values
        ax.bar(x + offset, values, width, label=context, color=colors[i], alpha=0.8,
               yerr=errors, capsize=3, error_kw={'elinewidth': 1, 'capthick': 1})

    better_text = "Lower is Better" if lower_is_better else "Higher is Better"
    ax.set_title(f'Model Performance by Context Level: {metric_name}\n(All Models with Context Variants Merged, {better_text})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(merged_models_metric, rotation=45, ha='right', fontsize=8)
    ax.legend(title='Context Level', loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(-0.5, len(merged_models_metric) - 0.5)

    plt.tight_layout()
    save_figure(fig, f"{figure_number}-performance_by_context_chart_{metric_column}")

print("\n" + "=" * 80)
print("GENERATING PERFORMANCE CHARTS FOR ALL METRICS")
print("=" * 80)

create_metric_chart("Clarke-Temporal-CRPS", "ct_crps", lower_is_better=True, figure_number=9)
print("✓ Generated: Clarke-Temporal-CRPS chart")

create_metric_chart("Clarke AB %", "clarke_ab", lower_is_better=False, figure_number=10)
print("✓ Generated: Clarke AB chart")

create_metric_chart("MAE (mg/dL)", "mae", lower_is_better=True, figure_number=11)
print("✓ Generated: MAE chart")

create_metric_chart("RMSE (mg/dL)", "rmse", lower_is_better=True, figure_number=12)
print("✓ Generated: RMSE chart")

print("\n" + "=" * 80)
print("SUMMARY REPORT")
print("=" * 80)

print(f"\n📊 Dataset:")
print(f"   - Models: {len(filtered_models)}")
print(f"   - Tasks: {len(filtered_tasks)}")
print(f"   - Total instances: {len(df):,}")

print(f"\n🏆 Best Model Overall:")
best_model = overall_stats.index[0]
best_crps = overall_stats.iloc[0]['crps_mean']
best_sem = overall_stats.iloc[0]['crps_sem']
print(f"   - Model: {best_model}")
print(f"   - CRPS: {best_crps:.4f} ± {best_sem:.4f}")

print(f"\n📈 Performance Ranges:")
print(f"   - CRPS: [{overall_stats['crps_mean'].min():.3f}, {overall_stats['crps_mean'].max():.3f}]")
print(f"   - Clarke-Temporal-CRPS: [{metric_df['ct_crps_mean'].min():.3f}, {metric_df['ct_crps_mean'].max():.3f}]")
print(f"   - Clarke AB: [{metric_df['clarke_ab_mean'].min():.3f}, {metric_df['clarke_ab_mean'].max():.3f}]")

print(f"\n📁 All outputs saved to: {output_dir}")
print("\nGenerated Files:")
print("   CSV Files:")
print("   - 1-overall_model_performance.csv")
print("   - 2-performance_by_context.csv")
print("   - 3-performance_by_diabetes_type.csv")
print("   - 4-performance_by_age.csv")
print("   - 5-performance_by_event_type.csv")
print("   - 6-publication_table.csv")
print("   - 7-multi_metric_comparison.csv")
print("\n   FigureR2 Files (4-Bar Context Comparison):")
print("   - FigureR2-context_comparison_crps.png/.pdf")
print("   - FigureR2-context_comparison_ct_crps.png/.pdf")
print("   - FigureR2-context_comparison_rmse.png/.pdf")
print("   - FigureR2-context_comparison_clarke_ab.png/.pdf")
print("   - FigureR2-context_comparison_mae.png/.pdf")
print("\n   Supplementary Figure Files (All Context Levels):")
print("   - 8-performance_by_context_chart_crps.png/.pdf")
print("   - 9-performance_by_context_chart_ct_crps.png/.pdf")
print("   - 10-performance_by_context_chart_clarke_ab.png/.pdf")
print("   - 11-performance_by_context_chart_mae.png/.pdf")
print("   - 12-performance_by_context_chart_rmse.png/.pdf")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)

if __name__ == '__main__':
    print("\nScript execution completed!")
