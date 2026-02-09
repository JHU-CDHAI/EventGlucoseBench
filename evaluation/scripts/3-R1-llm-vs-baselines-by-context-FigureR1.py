"""

across different context levels (NoCtx, Profile, MediumEvent, DetailedEvent).

Key insight: For context-aware tasks (Profile, Medium, Detailed), we compare

        evaluation/results/0-convert-Result-to-model-task-instance-score/clinical_metrics_detailed.csv
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import yaml
from pathlib import Path
import warnings
import os
import sys
import shutil

sys.path.insert(0, str(Path(__file__).parent))
from shared_config import filter_display_tasks, CONTEXT_DISPLAY_NAMES

TOPIC_NAME = '3-R1-llm-vs-baselines-by-context-FigureR1'

warnings.filterwarnings('ignore')

matplotlib.use('Agg')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 4)

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

data_path = results_root / '0-convert-Result-to-model-task-instance-score' / 'clinical_metrics_detailed.csv'
filtered_models_tasks_path = results_root / '1-describe-model-result-data-quality' / '7-filtered_models_and_tasks.json'
output_dir = results_root / TOPIC_NAME
if isinstance(section, dict) and section.get('output_dir'):
    out = Path(section['output_dir'])
    output_dir = out if out.is_absolute() else (repo_root / out)
output_dir.mkdir(parents=True, exist_ok=True)

print("Environment setup complete")
print(f"Topic: {TOPIC_NAME}")
print(f"Working directory: {Path.cwd()}")
print(f"Output directory: {output_dir}")

def save_figure(fig, filename):

    Args:

    Examples:

    Examples:
        EventCGMTask_D1_Age65_Diet_Ontime_NoCtx -> EventCGMTask_D1_Age65_Diet_Ontime_NoCtx (unchanged)

    Args:
        context_level: Context level to analyze (NoCtx, Profile, etc.)
        metric: Metric to use for comparison ('crps', 'ct_crps', 'rmse', 'clarke_ab')

    Returns:
        DataFrame with columns: [llm_model, task, task_baseline, n_baselines_beaten]
    if n_baselines_beaten == 6:
    elif n_baselines_beaten == 5:
    elif n_baselines_beaten in [3, 4]:
    elif n_baselines_beaten in [1, 2]:
    else:  # n_baselines_beaten == 0

def aggregate_llm_performance(comparison_df):
    comparison_df['bucket'] = comparison_df['n_baselines_beaten'].apply(assign_bucket)

    bucket_counts = comparison_df.groupby(['llm_model', 'bucket']).size().reset_index(name='count')

    total_tasks = comparison_df.groupby('llm_model').size().reset_index(name='total')

    result = bucket_counts.merge(total_tasks, on='llm_model')
    result['proportion'] = result['count'] / result['total']

all_aggregated_results = {}

for metric in METRICS:
    print(f"\n{'='*80}")
    print(f"Aggregating results for {metric.upper()}")
    print(f"{'='*80}")

    all_aggregated_results[metric] = {}
    for context in CONTEXT_LEVELS:
        print(f"  Aggregating {context}...")
        all_aggregated_results[metric][context] = aggregate_llm_performance(all_comparisons[metric][context])
        print(f"    ✓ {len(all_aggregated_results[metric][context])} bucket entries")

aggregated_results = all_aggregated_results['crps']

def get_base_model_name(model):
    pass

    Args:
        context_groups: Dict like {'With context': ['Profile', 'MediumEvent', 'DetailedEvent'],
                                   'Without context': ['NoCtx']}

    Returns:

    Args:
        context_level: Name of context level (e.g., 'NoCtx')
        aggregated_df: DataFrame with columns [llm_model, bucket, proportion]
        model_order: List of base model names in desired order (best to worst)
        show_ylabel: Whether to show y-axis labels (only for leftmost subplot)

    figure_path_config = config.get('FigurePath', '0-display/Figure') if isinstance(config, dict) else '0-display/Figure'
    if not Path(figure_path_config).is_absolute():
        figure_path = (repo_root / figure_path_config).resolve()
    else:
        figure_path = Path(figure_path_config)

    main_metric_norm = MAIN_METRIC.lower().replace('-', '_')

    for ext in ['.png', '.pdf']:
        main_file = None
        for metric_variant in [MAIN_METRIC, main_metric_norm, 'crps']:
            candidate = output_dir / f"{file_pattern}_{metric_variant}{ext}"
            if candidate.exists():
                main_file = candidate
                break

        if main_file:
            main_copy = output_dir / f"{file_pattern}_Main{ext}"
            print(f"Created Main copy: {main_copy.name}")

            if ext == '.pdf':
                if figure_path.exists():
                    paper_main = figure_path / main_copy.name
                    print(f"Copied Main to paper: {paper_main}")
                else:
                    print(f"Warning: Paper directory not found: {figure_path}")

print("\n" + "=" * 80)
print(f"CREATING MAIN METRIC COPIES ({MAIN_METRIC})")
print("=" * 80)

create_main_metric_copy("FigureR1-llm_vs_baselines_all_contexts")

print("\n" + "=" * 80)
print("SAVING SUMMARY TABLES (ALL METRICS)")
print("=" * 80)

csv_counter = 4  # Start after the 3 figure files

for metric in METRICS:
    print(f"\n{'='*80}")
    print(f"Saving tables for {metric.upper()}")
    print(f"{'='*80}")

    for context in CONTEXT_LEVELS:
        csv_counter += 1
        comparison_csv_path = output_dir / f"{csv_counter}-comparison_details_{metric}_{context.lower()}.csv"
        all_comparisons[metric][context].to_csv(comparison_csv_path, index=False)
        print(f"✓ Saved: {comparison_csv_path.name}")

    for context in CONTEXT_LEVELS:
        csv_counter += 1
        aggregated_csv_path = output_dir / f"{csv_counter}-aggregated_results_{metric}_{context.lower()}.csv"
        all_aggregated_results[metric][context].to_csv(aggregated_csv_path, index=False)
        print(f"✓ Saved: {aggregated_csv_path.name}")

    print(f"\nCreating combined summary table for {metric}...")
    combined_summary = []

    for context in CONTEXT_LEVELS:
        agg = all_aggregated_results[metric][context].copy()
        agg['context'] = context
        agg['metric'] = metric
        combined_summary.append(agg)

    combined_df = pd.concat(combined_summary, ignore_index=True)
    csv_counter += 1
    combined_csv_path = output_dir / f"{csv_counter}-combined_summary_{metric}.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"✓ Saved: {combined_csv_path.name}")

print("\n" + "=" * 80)
print("SUMMARY REPORT")
print("=" * 80)

print(f"\n📊 Dataset:")
print(f"   - Models: {len(filtered_models)}")
print(f"   - Tasks: {len(filtered_tasks)}")
print(f"   - LLM models: {len(set([m for ctx in CONTEXT_LEVELS for m in aggregated_results[ctx]['llm_model'].unique()]))}")
print(f"   - Quantitative baselines: {len(QUANTITATIVE_BASELINES)}")

print(f"\n🎯 Context Levels Analyzed:")
for context in CONTEXT_LEVELS:
    n_comparisons = len(context_comparisons[context])
    print(f"   - {context}: {n_comparisons} LLM-task comparisons")

print(f"\n🏆 Best Performers by Metric:")
for metric in METRICS:
    print(f"\n  {metric_names[metric]}:")
    for context in CONTEXT_LEVELS:
        agg = all_aggregated_results[metric][context]
        beats_all = agg[agg['bucket'] == 'Beats all']
        if len(beats_all) > 0:
            best = beats_all.loc[beats_all['proportion'].idxmax()]
            base_name = get_base_model_name(best['llm_model'])
            print(f"     {context}: {base_name} ({best['proportion']*100:.1f}% beat all 6 baselines)")

print(f"\n📁 All outputs saved to: {output_dir}")
print("\nGenerated Files:")
print(f"   Figure Files ({len(METRICS)} metrics):")
for i, metric in enumerate(METRICS, start=1):
    print(f"   - {i}-llm_vs_baselines_all_contexts_{metric}.png/.pdf")
print(f"\n   CSV Files ({len(METRICS) * 9} files total):")
print(f"   - Comparison details: 4 files per metric × {len(METRICS)} metrics")
print(f"   - Aggregated results: 4 files per metric × {len(METRICS)} metrics")
print(f"   - Combined summaries: 1 file per metric × {len(METRICS)} metrics")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)

if __name__ == '__main__':
    print("\nScript execution completed!")
