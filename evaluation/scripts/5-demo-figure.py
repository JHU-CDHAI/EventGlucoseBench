"""

contextual information (diet events, exercise events) dramatically improves

For each example, we show side-by-side comparisons:

"""

import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pickle
import warnings
import os
import yaml
import re

TOPIC_NAME = '5-demo-figure'

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

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

output_dir = results_root / TOPIC_NAME
if isinstance(section, dict) and section.get('output_dir'):
    out = Path(section['output_dir'])
    output_dir = out if out.is_absolute() else (repo_root / out)
output_dir.mkdir(parents=True, exist_ok=True)

model_results = config.get('model_inference_results', '_WorkSpace/Result') if isinstance(config, dict) else '_WorkSpace/Result'
result_dir = Path(model_results)
if not result_dir.is_absolute():
    result_dir = (repo_root / result_dir).resolve()

figure_path = config.get('FigurePath', '0-display/Figure') if isinstance(config, dict) else '0-display/Figure'
if not Path(figure_path).is_absolute():
    figure_path = (repo_root / figure_path).resolve()
else:
    figure_path = Path(figure_path)

print("Environment setup complete")
print(f"Topic: {TOPIC_NAME}")
print(f"Working directory: {Path.cwd()}")
print(f"Result directory: {result_dir}")
print(f"Output directory: {output_dir}")
print(f"Paper figure path: {figure_path}")

csv_path = results_root / '0-convert-Result-to-model-task-instance-score' / 'clinical_metrics_detailed.csv'
df = pd.read_csv(csv_path)

print(f"\nLoaded {len(df)} instances from CSV")

def get_base_model(model_name):
    return task_name.replace('_NoCtx', '').replace('_DetailedEvent', '')

def get_event_type(task_name):
    png_path = output_dir / f"{filename}.png"
    pdf_path = output_dir / f"{filename}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Saved: {png_path.name} and {pdf_path.name}")

    if figure_path.exists():
        paper_pdf = figure_path / pdf_path.name
        print(f"📋 Copied to paper: {paper_pdf}")

    plt.close(fig)

print("\n" + "=" * 80)
print("FINDING BEST EXAMPLES FOR DEMONSTRATION")
print("=" * 80)

df['base_model'] = df['model'].apply(get_base_model)
df['base_task'] = df['task'].apply(get_base_task)
df['event_type'] = df['task'].apply(get_event_type)

event_df = df[df['event_type'].isin(['Diet', 'Exercise'])].copy()

noctx_df = event_df[event_df['task'].str.contains('NoCtx', na=False)].copy()
detailed_df = event_df[event_df['task'].str.contains('DetailedEvent', na=False)].copy()

merged = pd.merge(
    on=['base_model', 'base_task', 'seed'],
    suffixes=('_noctx', '_detailed')
)

merged['crps_improvement_pct'] = ((merged['crps_noctx'] - merged['crps_detailed']) / merged['crps_noctx']) * 100
merged['rmse_improvement_pct'] = ((merged['rmse_noctx'] - merged['rmse_detailed']) / merged['rmse_noctx']) * 100

def model_priority(model_name):
    event_df = df[df['event_type'] == event_type].copy()
    selected = []
    used_models = set()

    for _, row in event_df.iterrows():
        if len(selected) >= n_examples:
            break
        if row['base_model'] not in used_models:
            selected.append(row)
            used_models.add(row['base_model'])

    if len(selected) < n_examples:
        for _, row in event_df.iterrows():
            if len(selected) >= n_examples:
                break
            if row['base_model'] not in [s['base_model'] for s in selected]:
                selected.append(row)

    return pd.DataFrame(selected)

diet_examples = select_diverse_examples(merged, 'Diet', n_examples=10)
exercise_examples = select_diverse_examples(merged, 'Exercise', n_examples=10)

print(f"\nFound {len(diet_examples)} diet examples and {len(exercise_examples)} exercise examples")
print(f"Diet models: {diet_examples['base_model'].tolist()}")
print(f"Exercise models: {exercise_examples['base_model'].tolist()}")

selected_examples = pd.concat([diet_examples.head(10), exercise_examples.head(10)])

print("\n" + "=" * 80)
print("SELECTED EXAMPLES FOR FIGURES:")
print("=" * 80)

for idx, row in selected_examples.iterrows():
    print(f"\nExample {idx+1}: {row['event_type']} Event")
    print(f"  Model: {row['base_model']}")
    print(f"  Task: {row['base_task']}")
    print(f"  Seed: {row['seed']}")
    print(f"  NoCtx: CRPS={row['crps_noctx']:.2f}, RMSE={row['rmse_noctx']:.2f}")
    print(f"  Detailed: CRPS={row['crps_detailed']:.2f}, RMSE={row['rmse_detailed']:.2f}")
    print(f"  Improvement: CRPS {row['crps_improvement_pct']:.1f}%, RMSE {row['rmse_improvement_pct']:.1f}%")

def create_demo_figure(row, figure_number):