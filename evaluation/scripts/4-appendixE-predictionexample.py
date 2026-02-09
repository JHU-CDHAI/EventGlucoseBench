"""

for each task type based on evaluation metrics (CRPS).

For each task type (Diet, Exercise, etc.):

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
import shutil

TOPIC_NAME = '4-appendixE-examples'

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

pkl_dir = output_dir / 'pkl_data'
pkl_dir.mkdir(parents=True, exist_ok=True)

figure_dir = output_dir / 'figures'
figure_dir.mkdir(parents=True, exist_ok=True)

model_results = config.get('model_inference_results', '_WorkSpace/Result') if isinstance(config, dict) else '_WorkSpace/Result'
result_dir = Path(model_results)
if not result_dir.is_absolute():
    result_dir = (repo_root / result_dir).resolve()

if not result_dir.exists():
    eventglucose_root = repo_root.parent.parent
    fallback_path = eventglucose_root / '_WorkSpace' / 'Result'

    if fallback_path.exists():
        result_dir = fallback_path.resolve()
        print(f"⚠️  Config path not found, using fallback: {result_dir}")
    else:
        raise FileNotFoundError(
            f"\n{'='*80}\n"
            f"{'='*80}\n"
            f"  Configured path: {Path(model_results)}\n"
            f"{'='*80}\n"
        )

appendix_figure_path = config.get('AppendixFigurePath', '0-display/AppendixFigure') if isinstance(config, dict) else '0-display/AppendixFigure'
if not Path(appendix_figure_path).is_absolute():
    appendix_figure_path = (repo_root / appendix_figure_path).resolve()
else:
    appendix_figure_path = Path(appendix_figure_path)
appendix_figure_path.mkdir(parents=True, exist_ok=True)

print("Environment setup complete")
print(f"Topic: {TOPIC_NAME}")
print(f"Working directory: {Path.cwd()}")
print(f"Result directory: {result_dir}")
print(f"Output directory: {output_dir}")
print(f"PKL data directory: {pkl_dir}")
print(f"Figure directory: {figure_dir}")
print(f"Appendix figure path: {appendix_figure_path}")

csv_path = results_root / '0-convert-Result-to-model-task-instance-score' / 'clinical_metrics_detailed.csv'
df = pd.read_csv(csv_path)

print(f"\nLoaded {len(df)} instances from CSV")

def get_base_model(model_name):
    return task_name.replace('_NoCtx', '').replace('_DetailedEvent', '').replace('_Profile', '').replace('_MediumEvent', '').replace('_NewMedium', '').replace('_NewDetail', '')

def get_event_type(task_name):
    if 'NewDetail' in task_name:
    elif 'NewMedium' in task_name:
    elif 'DetailedEvent' in task_name:
    elif 'MediumEvent' in task_name:
    elif 'Profile' in task_name:
    elif 'NoCtx' in task_name:

def is_llm_model(model_name):
    png_path = figure_dir / f"{filename}.png"
    pdf_path = figure_dir / f"{filename}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"  Saved: {png_path.name} and {pdf_path.name}")

    if save_to_appendix and appendix_figure_path.exists():
        appendix_pdf = appendix_figure_path / pdf_path.name
        print(f"  Copied to appendix: {appendix_pdf}")

    plt.close(fig)

def load_prediction_data(model, task, seed):
    output_data = {
        'event_type': event_type,
        'model': model,
        'task': task,
        'seed': seed,
        'metrics': metrics,
        'prediction_data': data
    }

    filename = f"{example_type}_{event_type}_{model}_{task}_seed{seed}.pkl"
    filename = filename.replace('/', '_').replace(' ', '_')

    pkl_path = pkl_dir / filename
    with open(pkl_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"  Saved PKL: {pkl_path.name}")

def create_prediction_figure(data, row, example_type, figure_number):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

        (ax1, best_data, best_row, 'best'),
        (ax2, worst_data, worst_row, 'worst')
    ]:
        history = data['input_data']['past_time']
        future = data['input_data']['future_time']
        predictions = data['predictions']['samples']

        pred_median = np.median(predictions[:, :, 0], axis=0)
        pred_q05 = np.percentile(predictions[:, :, 0], 5, axis=0)
        pred_q25 = np.percentile(predictions[:, :, 0], 25, axis=0)
        pred_q75 = np.percentile(predictions[:, :, 0], 75, axis=0)
        pred_q95 = np.percentile(predictions[:, :, 0], 95, axis=0)

        history_window = 96
        history_start = max(0, len(history) - history_window)
        history_show = history[history_start:]

        history_time = np.arange(-len(history_show), 0) * (5/60)
        future_time = np.arange(0, len(future)) * (5/60)

        model_display = row['model']
        if model_display.startswith('openrouter-'):
            model_display = model_display.replace('openrouter-', '')
        if model_display.startswith('claude-sdk-'):
            model_display = model_display.replace('claude-sdk-', 'Claude-')

        ax.plot(history_time, history_show, 'k-', linewidth=2, label='History')
        ax.plot(future_time, future, 'g-', linewidth=2, label='Ground Truth')
        ax.plot(future_time, pred_median, color='orange', linewidth=2, label='Forecast')

        ax.fill_between(future_time, pred_q05, pred_q95,
                        color='darkblue', alpha=0.3, label='5%-95%')
        ax.fill_between(future_time, pred_q25, pred_q75,
                        color='mediumblue', alpha=0.5, label='25%-75%')

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvspan(0, future_time[-1], alpha=0.1, color='lightgreen')

        if example_type == 'best':
            box_color = '#90EE90'
            title_prefix = 'BEST'
        else:
            box_color = '#FFB6C1'
            title_prefix = 'WORST'

        ax.set_xlabel('Forecast Horizon (h)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Glucose (mg/dL)', fontsize=12, fontweight='bold')
        ax.set_title(f'{row["task"]}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([history_time[0], future_time[-1]])

        crps_val = row.get('crps', float('nan'))
        rmse_val = row.get('rmse', float('nan'))
        ax.text(0.95, 0.05, f'CRPS: {crps_val:.1f}\nRMSE: {rmse_val:.1f}',
                transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8))

    plt.suptitle(f'{event_type} Event - {context_level}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

results_df = pd.DataFrame(results)

for event_type in ['Diet', 'Exercise']:
    for context_level in ['NoCtx', 'DetailedEvent', 'NewMedium', 'NewDetail']:
        subset = results_df[
            (results_df['event_type'] == event_type) &
            (results_df['context_level'] == context_level)
        ]

        best_rows = subset[subset['example_type'] == 'best']
        worst_rows = subset[subset['example_type'] == 'worst']

        if len(best_rows) == 0 or len(worst_rows) == 0:
            continue

        best_row = best_rows.iloc[0]
        worst_row = worst_rows.iloc[0]

        best_data = load_prediction_data(best_row['model'], best_row['task'], best_row['seed'])
        worst_data = load_prediction_data(worst_row['model'], worst_row['task'], worst_row['seed'])

        if best_data is None or worst_data is None:
            continue

        fig = create_comparison_figure(best_row, worst_row, best_data, worst_data, event_type, context_level)
        filename = f"E-{event_type}_{context_level}-comparison"
        save_figure(fig, filename, save_to_appendix=True)

        print(f"Created comparison figure: {filename}")

print("\n" + "=" * 80)
print("SAVING SUMMARY")
print("=" * 80)

summary_file = output_dir / "appendixE_examples_summary.csv"
results_df.to_csv(summary_file, index=False)
print(f"Saved summary: {summary_file}")

try: