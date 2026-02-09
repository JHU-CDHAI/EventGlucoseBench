"""
Parameter Performance Analysis (FigureR3)

defined in config.yaml (CRPS, Glucose-RCRPS, RMSE, MAE, Clarke AB).

Features:

"""

from __future__ import annotations

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import os
import sys
import re
import subprocess
import yaml
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from shared_config import filter_display_tasks, CONTEXT_DISPLAY_NAMES

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.lines import Line2D

matplotlib.use('Agg')

TOPIC_NAME = '3-R3-Parameter-Performance-FigureR3'

# ============================================================================
# ============================================================================

def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}

def find_project_root() -> Path:
    env_file = project_root / "env.sh"

    if not env_file.exists():
        print(f"Warning: Could not find env.sh at: {env_file}")

    print(f"Found env.sh at: {env_file}")
    try:
        cmd = f'source "{env_file}" && env'
        result = subprocess.run(
            capture_output=True,
            text=True,
            check=True,
            cwd=str(project_root)
        )

        for line in result.stdout.splitlines():
            if '=' in line:
                key, value = line.split('=', 1)
                if key not in os.environ:
                    os.environ[key] = value

        print("Successfully loaded environment from env.sh")

    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to source env.sh: {e}")

PROJECT_ROOT = find_project_root()
os.chdir(PROJECT_ROOT)
print(f"Changed working directory to: {os.getcwd()}")

cfg_candidates = [PROJECT_ROOT / '1-config.yaml']
cfg_path = next((p for p in cfg_candidates if p.exists()), cfg_candidates[0])
config = load_config(cfg_path) if cfg_path.exists() else {}

paper_root = config.get('paper_project_root') if isinstance(config, dict) else None
if paper_root:
    pr = Path(paper_root).expanduser().resolve()
    if pr.exists() and pr.is_dir():
        if Path.cwd().resolve() != pr:
            os.chdir(pr)
            print(f"Changed working directory to paper_project_root: {pr}")
        PROJECT_ROOT = pr

load_env_from_shell(PROJECT_ROOT)

results_base = config.get('evaluation_results_folder', 'evaluation/results') if isinstance(config, dict) else 'evaluation/results'
results_root = Path(results_base)
if not results_root.is_absolute():
    results_root = (PROJECT_ROOT / results_base).resolve()

section = config.get(TOPIC_NAME, {}) if isinstance(config, dict) else {}
OUTPUT_DIR = results_root / TOPIC_NAME
if isinstance(section, dict) and section.get('output_dir'):
    out = Path(section['output_dir'])
    OUTPUT_DIR = out if out.is_absolute() else (PROJECT_ROOT / out)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = OUTPUT_DIR

DATA_PATH = results_root / '0-convert-Result-to-model-task-instance-score' / 'clinical_metrics_detailed.csv'

FIG_DPI = 150

MAIN_METRIC = config.get('MainMetrics', 'CRPS') if isinstance(config, dict) else 'CRPS'
print(f"Main metric: {MAIN_METRIC}")

fig_size = section.get('figure_size', [14, 8]) if isinstance(section, dict) else [14, 8]
font_sizes = section.get('font_sizes', {}) if isinstance(section, dict) else {}
font_title = font_sizes.get('title', 14)
font_axis = font_sizes.get('axis_label', 12)
font_tick = font_sizes.get('tick_label', 10)
font_legend = font_sizes.get('legend', 9)

PARAM_COUNT = section.get('parameter_counts', {}) if isinstance(section, dict) else {}

print("\n" + "=" * 80)
print("LOADING MODELS FROM CONFIG")
print("=" * 80)
script1_section = config.get('1-describe-model-result-data-quality', {}) if isinstance(config, dict) else {}
VALID_MODELS = set()

if isinstance(script1_section, dict):
    grouped_include = script1_section.get('include_models', {})
    if isinstance(grouped_include, dict):
        for group_key, group_config in grouped_include.items():
            if isinstance(group_config, dict):
                include_list = group_config.get('include_models', [])
                VALID_MODELS.update(include_list)
                print(f"  {group_key}: {len(include_list)} models")

    exclude_tasks = set(script1_section.get('exclude_tasks', []) or [])
else:
    exclude_tasks = set()

print(f"\nInput data: {DATA_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Valid models from config: {len(VALID_MODELS)}")
print(f"Figure size: {fig_size}")
print(f"Font sizes: title={font_title}, axis={font_axis}, tick={font_tick}, legend={font_legend}")

sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = tuple(fig_size)
plt.rcParams['font.size'] = font_tick

# ============================================================================
# ============================================================================

def save_figure(fig, filename: str, show_in_notebook: bool = True):

    figure_path_config = config.get('FigurePath', '0-display/Figure') if isinstance(config, dict) else '0-display/Figure'
    if not Path(figure_path_config).is_absolute():
        figure_path = (PROJECT_ROOT / figure_path_config).resolve()
    else:
        figure_path = Path(figure_path_config)

    main_metric_norm = MAIN_METRIC.lower().replace('-', '_')

    for ext in ['.png', '.pdf']:
        main_file = None
        for metric_variant in [MAIN_METRIC, main_metric_norm, 'crps']:
            candidate = FIGURES_DIR / f"{file_pattern}_{metric_variant}{ext}"
            if candidate.exists():
                main_file = candidate
                break

        if main_file:
            main_copy = FIGURES_DIR / f"{file_pattern}_Main{ext}"
            print(f"Created Main copy: {main_copy.name}")

            if ext == '.pdf':
                if figure_path.exists():
                    paper_main = figure_path / main_copy.name
                    print(f"Copied Main to paper: {paper_main}")
                else:
                    print(f"Warning: Paper directory not found: {figure_path}")

# ============================================================================
# ============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

if not DATA_PATH.exists():
    raise FileNotFoundError(
    )

print(f"Loading data from: {DATA_PATH}")
df_results = pd.read_csv(DATA_PATH)

print(f"Initial shape: {df_results.shape}")
print(f"Initial unique models: {df_results['model'].nunique()}")

if VALID_MODELS:
    df_before = len(df_results)
    models_before = df_results['model'].nunique()
    df_results = df_results[df_results['model'].isin(VALID_MODELS)]
    print(f"Filtered to valid models: {df_before} -> {len(df_results)} rows, {models_before} -> {df_results['model'].nunique()} models")

if exclude_tasks:
    df_before = len(df_results)
    models_before = df_results['model'].nunique()
    df_results = df_results[~df_results['task'].isin(exclude_tasks)]
    print(f"Filtered excluded tasks: {df_before} -> {len(df_results)} rows, {models_before} -> {df_results['model'].nunique()} models")

print(f"\nApplying display context filtering...")
df_results = filter_display_tasks(df_results, task_column='task', verbose=True)

print(f"\n{'='*60}")
print("DataFrame Summary:")
print(f"{'='*60}")
print(f"Shape: {df_results.shape}")
print(f"Models loaded: {df_results['model'].nunique()} unique models")
print(f"Unique models: {sorted(df_results['model'].unique())}")
print(f"Tasks loaded: {df_results['task'].nunique()} unique tasks")

expected_metrics = config.get('MetricsList', ['crps']) if isinstance(config, dict) else ['crps']
available_metrics_in_data = [m for m in expected_metrics if m in df_results.columns]
missing_metrics = [m for m in expected_metrics if m not in df_results.columns]

print(f"\nMetric Availability:")
print(f"  Expected from config: {expected_metrics}")
print(f"  Available in data: {available_metrics_in_data}")
if missing_metrics:
    print(f"  ⚠️  Missing: {missing_metrics}")

# ============================================================================
# ============================================================================

def get_task_attributes_from_json(task_name: str) -> dict:
    task_attrs = df['task'].apply(get_task_attributes_from_json)
    df_parsed = pd.DataFrame(list(task_attrs))
    df = df.reset_index(drop=True)
    df_parsed = df_parsed.reset_index(drop=True)
    result = pd.concat([df, df_parsed], axis=1)
    if result['model'].isna().any():
        print(f"WARNING: Found {result['model'].isna().sum()} rows with NaN model, dropping...")
        result = result[result['model'].notna()]

df_results = add_task_attributes(df_results)

print("=" * 60)
print("After add_task_attributes:")
print("=" * 60)
print(f"Models: {df_results['model'].nunique()}")
print(f"Rows: {len(df_results)}")
unique_models = [m for m in df_results['model'].unique() if pd.notna(m)]
print(f"Unique model values ({len(unique_models)}): {sorted(unique_models)}")

print("\n" + "=" * 60)
print("Task Attributes Summary:")
print("=" * 60)
print(f"\nContext Level (context_level):")
print(df_results['context_level'].value_counts())

# ============================================================================
# ============================================================================

MODELS_CONFIG = {}

for model_name in VALID_MODELS:
    model_lower = model_name.lower()
    if 'gpt' in model_lower or 'claude' in model_lower or 'llama' in model_lower or 'qwen' in model_lower or 'mixtral' in model_lower or 'gemini' in model_lower:
        method = 'directprompt'
    elif 'timellm' in model_lower:
        method = 'timellm'
    elif 'unitime' in model_lower:
        method = 'unitime'
    elif 'chronos' in model_lower:
        method = 'chronos'
    elif 'moirai' in model_lower:
        method = 'moirai'
    elif 'lag-llama' in model_lower:
        method = 'lag_llama'
    elif 'arima' in model_lower:
        method = 'r_arima'
    elif 'ets' in model_lower:
        method = 'r_ets'
    elif 'exp-smoothing' in model_lower:
        method = 'statsmodels'
    else:
        method = 'unknown'

    if '-context' in model_name and '-nocontext' not in model_name:
        use_context = True
    elif '-nocontext' in model_name or '_nocontext' in model_name:
        use_context = False
    else:
        use_context = None

    MODELS_CONFIG[model_name] = {
        'method': method,
        'use_context': use_context
    }

print(f"Built config for {len(MODELS_CONFIG)} models from config.yaml")

# ============================================================================
# ============================================================================

def strip_model_context_suffix(model_name: str) -> str:
    k = str(model_base)
    for prefix in ["llmp-", "openrouter-", "claude-sdk-"]:
        if k.startswith(prefix):
            k = k[len(prefix):]
            break

def plot_param_vs_crps(df: pd.DataFrame, models_config: dict, metric_col: str):