"""
Generate Paper Table 3: Context Ablation by Event Type (Diet vs Exercise)

Unlike Table 2 which combines all events, this table shows:

        paper/0-display/Table/Table3-*.tex

"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import os
import sys
import shutil

warnings.filterwarnings('ignore')

TOPIC_NAME = '2-generate-Table3-diet-exercise'

current_dir = Path.cwd()
repo_root = current_dir
if current_dir.name == 'scripts' and current_dir.parent.name == 'evaluation':
    repo_root = current_dir.parent.parent
    os.chdir(repo_root)
    print(f"Changed working directory to: {repo_root}")

sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / 'evaluation' / 'scripts'))

from shared_config import (
    filter_excluded_tasks,
    filter_display_tasks,
    DISPLAY_EXCLUDE_CONTEXTS,
    CONTEXT_DISPLAY_NAMES,
    EXCLUDE_MODELS,
)

# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

def load_config(config_path: Path) -> dict:
    cfg = {}
    try:
        import yaml  # type: ignore
        if config_path and config_path.is_file():
            with config_path.open('r') as f:
                cfg = yaml.safe_load(f) or {}
    except Exception:
        pass
    try:
        if config_path and config_path.is_file():
            text = config_path.read_text()
            for line in text.splitlines():
                s = line.strip()
                if s.startswith('paper_project_root:'):
                    cfg['paper_project_root'] = line.split(':', 1)[1].strip().strip('"').strip("'")
                if s.startswith('evaluation_results_folder:'):
                    cfg['evaluation_results_folder'] = line.split(':', 1)[1].strip().strip('"').strip("'")
                if s.startswith('TablePath:'):
                    cfg['TablePath'] = line.split(':', 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass

def load_script_section(config_path: Path, section_key: str) -> dict:
    pass

    return model_name.replace('-context', '').replace('-nocontext', '')

def get_display_name(model_name):
    if pd.isna(mean) or pd.isna(sem):
    return f"{mean:.2f} $\\pm$ {sem:.2f}"

def format_value_with_delta(value, delta, add_color=True):
    if pd.isna(value) or pd.isna(delta):

    sign = "+" if delta > 0 else ""
    base_str = f"{value:.2f} ({sign}{delta:.2f})"

    if not add_color:

    if delta < -0.5:  # Improvement by more than 0.5
    elif delta > 5.0:  # Degradation by more than 5.0 (significant)
    else:

def bold_if_best(value_str, is_best):
    parts = task_name.replace('EventCGMTask_', '').split('_')

    result = {'diabetes': None, 'age': None, 'event': None, 'context': None}

    for part in parts:
        if part in ['D1', 'D2']:
            result['diabetes'] = part
        elif part.startswith('Age'):
            result['age'] = int(part.replace('Age', ''))
        elif part in ['Diet', 'Exercise']:
            result['event'] = part
        elif part in ['NoCtx', 'Profile', 'MediumEvent', 'DetailedEvent', 'NewMedium', 'NewDetail']:
            result['context'] = part

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_csv(data_path)
print(f"Loaded clinical metrics: {len(df):,} rows")

excluded_models = []
if isinstance(script_cfg, dict) and script_cfg.get('exclude_models'):
    excluded_models = script_cfg['exclude_models']
else:
    excluded_models = [
        'chronos-base',
        'chronos-small',
        'chronos-mini',
        'chronos-tiny',
        'moirai-base',
        'moirai-small'
    ]

if excluded_models:
    initial_count = len(df)
    df = df[~df['model'].isin(excluded_models)].copy()
    excluded_count = initial_count - len(df)
    if excluded_count > 0:
        print(f"Excluded {excluded_count:,} rows from {len(excluded_models)} models")
        print(f"  Excluded models: {', '.join(excluded_models)}")
    print(f"Remaining data: {len(df):,} rows")

print("\nApplying task filters...")
df = filter_excluded_tasks(df, task_column='task', verbose=True)
df = filter_display_tasks(df, task_column='task', verbose=True)

include_seeds = list(script_cfg.get('seeds', []) or []) if isinstance(script_cfg, dict) else []
if include_seeds and 'seed' in df.columns:
    before = len(df)
    df = df[df['seed'].isin(include_seeds)].copy()
    print(f"Applied seeds filter: {before - len(df)} rows removed (kept {len(df)})")

task_chars = df['task'].apply(parse_task_name).apply(pd.Series)
df = pd.concat([df, task_chars], axis=1)

print("\nTask characteristics:")
print(f"  Diabetes types: {sorted(df['diabetes'].dropna().unique())}")
print(f"  Ages: {sorted(df['age'].dropna().unique())}")
print(f"  Event types: {sorted(df['event'].dropna().unique())}")
print(f"  Context levels: {sorted(df['context'].dropna().unique())}")

# ============================================================================
# MULTI-METRIC TABLE GENERATION SETUP
# ============================================================================

AVAILABLE_METRICS = ['crps', 't_crps', 'c_crps', 'ct_crps', 'rmse', 'mae', 'clarke_ab', 'clarke_cde']
METRIC_DISPLAY_NAMES = {
    'crps': 'CRPS',
    't_crps': 'Temporal-CRPS',
    'c_crps': 'Clarke-CRPS',
    'ct_crps': 'Clarke-Temporal-CRPS',
    'rmse': 'RMSE',
    'mae': 'MAE',
    'clarke_ab': 'Clarke AB',
}

def create_main_metric_copy_table(file_pattern: str, output_dir, paper_dir):

    metric_display = METRIC_DISPLAY_NAMES.get(metric, metric.upper())
    print(f"\n{'='*80}")
    print(f"GENERATING TABLE 3 FOR METRIC: {metric_display}")
    print(f"{'='*80}")

    # ============================================================================
    # SELECT CONTEXT-AWARE MODELS (SAME AS TABLE 2)
    # ============================================================================

    print("\nSelecting context-aware models...")

    model_groups = {}
    for model in df['model'].unique():
        category = categorize_model(model)

        if category not in ['direct_prompt', 'multimodal']:
            continue

        base_name = get_base_model_name(model)

        if base_name not in model_groups:
            model_groups[base_name] = {'context': None, 'nocontext': None}

        if '-context' in model and 'nocontext' not in model:
            model_groups[base_name]['context'] = model
        elif 'nocontext' in model:
            model_groups[base_name]['nocontext'] = model

    table3_models = []
    for base_name, variants in model_groups.items():
        if variants['context'] and variants['nocontext']:
            if variants['context'] not in EXCLUDE_MODELS and variants['nocontext'] not in EXCLUDE_MODELS:
                table3_models.append(base_name)

    table3_models = sorted(table3_models)

    print(f"\nSelected {len(table3_models)} context-aware models for Table 3:")
    for base_name in table3_models:
        cat = categorize_model(base_name)
        print(f"  - {base_name} ({cat})")

    model_variant_map = {base: model_groups[base] for base in table3_models}

    # ============================================================================
    # COMPUTE STATISTICS BY EVENT TYPE AND CONTEXT LEVEL
    # ============================================================================

    print("\nComputing statistics by event type and context level...")

    all_context_levels = ['NoCtx', 'Profile', 'MediumEvent', 'DetailedEvent', 'NewMedium', 'NewDetail']
    context_levels = [c for c in all_context_levels if c not in DISPLAY_EXCLUDE_CONTEXTS]

    event_types = ['Diet', 'Exercise']

    results_t3 = []

    for base_name in table3_models:
        variants = model_variant_map[base_name]

        row = {
            'base_name': base_name,
            'category': categorize_model(base_name)
        }

        for event_type in event_types:
            df_event = df[df['event'] == event_type].copy()

            for ctx_level in context_levels:
                df_event_ctx = df_event[df_event['context'] == ctx_level].copy()

                if ctx_level == 'NoCtx':
                    model_to_use = variants['nocontext']
                else:
                    model_to_use = variants['context']

                df_model_event_ctx = df_event_ctx[df_event_ctx['model'] == model_to_use]

                if len(df_model_event_ctx) > 0:
                    row[f'{event_type}_{ctx_level}_mean'] = df_model_event_ctx[metric].mean()
                    row[f'{event_type}_{ctx_level}_sem'] = df_model_event_ctx[metric].sem()
                    row[f'{event_type}_{ctx_level}_count'] = len(df_model_event_ctx)
                else:
                    row[f'{event_type}_{ctx_level}_mean'] = np.nan
                    row[f'{event_type}_{ctx_level}_sem'] = np.nan
                    row[f'{event_type}_{ctx_level}_count'] = 0

        results_t3.append(row)

    results_df_t3 = pd.DataFrame(results_t3)

    print(f"Computed statistics for {len(table3_models)} models × {len(event_types)} events × {len(context_levels)} contexts")
    print(f"Total rows: {len(results_df_t3)} (one per model)")

    # ============================================================================
    # GENERATE LATEX TABLE 3 - HORIZONTAL LAYOUT
    # ============================================================================

    print("\nGenerating LaTeX table (horizontal layout: Diet | Exercise)...")

    results_df_t3 = results_df_t3.sort_values('base_name')

    richest_context = context_levels[-1]

    for event_type in event_types:
        for ctx in context_levels:
            if ctx == 'NoCtx':

            results_df_t3[f'{event_type}_{ctx}_delta'] = (
            )

    best_noctx = {}
    for event_type in event_types:
        best_noctx[f'{event_type}_NoCtx'] = results_df_t3[f'{event_type}_NoCtx_mean'].min()

    latex_lines_t3 = []

    context_descriptions = {
        'NoCtx': 'NoCtx (historical glucose only)',
        'Profile': 'ProfileOnly (patient demographics)',
        'MediumEvent': 'BasicEventInfo (demographics + event type)',
        'DetailedEvent': 'DetailedEvent (demographics + detailed event description)',
        'NewMedium': 'StandardEventInfo (timing + key metrics)',
        'NewDetail': 'DetailedEventInfo (full details, values rounded)',
    }
    context_desc_list = ', '.join(context_descriptions.get(c, c) for c in context_levels)

    col_count = 1 + 2 * len(context_levels)  # 1 for model name + contexts for each event type
    col_spec = 'l' + 'c' * (2 * len(context_levels))

    def make_two_row_header(context_name):