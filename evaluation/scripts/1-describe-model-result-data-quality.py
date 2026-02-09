"""
Model × Task Heatmaps (Merged View Only)

Generates two concise figures using YAML as the single source of truth:
  0-model_task_heatmap_merged_full: All included models (unfiltered)
  1-model_task_heatmap_merged:      Excluded models removed (filtered)
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
import shutil

TOPIC_NAME = '1-describe-model-result-data-quality'

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
grouped_include = section.get('include_models') if isinstance(section, dict) else None

ordered_model_variants_all = []  # All models (include + exclude) for version 1
ordered_model_variants_filtered = []  # Only included models for version 2
group_of_variant = {}

group_labels_map = {}
group_colors = {}
group_order_map = {}

if isinstance(grouped_include, dict):
    for gk, group_config in grouped_include.items():
        if isinstance(group_config, dict):
            display_name = group_config.get('display_name', gk)
            color = group_config.get('color', '#7f7f7f')  # Default gray
            order = group_config.get('order', 999)

            group_labels_map[gk] = display_name
            group_colors[display_name] = color
            group_order_map[display_name] = order

GROUP_ORDER = sorted(group_labels_map.values(), key=lambda x: group_order_map.get(x, 999))
GROUP_ORDER.append('Other')
group_colors['Other'] = '#d62728'  # Red for 'Other'

if isinstance(grouped_include, dict):
    for gk, group_config in grouped_include.items():
        group_label = group_labels_map.get(gk, gk)

        if isinstance(group_config, dict):
            include_list = group_config.get('include_models', [])
            exclude_list = group_config.get('exclude_models', [])

            for m in include_list:
                ordered_model_variants_all.append(m)
                group_of_variant[m] = group_label

            for m in exclude_list:
                ordered_model_variants_all.append(m)
                group_of_variant[m] = group_label

            for m in include_list:
                ordered_model_variants_filtered.append(m)

        elif isinstance(group_config, list):
            for m in group_config:
                ordered_model_variants_all.append(m)
                ordered_model_variants_filtered.append(m)
                group_of_variant[m] = group_label

ordered_model_variants = ordered_model_variants_filtered
allowed_models_set = set(ordered_model_variants)

def base_model_name(model: str) -> str:
    base_name = model.replace('-context', '').replace('-nocontext', '')

    if isinstance(config, dict) and 'model_display_names' in config:
        display_names = config['model_display_names']
        if base_name in display_names:
            return display_names[base_name]

    m = base_name
    if m.startswith('openrouter-'):
        m = m[len('openrouter-'):]
    m = m.replace('-sdk-', '-')
    while '--' in m:
        m = m.replace('--', '-')
    return m

preferred_bases_all = []  # All included models (version 1)
preferred_bases_filtered = []  # Filtered models (version 2)
base_group = {}

for v in ordered_model_variants_all:
    b = base_model_name(v)
    if b not in preferred_bases_all:
        preferred_bases_all.append(b)
    base_group[b] = group_of_variant.get(v, 'Other')

for v in ordered_model_variants_filtered:
    b = base_model_name(v)
    if b not in preferred_bases_filtered:
        preferred_bases_filtered.append(b)

preferred_bases = preferred_bases_filtered

data_path = results_root / '0-convert-Result-to-model-task-instance-score' / 'clinical_metrics_detailed.csv'
output_dir = results_root / TOPIC_NAME
if isinstance(section, dict) and section.get('output_dir'):
    out = Path(section['output_dir'])
    output_dir = out if out.is_absolute() else (repo_root / out)
output_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(repo_root / 'evaluation' / 'scripts'))
from shared_config import filter_display_tasks, get_task_display_name, categorize_task

print(f"Heatmap generation → data={data_path.name}, output={output_dir}")

def save_figure(fig, filename):
    fig.savefig(output_dir / f"{filename}.png", dpi=200, bbox_inches='tight')
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches='tight')
    plt.close(fig)

df_all = pd.read_csv(data_path)

exclude_tasks_from_section = set(section.get('exclude_tasks', []) or [])
include_tasks = set(section.get('include_tasks', []) or [])

if include_tasks:
    df_all = df_all[df_all['task'].isin(include_tasks)].copy()

all_tasks = sorted(df_all['task'].unique(), key=categorize_task)

# =============================================================================
# =============================================================================
print("\n" + "="*80)
print("GENERATING HEATMAP 1: UNFILTERED (All models: include + exclude)")
print("="*80)
print(f"Total model variants: {len(ordered_model_variants_all)}")
print(f"Total tasks: {len(all_tasks)}")

df_unfiltered = df_all[df_all['model'].isin(set(ordered_model_variants_all))].copy()
df_unfiltered['base_model'] = df_unfiltered['model'].apply(base_model_name)
df_unfiltered['is_ctx'] = df_unfiltered['model'].str.contains('context') & ~df_unfiltered['model'].str.contains('nocontext')

counts_u = df_unfiltered.groupby(['base_model', 'task', 'is_ctx']).size().unstack(fill_value=0)
counts_u = counts_u.rename(columns={True: 'C', False: 'N'})
counts_u['max'] = counts_u[['C', 'N']].max(axis=1)

pivot_unf = counts_u['max'].unstack(fill_value=0)
annot_unf = pivot_unf.astype(int).astype(str)

row_order_unf = [b for b in preferred_bases_all if b in pivot_unf.index]
pivot_unf = pivot_unf.reindex(index=row_order_unf)
annot_unf = annot_unf.reindex(index=row_order_unf)

col_order_unf = [t for t in all_tasks if t in pivot_unf.columns]
pivot_unf = pivot_unf.reindex(columns=col_order_unf)
annot_unf = annot_unf.reindex(columns=col_order_unf)

fig, ax = plt.subplots(figsize=(20, 12))
if pivot_unf.size == 0:
    ax.text(0.5, 0.5, 'No data to display', ha='center', va='center')
    ax.axis('off')
else:
    sns.heatmap(pivot_unf, annot=annot_unf, fmt='', cmap='Blues',
                vmin=0, vmax=20,
                cbar_kws={'label': 'Instance Count'},
                linewidths=0.8, linecolor='lightgray',
                ax=ax, annot_kws={'fontsize': 9, 'fontweight': 'bold'})

ax.set_title('UNFILTERED: All Models (Include + Exclude) × All Tasks\n(Merged context and no-context variants per base model)',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Task', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('Base Model', fontsize=13, fontweight='bold', labelpad=10)

if pivot_unf.shape[1] > 0:
    ax.set_xticklabels([get_task_display_name(t) for t in pivot_unf.columns], rotation=90, ha='center', fontsize=9)
if pivot_unf.shape[0] > 0:
    ax.set_yticklabels(pivot_unf.index, rotation=0, fontsize=10, fontweight='bold')

row_groups_unf = [base_group.get(m, 'Other') for m in pivot_unf.index]
for i in range(1, len(row_groups_unf)):
    if row_groups_unf[i] != row_groups_unf[i-1]:
        ax.axhline(y=i, color='red', linewidth=3, linestyle='-', alpha=0.8)

labels_order = [g for g in GROUP_ORDER if g in row_groups_unf]
pos = {}
for g in labels_order:
    idxs = [i for i, v in enumerate(row_groups_unf) if v == g]
    if idxs:
        pos[g] = (min(idxs) + max(idxs)) / 2

for lab, p in pos.items():
    ax.text(len(pivot_unf.columns) + 1.0, p + 0.5,
            lab, fontsize=13, fontweight='bold', va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=group_colors.get(lab, 'lightgray'),
                     edgecolor='black', linewidth=1.5, alpha=0.9),
            color='white')

plt.tight_layout()
save_figure(fig, "0-model_task_heatmap_merged_full")
print(f"✓ Saved: {pivot_unf.shape[0]} base models × {pivot_unf.shape[1]} tasks")

# =============================================================================
# =============================================================================
print("\n" + "="*80)
print("GENERATING HEATMAP 2: FILTERED (Excluded models and tasks removed)")
print("="*80)
print(f"Total model variants (filtered): {len(ordered_model_variants_filtered)}")

df_filtered = df_all[df_all['model'].isin(set(ordered_model_variants_filtered))].copy()

if exclude_tasks_from_section:
    print(f"Excluding tasks: {list(exclude_tasks_from_section)}")
    df_filtered = df_filtered[~df_filtered['task'].isin(exclude_tasks_from_section)].copy()
    print(f"Total tasks (after task exclusion): {df_filtered['task'].nunique()}")

df_filtered = filter_display_tasks(df_filtered, task_column='task', verbose=True)

df_filtered['base_model'] = df_filtered['model'].apply(base_model_name)
df_filtered['is_ctx'] = df_filtered['model'].str.contains('context') & ~df_filtered['model'].str.contains('nocontext')

counts_f = df_filtered.groupby(['base_model', 'task', 'is_ctx']).size().unstack(fill_value=0)
counts_f = counts_f.rename(columns={True: 'C', False: 'N'})
counts_f['max'] = counts_f[['C', 'N']].max(axis=1)

pivot_f = counts_f['max'].unstack(fill_value=0)
annot_f = pivot_f.astype(int).astype(str)

row_order_f = [b for b in preferred_bases_filtered if b in pivot_f.index]
pivot_f = pivot_f.reindex(index=row_order_f)
annot_f = annot_f.reindex(index=row_order_f)

filtered_tasks = sorted(df_filtered['task'].unique(), key=categorize_task)
col_order_f = [t for t in filtered_tasks if t in pivot_f.columns]
pivot_f = pivot_f.reindex(columns=col_order_f)
annot_f = annot_f.reindex(columns=col_order_f)

fig, ax = plt.subplots(figsize=(20, 12))
sns.heatmap(pivot_f, annot=annot_f, fmt='', cmap='Blues',
            vmin=0, vmax=20,
            cbar_kws={'label': 'Instance Count'},
            linewidths=0.8, linecolor='lightgray',
            ax=ax, annot_kws={'fontsize': 9, 'fontweight': 'bold'})

ax.set_title('FILTERED: Selected Models × Selected Tasks\n(Merged context and no-context variants per base model)',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Task', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('Base Model', fontsize=13, fontweight='bold', labelpad=10)
ax.set_xticklabels([get_task_display_name(t) for t in pivot_f.columns], rotation=90, ha='center', fontsize=9)
ax.set_yticklabels(pivot_f.index, rotation=0, fontsize=10, fontweight='bold')

row_groups_f = [base_group.get(m, 'Other') for m in pivot_f.index]
for i in range(1, len(row_groups_f)):
    if row_groups_f[i] != row_groups_f[i-1]:
        ax.axhline(y=i, color='red', linewidth=3, linestyle='-', alpha=0.8)

labels_f = [g for g in GROUP_ORDER if g in row_groups_f]
pos_f = {}
for g in labels_f:
    idxs = [i for i, v in enumerate(row_groups_f) if v == g]
    if idxs:
        pos_f[g] = (min(idxs) + max(idxs)) / 2

for lab, p in pos_f.items():
    ax.text(len(pivot_f.columns) + 1.0, p + 0.5,
            lab, fontsize=13, fontweight='bold', va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=group_colors.get(lab, 'lightgray'),
                     edgecolor='black', linewidth=1.5, alpha=0.9),
            color='white')

plt.tight_layout()
save_figure(fig, "1-model_task_heatmap_merged")
print(f"✓ Saved: {pivot_f.shape[0]} base models × {pivot_f.shape[1]} tasks")

appendix_figure_path = repo_root / "0-display" / "AppendixFigure"
appendix_figure_path.mkdir(parents=True, exist_ok=True)
source_pdf = output_dir / "1-model_task_heatmap_merged.pdf"
dest_pdf = appendix_figure_path / "A1-task_completeness_heatmap.pdf"
if source_pdf.exists():
    print(f"✓ Copied to appendix: {dest_pdf}")

print("\n" + "="*80)
print("HEATMAP GENERATION COMPLETE")
print("="*80)
print(f"Output directory: {output_dir}")
print(f"  - 0-model_task_heatmap_merged_full.png/pdf (unfiltered)")
print(f"  - 1-model_task_heatmap_merged.png/pdf (filtered)")
