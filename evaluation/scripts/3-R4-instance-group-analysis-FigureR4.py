"""
Instance-Level Group Analysis (FigureR4)

This script analyzes instance-level prediction difficulty vs model disagreement,

Figures:

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
import json
import yaml
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).parent))
from shared_config import filter_display_tasks, CONTEXT_DISPLAY_NAMES

TOPIC_NAME = '3-R4-instance-group-analysis-FigureR4'

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

data_path = results_root / '0-convert-Result-to-model-task-instance-score' / 'clinical_metrics_detailed.csv'
output_dir = results_root / TOPIC_NAME
if isinstance(section, dict) and section.get('output_dir'):
    out = Path(section['output_dir'])
    output_dir = out if out.is_absolute() else (repo_root / out)
output_dir.mkdir(parents=True, exist_ok=True)

script1_section = config.get('1-describe-model-result-data-quality', {}) if isinstance(config, dict) else {}
filtered_models = []
exclude_tasks_set = set()

if isinstance(script1_section, dict):
    grouped_include = script1_section.get('include_models', {})
    if isinstance(grouped_include, dict):
        for group_key, group_config in grouped_include.items():
            if isinstance(group_config, dict):
                include_list = group_config.get('include_models', [])
                filtered_models.extend(include_list)

    exclude_tasks_set = set(script1_section.get('exclude_tasks', []) or [])

fig_size = section.get('figure_size', [9.5, 9]) if isinstance(section, dict) else [9.5, 9]
font_sizes = section.get('font_sizes', {}) if isinstance(section, dict) else {}
font_title = font_sizes.get('title', 18)
font_axis = font_sizes.get('axis_label', 32)
font_tick = font_sizes.get('tick_label', 32)
font_legend = font_sizes.get('legend', 20)

kde_configs_from_yaml = section.get('kde_configs', {}) if isinstance(section, dict) else {}

MAIN_METRIC = config.get('MainMetrics', 'CRPS') if isinstance(config, dict) else 'CRPS'

plt.rcParams['figure.figsize'] = tuple(fig_size)
plt.rcParams['font.size'] = font_tick

print("=" * 80)
print("FIGURE R4: INSTANCE-LEVEL GROUP ANALYSIS")
print("=" * 80)
print(f"Output directory: {output_dir}")
print(f"Main metric: {MAIN_METRIC}")
print(f"Figure size: {fig_size}")
print(f"Font sizes: title={font_title}, axis={font_axis}, tick={font_tick}, legend={font_legend}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_quant_vs_llm(model_name):
    fig.savefig(output_dir / f"{filename}.png", dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches='tight')
    print(f"  Saved: {filename}.png and {filename}.pdf")
    plt.close(fig)

def create_main_metric_copy(file_pattern: str):
    df.to_csv(output_dir / f"{filename}.csv")
    print(f"  Saved: {filename}.csv")

def parse_task(task_name):

    Args:
        levels: Density levels for contours as fraction of peak density (default [0.1, 0.5]).
                - First level (0.1): Outer contour, drawn with DASHED line
                - Second level (0.5): Inner contour, drawn with SOLID line
                Lower values = looser/wider contours encompassing more data.
        alpha: Fill transparency for the inner region (default 0.3)
        bw_factor: Bandwidth multiplier. Higher = smoother/looser contours (default 1.5)

    Returns:

    Args:
        model_type: Model type filter ('llm', 'all', 'quant')
        metric: Metric to use for analysis ('crps', 't_crps', 'c_crps', 'ct_crps', 'rmse', 'mae', 'clarke_ab', 'clarke_cde')
    Create Figure R4a2: Instance difficulty vs model disagreement with 4 categories:

    Args: