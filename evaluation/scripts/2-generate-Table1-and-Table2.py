"""
Generate Paper Tables (Table 1 + Table 2)

This script generates both main results tables for the EventGlucose paper:

Table 1: Main Results (Following CiK Format)

"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import os
import sys
import shutil

warnings.filterwarnings('ignore')

TOPIC_NAME = '2-generate-Table1-and-Table2'

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
    category = categorize_model(model_name)

def format_mean_sem(mean, sem):
    if is_best and value_str != "N/A":

def parse_task_name(task_name):
    
    main_metric_norm = MAIN_METRIC.lower().replace('-', '_')
    
    for ext in ['.tex', '.csv']:
        main_file = None
        for metric_variant in [MAIN_METRIC, main_metric_norm, 'crps']:
            candidate = output_dir / f"{file_pattern}_{metric_variant}{ext}"
            if candidate.exists():
                main_file = candidate
                break
        
        if main_file:
            main_copy = output_dir / f"{file_pattern}_Main{ext}"
            print(f"Created Main copy: {main_copy.name}")

            if ext == '.tex':
                if paper_dir and paper_dir.exists():
                    paper_main = paper_dir / main_copy.name
                    print(f"Copied Main to paper: {paper_main}")

print("\n" + "=" * 80)
print("GENERATING TABLES FOR ALL METRICS")
print("=" * 80)

def generate_tables_for_metric(df, metric):
    pass

        Strategy (following CiK):
        - LLMs: use -context variant (show performance WITH context)
        - TS Foundation: use base variant (they can't use context anyway)

        Returns: