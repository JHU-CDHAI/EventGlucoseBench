"""

across all context levels (NoCtx, Profile, MediumEvent, DetailedEvent, NewMedium, NewDetail)
for a specific subgroup and event type.

Usage:

================================================================================
PREDEFINED MODEL GROUPS (copy-paste ready commands):
================================================================================

# ---- Exercise examples (just change -e Diet to -e Exercise) ----

# ---- Different subgroups (change -g option) ----

================================================================================
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import warnings
import os
import yaml
import re
import shutil

TOPIC_NAME = '5-democompare'

warnings.filterwarnings('ignore')
matplotlib.use('Agg')

CONTEXT_LEVELS = ['NoCtx', 'Profile', 'MediumEvent', 'DetailedEvent', 'NewMedium', 'NewDetail']

CONTEXT_COLORS = {
}

def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}

def setup_environment():
    return model_name.replace('-context', '').replace('-nocontext', '')

def get_model_variants(base_model: str, result_dir: Path) -> dict:
    variants = {'context': None, 'nocontext': None}

    context_path = result_dir / f"{base_model}-context"
    nocontext_path = result_dir / f"{base_model}-nocontext"

    if context_path.exists():
        variants['context'] = f"{base_model}-context"
    if nocontext_path.exists():
        variants['nocontext'] = f"{base_model}-nocontext"

    if base_model.endswith('-context'):
        if (result_dir / base_model).exists():
            variants['context'] = base_model
    elif base_model.endswith('-nocontext'):
        if (result_dir / base_model).exists():
            variants['nocontext'] = base_model

    direct_path = result_dir / base_model
    if direct_path.exists() and not variants['context'] and not variants['nocontext']:
        variants['nocontext'] = base_model  # Treat as nocontext

def parse_task_name(task: str) -> tuple:
    m = re.match(r'(?:EventCGMTask_)?(D\d+_Age\d+)_(Diet|Exercise)(?:_Ontime_\w+)?$', task)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(
    )

def build_task_name(subgroup: str, event: str, context_level: str) -> str:
    return result_dir / model / task / str(seed) / 'complete_data.pkl'

def load_prediction_data(pkl_path: Path) -> dict:
    pass

    Args:

    Args:
        base_model: Base model name (without -context/-nocontext)
        seed: Instance number (seed)
        subgroup: Subgroup name (e.g., 'D1_Age18')
        event: Event type ('Diet' or 'Exercise')

    Returns:

    Layout:
    - Rows: Models x Seeds (each model-seed combination is a row)
    - Columns: Context levels (NoCtx, Profile, MediumEvent, DetailedEvent, NewMedium, NewDetail)

    Args:
        models: List of model names (base names without -context/-nocontext)
        subgroup: Subgroup name (e.g., 'D1_Age18')
        event: Event type ('Diet' or 'Exercise')

    Layout (all 5 context levels shown):
        (0,0) NoCtx              (0,1) ProfileOnly          (0,2) Context-level legend
        (1,0) BasicEventInfo     (1,1) StandardEventInfo    (1,2) DetailedEventInfo

    Position (0,2) contains a text box explaining the different context levels

    Args:
        model:       Model name (base name, e.g. 'gpt-4o').
                     ('EventCGMTask_D1_Age18_Diet_Ontime_DetailedEvent')
                     or short form ('D1_Age18_Diet').
        output_filename: Optional custom output filename (without extension).

    Returns:
        (pdf_path, png_path) tuple.
Examples:

