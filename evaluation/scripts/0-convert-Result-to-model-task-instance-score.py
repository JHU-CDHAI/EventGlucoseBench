"""
Convert Raw Results to Model-Task-Instance Level Scores

This notebook converts raw model results into a structured
CSV file with clinical metrics at the model-task-instance (seed) level.


"""

import sys
import subprocess
from pathlib import Path
import numpy as np
import warnings
import os

TOPIC_NAME = '0-convert-Result-to-model-task-instance-score'

warnings.filterwarnings('ignore')

current_dir = Path.cwd()
if current_dir.name == 'notebooks' and current_dir.parent.name == 'evaluation':
    repo_root = current_dir.parent.parent
    os.chdir(repo_root)
    print(f"Changed working directory from {current_dir} to {repo_root}")
elif current_dir.name == 'scripts' and current_dir.parent.name == 'evaluation':
    repo_root = current_dir.parent.parent
    os.chdir(repo_root)
    print(f"Changed working directory from {current_dir} to {repo_root}")
else:
    repo_root = current_dir

sys.path.insert(0, str(repo_root))

def load_config(config_path: Path) -> dict:
    cfg = {}
    try:
        import yaml  # type: ignore
        try:
            if config_path and config_path.is_file():
                with config_path.open('r') as f:
                    cfg = yaml.safe_load(f) or {}
                return cfg
        except Exception as e:
            print(f"Warning: Failed to load config via PyYAML from {config_path}: {e}")
    except Exception:
        pass

    try:
        if config_path and config_path.is_file():
            text = config_path.read_text()
            for line in text.splitlines():
                if line.strip().startswith('paper_project_root:'):
                    cfg['paper_project_root'] = line.split(':', 1)[1].strip().strip('"').strip("'")
                if line.strip().startswith('model_inference_results:'):
                    cfg['model_inference_results'] = line.split(':', 1)[1].strip().strip('"').strip("'")
                if line.strip().startswith('evaluation_results_folder:'):
                    cfg['evaluation_results_folder'] = line.split(':', 1)[1].strip().strip('"').strip("'")
                if line.strip().startswith('convert_to_notebooks_script:'):
                    cfg['convert_to_notebooks_script'] = line.split(':', 1)[1].strip().strip('"').strip("'")
                if line.strip().startswith('auto_convert_to_notebook:'):
                    val = line.split(':', 1)[1].strip().lower()
                    cfg['auto_convert_to_notebook'] = val in ('1', 'true', 'yes', 'on')
                if line.strip().startswith('evalfn_path:'):
                    cfg['evalfn_path'] = line.split(':', 1)[1].strip().strip('"').strip("'")
    except Exception as e:
        print(f"Warning: Failed to minimally parse config from {config_path}: {e}")
    return cfg

def load_script_section(config_path: Path, section_key: str) -> dict:
    if not config_path or not config_path.exists():
        return {}
    try:
        text = config_path.read_text()
        lines = text.splitlines()
        start = None
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                continue
            if line.startswith(section_key + ':'):
                start = i
                break
        if start is None:
            return {}
        block = [lines[start]]
        for j in range(start + 1, len(lines)):
            l = lines[j]
            if not l.strip():
                block.append(l)
                continue
            if l.lstrip().startswith('#'):
                block.append(l)
                continue
            if not l.startswith(' ') and ':' in l and not l.strip().startswith('- '):
                break
            block.append(l)
        block_text = '\n'.join(block)
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(block_text)
            if isinstance(data, dict):
                return data.get(section_key, {}) or {}
            return {}
        except Exception as e:
            print(f"Warning: Failed to parse script section via YAML: {e}")
            return {}
    except Exception as e:
        print(f"Warning: Failed to extract script section from {config_path}: {e}")
        return {}

def resolve_paths_from_config() -> tuple[Path, Path, dict, Path | None, Path]:
    default_cfg_candidates = [
        repo_root / '1-config.yaml',
    ]
    cfg_path = next((p for p in default_cfg_candidates if p.exists()), None)
    cfg = load_config(cfg_path) if cfg_path else {}

    result_dir_str = cfg.get('model_inference_results') or 'results'
    result_dir = Path(result_dir_str)

    results_root_str = cfg.get('evaluation_results_folder') or 'evaluation/results'
    results_root = Path(results_root_str)
    if not results_root.is_absolute():
        results_root = (repo_root / results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    output_dir = results_root / TOPIC_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    return result_dir, output_dir, cfg, cfg_path, results_root

result_dir, output_dir, config, config_path, results_root = resolve_paths_from_config()

paper_root = None
if isinstance(config, dict):
    paper_root = config.get('paper_project_root')
if paper_root:
    new_root = Path(paper_root).expanduser().resolve()
    if new_root.exists() and new_root.is_dir():
        if Path.cwd().resolve() != new_root:
            os.chdir(new_root)
            print(f"Changed working directory to paper_project_root: {new_root}")
        repo_root = new_root
        if not (isinstance(config.get(TOPIC_NAME, {}), dict) and config[TOPIC_NAME].get('output_dir')):
            cfg_results_str = (config.get('evaluation_results_folder')
                               if isinstance(config, dict) else None) or 'evaluation/results'
            rr = Path(cfg_results_str)
            if not rr.is_absolute():
                rr = (repo_root / rr).resolve()
            rr.mkdir(parents=True, exist_ok=True)
            output_dir = rr / TOPIC_NAME
            output_dir.mkdir(parents=True, exist_ok=True)

script_cfg = load_script_section(config_path, TOPIC_NAME) if config_path else {}
if not script_cfg and isinstance(config, dict):
    section = config.get(TOPIC_NAME)
    if isinstance(section, dict):
        script_cfg = section

if script_cfg.get('model_inference_results'):
    result_dir = Path(script_cfg['model_inference_results'])

if script_cfg.get('output_dir'):
    out = Path(script_cfg['output_dir'])
    output_dir = out if out.is_absolute() else (repo_root / out)
    output_dir.mkdir(parents=True, exist_ok=True)

include_seeds = list(script_cfg.get('seeds', []) or [])
low_mgdl = float(script_cfg.get('low_mgdl', 70.0))
high_mgdl = float(script_cfg.get('high_mgdl', 180.0))
do_glucose_rcrps = bool(script_cfg.get('compute_glucose_rcrps', True))

extra_paths = []
if isinstance(config, dict):
    extra_paths += config.get('extra_sys_paths', []) or []
if isinstance(script_cfg, dict):
    extra_paths += script_cfg.get('extra_sys_paths', []) or []
for p in extra_paths:
    pth = (repo_root / p).resolve() if not os.path.isabs(p) else Path(p).resolve()
    if pth.exists():
        sys.path.insert(0, str(pth))
        print(f"Added to sys.path: {pth}")

print("Environment setup complete")
print(f"Topic: {TOPIC_NAME}")
print(f"Working directory: {Path.cwd()}")
print(f"Config path: {str(config_path) if config_path else 'none'}")
print(f"Result directory: {result_dir}")
print(f"Output directory: {output_dir}")
if include_seeds:
    print(f"Limit to seeds (script cfg): {include_seeds}")
print(f"CRPS glucose range: low={low_mgdl}, high={high_mgdl}")

evalfn_path_cfg = None
if isinstance(script_cfg, dict) and script_cfg.get('evalfn_path'):
    evalfn_path_cfg = script_cfg.get('evalfn_path')
elif isinstance(config, dict) and config.get('evalfn_path'):
    evalfn_path_cfg = config.get('evalfn_path')

if not evalfn_path_cfg:
    raise ImportError("Missing evalfn_path in YAML. Set 'evalfn_path: evaluation/evalfn'.")

evalfn_path = (repo_root / evalfn_path_cfg).resolve()
evalfn_parent = evalfn_path.parent if evalfn_path.name == 'evalfn' else evalfn_path
if not evalfn_parent.exists():
    raise ImportError(f"evalfn parent directory does not exist: {evalfn_parent}")

sys.path.insert(0, str(evalfn_parent))
print(f"Added evalfn parent to sys.path: {evalfn_parent}")

from evalfn.clinical_metrics import collect_clinical_metrics  # type: ignore

print("=" * 80)
print("COLLECTING CLINICAL METRICS")
print("=" * 80)

if not result_dir.exists():
    print(f"Error: Result directory not found: {result_dir}")
    raise FileNotFoundError(f"Result directory not found: {result_dir}")

print(f"\nScanning results from: {result_dir}")
print("This may take several minutes...")

df = collect_clinical_metrics(result_dir, low_mgdl=low_mgdl, high_mgdl=high_mgdl)

if include_seeds:
    before = len(df)
    df = df[df['seed'].isin(include_seeds)].copy()
    print(f"Applied seeds filter: {before - len(df)} rows removed (kept {len(df)})")

if df.empty:
    print("ERROR: No results found!")
    raise ValueError("No data collected from Result directory")

print(f"\n✓ Collected {len(df)} instances across {df['model'].nunique()} models")

print("\n" + "=" * 80)
print("DATA SUMMARY")
print("=" * 80)

print(f"\nDataFrame shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
df.head()

print("\n" + "=" * 80)
print("DATA QUALITY CHECK")
print("=" * 80)

print(f"\nUnique models: {df['model'].nunique()}")
print(f"Unique tasks: {df['task'].nunique()}")
print(f"Unique seeds: {df['seed'].nunique()}")

print("\nMissing values per column:")
missing = df.isnull().sum()
for col in df.columns:
    count = missing[col]
    pct = 100 * count / len(df)
    if count > 0:
        print(f"  {col:20s}: {count:5d} ({pct:5.2f}%)")

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

output_file = output_dir / 'clinical_metrics_detailed.csv'
df.to_csv(output_file, index=False)
print(f"\n✓ Saved: {output_file}")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")

models_list = sorted(df['model'].unique().tolist())
tasks_list = sorted(df['task'].unique().tolist())

import json
lists_file = output_dir / 'models_and_tasks.json'
with open(lists_file, 'w') as f:
    json.dump({
        'models': models_list,
        'tasks': tasks_list,
        'n_models': len(models_list),
        'n_tasks': len(tasks_list),
        'n_instances': len(df)
    }, f, indent=2)
print(f"\n✓ Saved: {lists_file}")
print(f"  Models: {len(models_list)}")
print(f"  Tasks: {len(tasks_list)}")

models_txt = output_dir / 'models.txt'
with open(models_txt, 'w') as f:
    f.write('\n'.join(models_list))
print(f"\n✓ Saved: {models_txt}")

tasks_txt = output_dir / 'tasks.txt'
with open(tasks_txt, 'w') as f:
    f.write('\n'.join(tasks_list))
print(f"✓ Saved: {tasks_txt}")

print("\n" + "=" * 80)
print("TASK CONTEXT LEVEL SUMMARY")
print("=" * 80)

TASK_CONTEXT_LEVELS = ['Base', 'NoCtx', 'Profile', 'BasicEventInfo', 'StandardEventInfo', 'DetailedEventInfo']

def parse_task_context(task_name):
    if 'DetailedEventInfo' in task_name:
        return 'DetailedEventInfo'
    elif 'StandardEventInfo' in task_name:
        return 'StandardEventInfo'
    elif 'BasicEventInfo' in task_name:
        return 'BasicEventInfo'
    elif 'Profile' in task_name:
        return 'Profile'
    elif 'NoCtx' in task_name:
        return 'NoCtx'
    else:
        return 'Base'

df['task_context'] = df['task'].apply(parse_task_context)

context_task_counts = {}
context_instance_counts = df.groupby('task_context').size()

for ctx in TASK_CONTEXT_LEVELS:
    tasks_with_ctx = [t for t in tasks_list if parse_task_context(t) == ctx]
    context_task_counts[ctx] = len(tasks_with_ctx)

print("\nContext Level Distribution:")
print(f"{'Context Level':<20} {'Tasks':<10} {'Instances':<15}")
print("-" * 45)
for ctx in TASK_CONTEXT_LEVELS:
    n_tasks = context_task_counts.get(ctx, 0)
    n_instances = context_instance_counts.get(ctx, 0)
    if n_tasks > 0 or n_instances > 0:
        print(f"{ctx:<20} {n_tasks:<10} {n_instances:<15,}")

context_summary = {
    'context_levels': TASK_CONTEXT_LEVELS,
    'tasks_by_context': context_task_counts,
    'instances_by_context': {k: int(v) for k, v in context_instance_counts.items()},
}
context_file = output_dir / 'task_context_summary.json'
with open(context_file, 'w') as f:
    json.dump(context_summary, f, indent=2)
print(f"\n✓ Saved: {context_file}")

df = df.drop(columns=['task_context'])

if 'crps' in df.columns:
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    model_crps = df.groupby('model')['crps'].mean().sort_values()
    print("\nTop 10 Models by CRPS (lower is better):")
    for i, (model, crps) in enumerate(model_crps.head(10).items(), 1):
        print(f"  {i:2d}. {model:50s} - {crps:.4f}")

print("\n" + "=" * 80)
print("CONVERSION COMPLETE")
print("=" * 80)
print(f"\n✓ Processed {len(df):,} instances")
print(f"✓ Output: {output_file}")
print("=" * 80)

if __name__ == '__main__':
    print("\nScript execution completed!")