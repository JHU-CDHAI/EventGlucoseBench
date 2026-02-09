"""
Data Source Statistics for EventGlucose Benchmark

Reads raw task data files from the EventGlucose @task directory and generates:
1. A summary statistics table (LaTeX) for the appendix
2. Per-file breakdown with patient counts, instance counts, and event counts
3. Data provenance information for reviewer response


Config section: 6-data-source-statistics (in 1-config.yaml)
"""

import sys
import os
import pickle
import json
import warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

TOPIC_NAME = '6-data-source-statistics'

current_dir = Path.cwd()
if current_dir.name in ('notebooks', 'scripts') and current_dir.parent.name == 'evaluation':
    repo_root = current_dir.parent.parent
    os.chdir(repo_root)
else:
    repo_root = current_dir

sys.path.insert(0, str(repo_root))

def load_config(config_path: Path) -> dict:
    cfg = {}
    try:
        import yaml
        if config_path and config_path.is_file():
            with config_path.open('r') as f:
                cfg = yaml.safe_load(f) or {}
    except Exception:
        pass
    return cfg

def load_script_section(config_path: Path, section_key: str) -> dict:
    if not config_path or not config_path.exists():
        return {}
    try:
        import yaml
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
            if not l.strip() or l.lstrip().startswith('#'):
                block.append(l)
                continue
            if not l.startswith(' ') and ':' in l and not l.strip().startswith('- '):
                break
            block.append(l)
        data = yaml.safe_load('\n'.join(block))
        if isinstance(data, dict):
            return data.get(section_key, {}) or {}
        return {}
    except Exception:
        return {}

cfg_path = repo_root / '1-config.yaml'
cfg = load_config(cfg_path) if cfg_path.exists() else {}
script_cfg = load_script_section(cfg_path, TOPIC_NAME) if cfg_path.exists() else {}

paper_root = cfg.get('paper_project_root')
if paper_root:
    new_root = Path(paper_root).expanduser().resolve()
    if new_root.exists() and new_root.is_dir():
        if Path.cwd().resolve() != new_root:
            os.chdir(new_root)
            print(f"Changed working directory to: {new_root}")
        repo_root = new_root

data_dir_str = script_cfg.get('task_data_dir') or cfg.get('task_data_dir', '')
if not data_dir_str:
    candidates = [
        Path.home() / 'Desktop/EventGlucose/_WorkSpace/Data/EventGlucose/@task',
        repo_root.parent.parent / '_WorkSpace/Data/EventGlucose/@task',
    ]
    data_dir = next((p for p in candidates if p.exists()), None)
    if data_dir is None:
        print("ERROR: Cannot find task data directory. Set task_data_dir in 1-config.yaml")
        sys.exit(1)
else:
    data_dir = Path(data_dir_str)

print(f"Task data directory: {data_dir}")

results_root = Path(cfg.get('evaluation_results_folder', 'evaluation/results'))
if not results_root.is_absolute():
    results_root = (repo_root / results_root).resolve()

output_dir = results_root / TOPIC_NAME
output_dir.mkdir(parents=True, exist_ok=True)

appendix_table_path = repo_root / cfg.get('AppendixTablePath', '0-display/AppendixTable')
appendix_table_path.mkdir(parents=True, exist_ok=True)

files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pkl')])
print(f"\nFound {len(files)} task data files")

file_stats = []
all_patients = set()
event_type_counts = Counter()
all_glucose_values = []
year_range = [9999, 0]

for fname in files:
    filepath = data_dir / fname
    with open(filepath, 'rb') as f:
        df = pickle.load(f)

    parts = fname.replace('.pkl', '').split('-')

    if 'Exercise5Min' in fname:
        event_type = 'Exercise'
    elif 'Diet5Min-Med5Min' in fname:
        event_type = 'Diet+Medication'
    elif 'Diet5Min' in fname:
        event_type = 'Diet'
    else:
        event_type = 'Other'

    disease_type = df['DiseaseType'].iloc[0]
    age_group = df['AgeGroup'].iloc[0]

    patients_in_file = set()
    n_events_by_type = Counter()

    for _, row in df.iterrows():
        pd_ = row['profile_dict']
        patient_key = (
            pd_.get('MRSegmentID'),
            pd_.get('YearOfBirth'),
            pd_.get('Gender'),
            pd_.get('DiseaseType')
        )
        patients_in_file.add(patient_key)
        all_patients.add(patient_key)

        pt = str(row.get('prediction_time', ''))
        if pt and len(pt) >= 4:
            try:
                yr = int(pt[:4])
                year_range[0] = min(year_range[0], yr)
                year_range[1] = max(year_range[1], yr)
            except ValueError:
                pass

        ev = row['events']
        if isinstance(ev, dict) and 'event_info' in ev:
            for ei in ev['event_info']:
                if isinstance(ei, str):
                    try:
                        einfo = json.loads(ei)
                        etype = einfo.get('event_type', 'unknown')
                        n_events_by_type[etype] += 1
                        event_type_counts[etype] += 1
                    except (json.JSONDecodeError, TypeError):
                        pass

        seq = row['target_sequence']
        all_glucose_values.extend(seq)

    n_total_events = sum(n_events_by_type.values())

    file_stats.append({
        'filename': fname,
        'event_type': event_type,
        'disease_type': f"T{int(float(disease_type))}D",
        'age_group': age_group,
        'n_instances': len(df),
        'n_patients': len(patients_in_file),
        'n_events': n_total_events,
        'n_diet_events': n_events_by_type.get('Diet5Min', 0),
        'n_med_events': n_events_by_type.get('Med5Min', 0),
        'n_exercise_events': n_events_by_type.get('Exercise5Min', 0),
        'window_size': df['window_size'].iloc[0],
        'time_step_min': df['time_step_minutes'].iloc[0],
    })

stats_df = pd.DataFrame(file_stats)

all_glucose = np.array(all_glucose_values)
total_instances = stats_df['n_instances'].sum()
total_events = sum(event_type_counts.values())

print("\n" + "=" * 70)
print("EVENTGLUCOSE DATA SOURCE SUMMARY")
print("=" * 70)

print(f"\n--- Data Provenance ---")
print(f"Source platform:        Welldoc digital health platform")
print(f"CGM devices:            Dexcom, Abbott FreeStyle Libre")
print(f"Event annotation:       Patient self-reported via mobile app")
print(f"Geography:              United States")
print(f"IRB approval:           IRB00447704")
print(f"Data collection period: {year_range[0]}--{year_range[1]}")

print(f"\n--- Scale ---")
print(f"Task data files:        {len(files)}")
print(f"Unique patients:        {len(all_patients)}")
print(f"Total instances:        {total_instances:,}")
print(f"Total CGM readings:     {len(all_glucose):,}")
print(f"Total events:           {total_events:,}")

print(f"\n--- CGM Statistics ---")
print(f"Sampling resolution:    {stats_df['time_step_min'].iloc[0]:.0f} minutes")
print(f"Window size:            {stats_df['window_size'].iloc[0]} timesteps "
      f"({stats_df['window_size'].iloc[0] * 5 / 60:.1f} hours)")
print(f"Glucose range:          {all_glucose.min():.0f}--{all_glucose.max():.0f} mg/dL")
print(f"Mean glucose:           {all_glucose.mean():.1f} mg/dL")
print(f"Std glucose:            {all_glucose.std():.1f} mg/dL")

print(f"\n--- Event Breakdown ---")
for etype, count in event_type_counts.most_common():
    pct = count / total_events * 100
    print(f"  {etype:20s}: {count:6d} ({pct:.1f}%)")

print(f"\n--- Patients by Subgroup ---")
subgroup_patients = defaultdict(set)
for fname in files:
    filepath = data_dir / fname
    with open(filepath, 'rb') as f:
        df = pickle.load(f)
    for _, row in df.iterrows():
        pd_ = row['profile_dict']
        pk = (pd_.get('MRSegmentID'), pd_.get('YearOfBirth'),
              pd_.get('Gender'), pd_.get('DiseaseType'))
        dt = f"T{int(float(row['DiseaseType']))}D"
        sg = (dt, row['AgeGroup'])
        subgroup_patients[sg].add(pk)

for sg in sorted(subgroup_patients.keys()):
    print(f"  {sg[0]:4s} {sg[1]:6s}: {len(subgroup_patients[sg]):3d} patients")

gender_map = {1: 'Male', 2: 'Female'}
gender_counts = Counter()
for pk in all_patients:
    g = gender_map.get(pk[2], f'Unknown({pk[2]})')
    gender_counts[g] += 1
print(f"\n--- Patients by Gender ---")
for g, c in gender_counts.most_common():
    print(f"  {g:10s}: {c:3d} patients")

subgroup_rows = []
for (dt, ag), patients in sorted(subgroup_patients.items()):
    mask = (stats_df['disease_type'] == dt) & (stats_df['age_group'] == ag)
    sub = stats_df[mask]
    subgroup_rows.append({
        'Disease Type': dt,
        'Age Group': ag,
        'Patients': len(patients),
        'Instances': sub['n_instances'].sum(),
        'Diet Events': sub['n_diet_events'].sum(),
        'Med Events': sub['n_med_events'].sum(),
        'Exercise Events': sub['n_exercise_events'].sum(),
        'Total Events': sub['n_events'].sum(),
    })

sg_df = pd.DataFrame(subgroup_rows)

latex_lines = []
latex_lines.append(r'\begin{table}[t]')
latex_lines.append(r'\centering')
latex_lines.append(r'\caption{EventGlucose dataset statistics by demographic subgroup. '
                   r'Data sourced from the Welldoc digital health platform with CGM readings '
                   r'from Dexcom and Abbott FreeStyle Libre devices. Events are self-reported '
                   r'by patients via mobile app. IRB approval: IRB00447704.}')
latex_lines.append(r'\label{tab:data_source_summary}')
latex_lines.append(r'\small')
latex_lines.append(r'\setlength{\tabcolsep}{3pt}')
latex_lines.append(r'\begin{tabularx}{\columnwidth}{@{}llrrrrr@{}}')
latex_lines.append(r'\toprule')
latex_lines.append(r'\textbf{Type} & \textbf{Age} & \textbf{Patients} & '
                   r'\textbf{Instances} & \textbf{Diet} & \textbf{Med} & \textbf{Exercise} \\')
latex_lines.append(r'\midrule')

for _, row in sg_df.iterrows():
    latex_lines.append(
        f"{row['Disease Type']} & {row['Age Group']} & "
        f"{row['Patients']} & {row['Instances']:,} & "
        f"{row['Diet Events']:,} & {row['Med Events']:,} & "
        f"{row['Exercise Events']:,} \\\\"
    )

latex_lines.append(r'\midrule')
latex_lines.append(
    f"\\textbf{{Total}} & & \\textbf{{{len(all_patients)}}} & "
    f"\\textbf{{{total_instances:,}}} & "
    f"\\textbf{{{event_type_counts.get('Diet5Min', 0):,}}} & "
    f"\\textbf{{{event_type_counts.get('Med5Min', 0):,}}} & "
    f"\\textbf{{{event_type_counts.get('Exercise5Min', 0):,}}} \\\\"
)

latex_lines.append(r'\bottomrule')
latex_lines.append(r'\end{tabularx}')
latex_lines.append(r'\end{table}')

latex_content = '\n'.join(latex_lines)

latex_output_path = appendix_table_path / 'data_source_summary.tex'
with open(latex_output_path, 'w') as f:
    f.write(latex_content)
print(f"\nLaTeX table written to: {latex_output_path}")

detail_lines = []
detail_lines.append(r'\begin{table*}[t]')
detail_lines.append(r'\centering')
detail_lines.append(r'\caption{Per-file statistics for EventGlucose task data. '
                    r'Each file contains pre-filtered prediction instances for a specific '
                    r'event type and demographic subgroup.}')
detail_lines.append(r'\label{tab:data_source_detail}')
detail_lines.append(r'\small')
detail_lines.append(r'\begin{tabularx}{\textwidth}{@{}lllrrrrrr@{}}')
detail_lines.append(r'\toprule')
detail_lines.append(r'\textbf{Event Type} & \textbf{Disease} & \textbf{Age} & '
                    r'\textbf{Patients} & \textbf{Instances} & '
                    r'\textbf{Diet} & \textbf{Med} & \textbf{Exercise} & '
                    r'\textbf{Total Events} \\')
detail_lines.append(r'\midrule')

for _, row in stats_df.iterrows():
    detail_lines.append(
        f"{row['event_type']} & {row['disease_type']} & {row['age_group']} & "
        f"{row['n_patients']} & {row['n_instances']:,} & "
        f"{row['n_diet_events']:,} & {row['n_med_events']:,} & "
        f"{row['n_exercise_events']:,} & {row['n_events']:,} \\\\"
    )

detail_lines.append(r'\midrule')
detail_lines.append(
    f"\\textbf{{Total}} & & & \\textbf{{{len(all_patients)}}} & "
    f"\\textbf{{{total_instances:,}}} & "
    f"\\textbf{{{event_type_counts.get('Diet5Min', 0):,}}} & "
    f"\\textbf{{{event_type_counts.get('Med5Min', 0):,}}} & "
    f"\\textbf{{{event_type_counts.get('Exercise5Min', 0):,}}} & "
    f"\\textbf{{{total_events:,}}} \\\\"
)

detail_lines.append(r'\bottomrule')
detail_lines.append(r'\end{tabularx}')
detail_lines.append(r'\end{table*}')

detail_latex = '\n'.join(detail_lines)

detail_output_path = appendix_table_path / 'data_source_detail.tex'
with open(detail_output_path, 'w') as f:
    f.write(detail_latex)
print(f"Detail table written to: {detail_output_path}")

csv_path = output_dir / 'data_source_statistics.csv'
stats_df.to_csv(csv_path, index=False)
print(f"CSV saved to: {csv_path}")

summary_data = {
    'metric': [
        'source_platform', 'cgm_devices', 'event_annotation_method',
        'geography', 'irb_approval', 'data_collection_period',
        'n_task_files', 'n_unique_patients', 'n_total_instances',
        'n_total_cgm_readings', 'n_total_events',
        'n_diet_events', 'n_med_events', 'n_exercise_events',
        'cgm_resolution_min', 'window_size_timesteps',
        'glucose_min', 'glucose_max', 'glucose_mean', 'glucose_std',
    ],
    'value': [
        'Welldoc digital health platform',
        'Dexcom and Abbott FreeStyle Libre',
        'Patient self-reported via mobile app',
        'United States',
        'IRB00447704',
        f'{year_range[0]}-{year_range[1]}',
        len(files),
        len(all_patients),
        total_instances,
        len(all_glucose),
        total_events,
        event_type_counts.get('Diet5Min', 0),
        event_type_counts.get('Med5Min', 0),
        event_type_counts.get('Exercise5Min', 0),
        stats_df['time_step_min'].iloc[0],
        stats_df['window_size'].iloc[0],
        f'{all_glucose.min():.0f}',
        f'{all_glucose.max():.0f}',
        f'{all_glucose.mean():.1f}',
        f'{all_glucose.std():.1f}',
    ]
}
summary_csv_path = output_dir / 'data_source_summary.csv'
pd.DataFrame(summary_data).to_csv(summary_csv_path, index=False)
print(f"Summary CSV saved to: {summary_csv_path}")

print("\n" + "=" * 70)
print("DONE - Data source statistics generated successfully")
print("=" * 70)
