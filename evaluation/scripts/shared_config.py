"""
Shared configuration for evaluation scripts.

This module contains common settings used across evaluation scripts 4+.
Import this module to ensure consistent model exclusions, color schemes, and other settings.
"""

# =============================================================================
# MODEL EXCLUSIONS
# =============================================================================

# Models to exclude from analysis (models not in final paper)
EXCLUDE_MODELS = [
    # Baselines
    'random',
    'oracle',
    # Chronos non-large variants (keep only chronos-large)
    'chronos-base',
    'chronos-mini',
    'chronos-small',
    'chronos-tiny',
    # Moirai non-large variants (keep only moirai-large)
    'moirai-base',
    'moirai-small',
    # Mixtral models (removed from final model set)
    'openrouter-mixtral-8x7b-instruct-context',
    'openrouter-mixtral-8x7b-instruct-nocontext',
    # Llama non-instruct variants (keep only instruct versions)
    'llmp-llama-3-8B-context',
    'llmp-llama-3-8B-nocontext',
    'llmp-llama-3-70b-context',
    'llmp-llama-3-70b-nocontext',
    'openrouter-llama-3-8b-context',
    'openrouter-llama-3-8b-nocontext',
    # Qwen 2.5 models (removed, keep only Qwen 3)
    'llmp-qwen2.5-0.5B-Instruct-context',
    'llmp-qwen2.5-0.5B-Instruct-nocontext',
    'llmp-qwen2.5-7B-Instruct-context',
    'llmp-qwen2.5-7B-Instruct-nocontext',
    # Claude Haiku models (removed, keep only Sonnet)
    'claude-sdk-haiku-4.5-context',
    'claude-sdk-haiku-4.5-nocontext',
    'openrouter-claude-3.5-haiku-context',
    'openrouter-claude-3.5-haiku-nocontext',
    # Multimodal TS without -etth1- suffix
    'timellm-context-pred96',
    'unitime-context-pred96',
]

# Tasks to exclude from analysis
EXCLUDE_TASKS = [
    'EventCGMTask_Base',  # Baseline task without specific conditions
]

# =============================================================================
# COLOR CONFIGURATIONS
# =============================================================================

# 1. Model by class (Statistical, TS Foundation, Multimodal TS, Direct Prompt LLM, etc.)
COLORS_MODEL_CLASS = {
    'Statistical': '#9467bd',           # Purple
    'TS Foundation': '#d62728',         # Red
    'Multimodal TS': '#2ca02c',         # Green
    'Direct Prompt LLM': '#1f77b4',     # Blue
    'Direct Prompt LLM (Small)': '#aec7e8',  # Light blue
    'LLM Process': '#ff7f0e',           # Orange
    'Baseline': '#7f7f7f',              # Gray
    'Other': '#bcbd22',                 # Yellow-green
}

# 2. Model by Quantitative vs LLM-based
COLORS_QUANT_VS_LLM = {
    'Quantitative': '#d62728',          # Red - Statistical & TS Foundation models
    'LLM-based': '#1f77b4',             # Blue - All LLM-based models
    'Hybrid': '#2ca02c',                # Green - Multimodal TS (combines both)
}

# 3. Instance by patient subgroup (Disease + Age)
COLORS_SUBGROUP = {
    'D1_Age18': '#1f77b4',              # Blue
    'D1_Age40': '#2ca02c',              # Green
    'D1_Age65': '#d62728',              # Red
    'D2_Age18': '#9467bd',              # Purple
    'D2_Age40': '#ff7f0e',              # Orange
    'D2_Age65': '#8c564b',              # Brown
    'Base': '#7f7f7f',                  # Gray (for tasks without subgroup)
}

# 4. Instance by event type (Diet, Exercise)
COLORS_EVENT_TYPE = {
    'Diet': '#ff7f0e',                  # Orange
    'Exercise': '#2ca02c',              # Green
    'Base': '#7f7f7f',                  # Gray (for base task without event)
}

# 5. Model by context usage (Context vs No Context)
COLORS_CONTEXT = {
    'Context': '#2ca02c',               # Green - uses context
    'No Context': '#d62728',            # Red - no context
    'N/A': '#7f7f7f',                   # Gray - not applicable (e.g., Chronos)
}

# 6. Task by context level (NoCtx, Profile, BasicEventInfo, StandardEventInfo, DetailedEventInfo)
COLORS_TASK_CONTEXT = {
    'Base': '#7f7f7f',                  # Gray - base task without context
    'NoCtx': '#d62728',                 # Red - no context provided
    'Profile': '#ff7f0e',               # Orange - patient profile only
    'BasicEventInfo': '#2ca02c',        # Green - basic event info
    'StandardEventInfo': '#9467bd',     # Purple - timing + key metrics
    'DetailedEventInfo': '#e377c2',     # Pink - full details rounded
}

# Task context level ordering (for sorting)
TASK_CONTEXT_ORDER = {
    'Base': 0,
    'NoCtx': 1,
    'Profile': 2,
    'BasicEventInfo': 3,
    'StandardEventInfo': 4,
    'DetailedEventInfo': 5,
}

# =============================================================================
# MODEL CLASSIFICATION FUNCTIONS
# =============================================================================

def get_model_class(model_name):
    """
    Classify model into category for COLORS_MODEL_CLASS.

    Returns one of: 'Statistical', 'TS Foundation', 'Multimodal TS',
                    'Direct Prompt LLM', 'Direct Prompt LLM (Small)',
                    'LLM Process', 'Baseline', 'Other'
    """
    model_lower = model_name.lower()

    # Baselines
    if model_name in ['random', 'oracle']:
        return 'Baseline'

    # Statistical models
    if any(x in model_lower for x in ['arima', 'ets', 'exp-smoothing']):
        return 'Statistical'

    # TS Foundation models
    if any(x in model_lower for x in ['chronos', 'lag-llama', 'moirai']):
        return 'TS Foundation'

    # Multimodal TS models
    if any(x in model_lower for x in ['unitime', 'timellm']):
        return 'Multimodal TS'

    # LLM Process models
    if 'llmp' in model_lower:
        return 'LLM Process'

    # Direct Prompt LLM (Large)
    if any(x in model_lower for x in ['gpt-4o', 'gpt-5', 'claude', 'gemini', 'qwen3-235b']):
        return 'Direct Prompt LLM'
    if 'openrouter' in model_lower and 'llama-3-70b' in model_lower:
        return 'Direct Prompt LLM'

    # Direct Prompt LLM (Small)
    if 'openrouter' in model_lower and any(x in model_lower for x in ['llama-3-8b', 'mixtral']):
        return 'Direct Prompt LLM (Small)'

    return 'Other'


def get_quant_vs_llm(model_name):
    """
    Classify model as Quantitative, LLM-based, or Hybrid for COLORS_QUANT_VS_LLM.

    - Quantitative: Statistical, TS Foundation (Chronos, Moirai, Lag-Llama)
    - LLM-based: Direct Prompt LLM, LLM Process
    - Hybrid: Multimodal TS (UniTime, TimeLLM)
    """
    model_class = get_model_class(model_name)

    if model_class in ['Statistical', 'TS Foundation']:
        return 'Quantitative'
    elif model_class == 'Multimodal TS':
        return 'Hybrid'
    elif model_class in ['Direct Prompt LLM', 'Direct Prompt LLM (Small)', 'LLM Process']:
        return 'LLM-based'
    else:
        return 'Quantitative'  # Default for baselines/other


def get_context_status(model_name):
    """
    Determine if model uses context for COLORS_CONTEXT.

    Checks for -context or -nocontext anywhere in the name.
    Returns: 'Context', 'No Context', or 'N/A'
    """
    if '-nocontext' in model_name:
        return 'No Context'
    elif '-context' in model_name:
        return 'Context'
    else:
        return 'N/A'


def parse_instance_subgroup(task_name):
    """
    Extract patient subgroup from task name for COLORS_SUBGROUP.

    Args:
        task_name: Task name like 'EventCGMTask_D1_Age18_Diet_Ontime_DetailedEvent'

    Returns:
        Subgroup string like 'D1_Age18', or 'Base' if not found
    """
    parts = task_name.split('_')
    disease = None
    age = None

    for part in parts:
        if part in ['D1', 'D2']:
            disease = part
        elif part.startswith('Age'):
            age = part

    if disease and age:
        return f"{disease}_{age}"
    return 'Base'


def parse_instance_event(task_name):
    """
    Extract event type from task name for COLORS_EVENT_TYPE.

    Args:
        task_name: Task name like 'EventCGMTask_D1_Age18_Diet_Ontime_DetailedEvent'

    Returns:
        'Diet', 'Exercise', or 'Base'
    """
    parts = task_name.split('_')

    for part in parts:
        if part == 'Diet':
            return 'Diet'
        elif part == 'Exercise':
            return 'Exercise'

    return 'Base'


def parse_task_context(task_name):
    """
    Extract context level from task name for COLORS_TASK_CONTEXT.

    Context levels (in order of information richness):
    - NoCtx: No context provided to the model
    - Profile: Patient profile information only
    - BasicEventInfo: Basic event information (type, timing)
    - StandardEventInfo: Timing + key metrics
    - DetailedEventInfo: Full details (values rounded)

    Args:
        task_name: Task name like 'EventCGMTask_D1_Age18_Diet_Ontime_DetailedEventInfo'

    Returns:
        Context level string: 'NoCtx', 'Profile', 'BasicEventInfo',
                             'StandardEventInfo', 'DetailedEventInfo', or 'Base'
    """
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


def get_task_context_order(task_name):
    """
    Get the sorting order for a task based on its context level.

    Args:
        task_name: Task name or context level string

    Returns:
        Integer for sorting (lower = less context)
    """
    context = parse_task_context(task_name) if '_' in task_name else task_name
    return TASK_CONTEXT_ORDER.get(context, 99)


def categorize_task(task_name):
    """
    Categorize task for sorting purposes.

    Returns a tuple for sorting: (diabetes_type, age_group, event_type, context_level, task_name)

    This enables consistent task ordering across all evaluation scripts:
    - Primary: Disease type (D1 before D2)
    - Secondary: Age group (18 < 40 < 65)
    - Tertiary: Event type (Diet before Exercise)
    - Quaternary: Context level (NoCtx < Profile < MediumEvent < DetailedEvent < NewMedium < NewDetail)

    Args:
        task_name: Task name like 'EventCGMTask_D1_Age18_Diet_Ontime_DetailedEvent'

    Returns:
        Tuple for sorting: (diabetes, age, event, context, task_name)
    """
    # Extract diabetes type
    if 'D1' in task_name:
        diabetes = 1
    elif 'D2' in task_name:
        diabetes = 2
    else:
        diabetes = 9

    # Extract age group
    if 'Age18' in task_name:
        age = 18
    elif 'Age40' in task_name:
        age = 40
    elif 'Age65' in task_name:
        age = 65
    else:
        age = 99

    # Extract event type
    if 'Diet' in task_name:
        event = 1
    elif 'Exercise' in task_name:
        event = 2
    else:
        event = 9

    # Extract context level using the shared function
    context = get_task_context_order(task_name)

    return (diabetes, age, event, context, task_name)


def get_task_base_name(task_name):
    """
    Extract the base task name (without context level suffix).

    Args:
        task_name: Full task name like 'EventCGMTask_D1_Age18_Diet_Ontime_DetailedEvent'

    Returns:
        Base name like 'D1_Age18_Diet' (without prefix, 'Ontime', and context level)
    """
    # Remove prefix
    task_short = task_name.replace('EventCGMTask_', '')

    # Split and filter
    parts = task_short.split('_')

    # Remove context level (last component) and 'Ontime'
    context_levels = {'NoCtx', 'Profile', 'BasicEventInfo', 'StandardEventInfo', 'DetailedEventInfo', 'Base'}
    base_parts = [p for p in parts if p != 'Ontime' and p not in context_levels]

    return '_'.join(base_parts)


# =============================================================================
# COLOR HELPER FUNCTIONS
# =============================================================================

def get_model_color(model_name, color_scheme='class'):
    """
    Get color for a model based on specified scheme.

    Args:
        model_name: Name of the model
        color_scheme: One of 'class', 'quant_llm', or 'context'

    Returns:
        Color string (hex)
    """
    if color_scheme == 'class':
        return COLORS_MODEL_CLASS.get(get_model_class(model_name), '#7f7f7f')
    elif color_scheme == 'quant_llm':
        return COLORS_QUANT_VS_LLM.get(get_quant_vs_llm(model_name), '#7f7f7f')
    elif color_scheme == 'context':
        return COLORS_CONTEXT.get(get_context_status(model_name), '#7f7f7f')
    else:
        raise ValueError(f"Unknown color scheme: {color_scheme}")


def get_instance_color(task_name, color_scheme='subgroup'):
    """
    Get color for an instance/task based on specified scheme.

    Args:
        task_name: Name of the task
        color_scheme: One of 'subgroup', 'event', or 'context'

    Returns:
        Color string (hex)
    """
    if color_scheme == 'subgroup':
        return COLORS_SUBGROUP.get(parse_instance_subgroup(task_name), '#7f7f7f')
    elif color_scheme == 'event':
        return COLORS_EVENT_TYPE.get(parse_instance_event(task_name), '#7f7f7f')
    elif color_scheme == 'context':
        return COLORS_TASK_CONTEXT.get(parse_task_context(task_name), '#7f7f7f')
    else:
        raise ValueError(f"Unknown color scheme: {color_scheme}")


def filter_excluded_models(df, model_column='model', verbose=True):
    """
    Filter out excluded models from a dataframe.

    Args:
        df: DataFrame with model column
        model_column: Name of the column containing model names
        verbose: Whether to print filtering info

    Returns:
        Filtered DataFrame
    """
    original_count = len(df)
    filtered_df = df[~df[model_column].isin(EXCLUDE_MODELS)]

    if verbose:
        excluded_count = original_count - len(filtered_df)
        if excluded_count > 0:
            print(f"Excluded {excluded_count:,} rows from {len(EXCLUDE_MODELS)} models")
            print(f"After filtering: {len(filtered_df):,} rows")

    return filtered_df


def filter_excluded_tasks(df, task_column='task', verbose=True):
    """
    Filter out excluded tasks from a dataframe.

    Args:
        df: DataFrame with task column
        task_column: Name of the column containing task names
        verbose: Whether to print filtering info

    Returns:
        Filtered DataFrame
    """
    original_count = len(df)
    filtered_df = df[~df[task_column].isin(EXCLUDE_TASKS)]

    if verbose:
        excluded_count = original_count - len(filtered_df)
        if excluded_count > 0:
            print(f"Excluded {excluded_count:,} rows from {len(EXCLUDE_TASKS)} tasks")
            print(f"After filtering: {len(filtered_df):,} rows")

    return filtered_df


# =============================================================================
# MODEL DISPLAY NAMES
# =============================================================================

# Mapping from full model names to simple display names
# Fill in your preferred display names below
MODEL_DISPLAY_NAMES = {
    # TS Foundation models (kept)
    'chronos-large': 'Chronos (large)',
    'lag-llama': 'Lag-Llama',
    'moirai-large': 'Moirai (large)',

    # Statistical models
    'exp-smoothing': 'Exp-Smoothing',
    'r-arima': 'Arima',
    'r-ets': 'ETS',

    # Multimodal TS models
    'timellm-etth1-context-pred96': 'TimeLLM-ctx',
    'timellm-etth1-nocontext-pred96': 'TimeLLM-noctx',
    'unitime-etth1-context-pred96': 'UniTime-ctx',
    'unitime-etth1-nocontext-pred96': 'UniTime-noctx',

    # Claude models
    'claude-sdk-sonnet-4.5-context': 'Claude-Sonnet4.5-ctx',
    'claude-sdk-sonnet-4.5-nocontext': 'Claude-Sonnet4.5-noctx',
    'openrouter-claude-3.5-sonnet-context': 'Claude-Sonnet3.5-ctx',
    'openrouter-claude-3.5-sonnet-nocontext': 'Claude-Sonnet3.5-noctx',

    # GPT models
    'gpt-4o-context': 'GPT-4o-ctx',
    'gpt-4o-nocontext': 'GPT-4o-noctx',
    'gpt-4o-mini-context': 'GPT-4o-mini-ctx',
    'gpt-4o-mini-nocontext': 'GPT-4o-mini-noctx',
    'gpt-5-mini-context': 'GPT-5-mini-ctx',
    'gpt-5-mini-nocontext': 'GPT-5-mini-noctx',

    # Gemini models
    'openrouter-gemini-2.5-flash-context': 'Gemini-2.5-flash-ctx',
    'openrouter-gemini-2.5-flash-nocontext': 'Gemini-2.5-flash-noctx',

    # Qwen models
    'openrouter-qwen3-235b-a22b-instruct-context': 'Qwen3-235b-Instr-ctx',
    'openrouter-qwen3-235b-a22b-instruct-nocontext': 'Qwen3-235b-Instr-noctx',

    # LLM Process models (Llama 70B and 8B instruct only)
    'llmp-llama-3-70b-instruct-context': 'Llama-3-70B-Instr-ctx',
    'llmp-llama-3-70b-instruct-nocontext': 'Llama-3-70B-Instr-noctx',
    'llmp-llama-3-8b-instruct-context': 'Llama-3-8B-Instr-ctx',
    'llmp-llama-3-8b-instruct-nocontext': 'Llama-3-8B-Instr-noctx',
}


def get_model_display_name(model_name):
    """
    Get the simple display name for a model.

    Args:
        model_name: Full model name

    Returns:
        Simple display name, or original name if not found in mapping
    """
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


# =============================================================================
# TASK DISPLAY NAMES AND FILTERING
# =============================================================================

def get_task_display_name(task_name):
    """
    Simplified task name for paper display.

    Removes EventCGMTask_ prefix, removes Ontime, replaces D1/D2 with T1D/T2D.

    Args:
        task_name: Full task name like 'EventCGMTask_D1_Age18_Diet_Ontime_DetailedEventInfo'

    Returns:
        Display name like 'T1D_Age18_Diet_DetailedEventInfo'
    """
    name = get_task_base_name(task_name)
    name = name.replace('D1', 'T1D').replace('D2', 'T2D')
    context = parse_task_context(task_name)
    if context and context != 'Base':
        name = f"{name}_{context}" if name else context
    return name
