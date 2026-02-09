"""
Run individual model and task combinations

This script provides easy access to run specific model-task combinations
using string names instead of JSON configuration files.

Usage examples:
    # List all available tasks
    python code/scripts/run_individual.py --list-tasks

    # List all available individual models
    python code/scripts/run_individual.py --list-models

    # List all available model types (groups)
    python code/scripts/run_individual.py --list-model-types

    # Run a specific combination
    python code/scripts/run_individual.py \
        --task EventCGMTask_Base \
        --model chronos-small \
        --n-instances 10 \
        --n-samples 25

    # Run task variants via aliases:
    #   *_context    -> expands to *_Profile + *_MediumEvent + *_DetailedEvent
    #   *_newcontext -> expands to *_NewMedium + *_NewDetail
    #   *_allcontext -> expands to all 5 context levels
    #   *_nocontext  -> expands to *_NoCtx
    python code/scripts/run_individual.py \
        --task EventCGMTask_D1_Age18_Diet_Ontime_context \
        --model gpt-4o-context chronos-small \
        --n-instances 10 --n-samples 25

    # Run new context levels only
    python code/scripts/run_individual.py \
        --task EventCGMTask_D1_Age18_Diet_Ontime_newcontext \
        --model gpt-4o-context \
        --n-instances 10 --n-samples 25

    # Run all context levels (original + new)
    python code/scripts/run_individual.py \
        --task EventCGMTask_D1_Age18_Diet_Ontime_allcontext \
        --model gpt-4o-context \
        --n-instances 10 --n-samples 25

    # Run with model types (auto-expands to all configs)
    python code/scripts/run_individual.py \
        --task EventCGMTask_Base \
        --model chronos \
        --n-instances 10 \
        --n-samples 25
    # This runs all 5 Chronos variants: tiny, mini, small, base, large

    # Mix individual models and model types
    python code/scripts/run_individual.py \
        --task EventCGMTask_Base EventCGMTask_Diet5Min \
        --model random chronos moirai \
        --n-instances 10 \
        --n-samples 25
    # This runs: random + all Chronos variants + all Moirai variants

    # Run comprehensive evaluation
    python code/scripts/run_individual.py \
        --task EventCGMTask_Base \
        --model foundation-all \
        --n-instances 20 \
        --n-samples 50
    # This runs all foundation models (Chronos, Moirai, Lag-Llama, TimeGEN)

    # Run all tasks on a single model
    python code/scripts/run_individual.py \
        --all-tasks \
        --model chronos-small \
        --n-instances 10 \
        --n-samples 5
"""

import argparse
import logging
import os
import sys
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Callable

# Configure matplotlib to not display figures (only save them)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - saves figures without displaying


# ============================================================================
# Auto-load environment from env.sh
# ============================================================================
def load_env_from_shell():
    """
    Automatically find and load environment variables from env.sh.

    This function:
    1. Searches for env.sh in the project root (up to 3 levels from this file)
    2. Sources it using bash to get all environment variables
    3. Updates os.environ with the loaded values

    This allows the script to work without manually running 'source env.sh' first.
    """
    # Find project root by looking for env.sh
    current_path = Path(__file__).resolve()

    # Try up to 3 levels up from the script location
    for _ in range(3):
        current_path = current_path.parent
        env_file = current_path / "env.sh"

        if env_file.exists():
            # Only log in main process (not in multiprocessing workers)
            # Check if we're in a subprocess by looking for common markers
            is_subprocess = os.environ.get('_EVENTGLUCOSE_SUBPROCESS') == '1'

            if not is_subprocess:
                logging.info(f"Found env.sh at: {env_file}")

            try:
                # Source env.sh and capture the resulting environment
                # We use a bash command that sources env.sh then prints all env vars
                cmd = f'source "{env_file}" && env'
                result = subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=str(current_path)
                )

                # Parse the environment variables from output
                for line in result.stdout.splitlines():
                    if '=' in line:
                        # Split only on first '=' to handle values with '=' in them
                        key, value = line.split('=', 1)
                        # Only set if not already in environment (allow user overrides)
                        if key not in os.environ:
                            os.environ[key] = value

                # Mark that we're in a subprocess for future imports
                os.environ['_EVENTGLUCOSE_SUBPROCESS'] = '1'

                if not is_subprocess:
                    logging.info(f"Successfully loaded environment from {env_file}")
                return True

            except subprocess.CalledProcessError as e:
                if not is_subprocess:
                    logging.warning(f"Failed to source env.sh: {e}")
                return False

    if not os.environ.get('_EVENTGLUCOSE_SUBPROCESS') == '1':
        logging.warning("Could not find env.sh in project root")
    return False


# Configure logging before loading environment
logging.basicConfig(level=logging.INFO)

# Load environment before importing eventglucose modules
load_env_from_shell()

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# IMPORTANT:
# Keep baseline imports LAZY (inside factories) so listing commands like
# `--list-models` don't import heavy deps (torch/transformers) and crash/slow down.


# ============================================================================
# TASK REGISTRY - Map string names to task classes
# ============================================================================

def get_task_registry() -> Dict[str, type]:
    """
    Return a dictionary mapping task names to task classes.

    Returns:
        Dict mapping task name strings to task class objects
    """
    # Base tasks (broadest / debugging-friendly)
    #
    # These exist in the codebase and are referenced in this script's docstring
    # examples, so we include them in the registry by default.
    from eventglucose.tasks.factory_task_Base import (
        EventCGMTask_Base,
        EventCGMTask_test,
    )

    from eventglucose.tasks.task_diet_ontime_profile import (
        EventCGMTask_D1_Age18_Diet_Ontime_Profile,
        EventCGMTask_D1_Age40_Diet_Ontime_Profile,
        EventCGMTask_D1_Age65_Diet_Ontime_Profile,
        EventCGMTask_D2_Age18_Diet_Ontime_Profile,
        EventCGMTask_D2_Age40_Diet_Ontime_Profile,
        EventCGMTask_D2_Age65_Diet_Ontime_Profile,
    )

    from eventglucose.tasks.task_diet_ontime_profile_mediumevent import (
        EventCGMTask_D1_Age18_Diet_Ontime_MediumEvent,
        EventCGMTask_D1_Age40_Diet_Ontime_MediumEvent,
        EventCGMTask_D1_Age65_Diet_Ontime_MediumEvent,
        EventCGMTask_D2_Age18_Diet_Ontime_MediumEvent,
        EventCGMTask_D2_Age40_Diet_Ontime_MediumEvent,
        EventCGMTask_D2_Age65_Diet_Ontime_MediumEvent,
    )

    from eventglucose.tasks.task_diet_ontime_profile_detailedevent import (
        EventCGMTask_D1_Age18_Diet_Ontime_DetailedEvent,
        EventCGMTask_D1_Age40_Diet_Ontime_DetailedEvent,
        EventCGMTask_D1_Age65_Diet_Ontime_DetailedEvent,
        EventCGMTask_D2_Age18_Diet_Ontime_DetailedEvent,
        EventCGMTask_D2_Age40_Diet_Ontime_DetailedEvent,
        EventCGMTask_D2_Age65_Diet_Ontime_DetailedEvent,
    )

    from eventglucose.tasks.task_diet_ontime_noctx import (
        EventCGMTask_D1_Age18_Diet_Ontime_NoCtx,
        EventCGMTask_D1_Age40_Diet_Ontime_NoCtx,
        EventCGMTask_D1_Age65_Diet_Ontime_NoCtx,
        EventCGMTask_D2_Age18_Diet_Ontime_NoCtx,
        EventCGMTask_D2_Age40_Diet_Ontime_NoCtx,
        EventCGMTask_D2_Age65_Diet_Ontime_NoCtx,
    )

    from eventglucose.tasks.task_exercise_ontime_profile import (
        EventCGMTask_D1_Age18_Exercise_Ontime_Profile,
        EventCGMTask_D1_Age40_Exercise_Ontime_Profile,
        EventCGMTask_D1_Age65_Exercise_Ontime_Profile,
        EventCGMTask_D2_Age18_Exercise_Ontime_Profile,
        EventCGMTask_D2_Age40_Exercise_Ontime_Profile,
        EventCGMTask_D2_Age65_Exercise_Ontime_Profile,
    )

    from eventglucose.tasks.task_exercise_ontime_profile_mediumevent import (
        EventCGMTask_D1_Age18_Exercise_Ontime_MediumEvent,
        EventCGMTask_D1_Age40_Exercise_Ontime_MediumEvent,
        EventCGMTask_D1_Age65_Exercise_Ontime_MediumEvent,
        EventCGMTask_D2_Age18_Exercise_Ontime_MediumEvent,
        EventCGMTask_D2_Age40_Exercise_Ontime_MediumEvent,
        EventCGMTask_D2_Age65_Exercise_Ontime_MediumEvent,
    )

    from eventglucose.tasks.task_exercise_ontime_profile_detailedevent import (
        EventCGMTask_D1_Age18_Exercise_Ontime_DetailedEvent,
        EventCGMTask_D1_Age40_Exercise_Ontime_DetailedEvent,
        EventCGMTask_D1_Age65_Exercise_Ontime_DetailedEvent,
        EventCGMTask_D2_Age18_Exercise_Ontime_DetailedEvent,
        EventCGMTask_D2_Age40_Exercise_Ontime_DetailedEvent,
        EventCGMTask_D2_Age65_Exercise_Ontime_DetailedEvent,
    )

    from eventglucose.tasks.task_exercise_ontime_noctx import (
        EventCGMTask_D1_Age18_Exercise_Ontime_NoCtx,
        EventCGMTask_D1_Age40_Exercise_Ontime_NoCtx,
        EventCGMTask_D1_Age65_Exercise_Ontime_NoCtx,
        EventCGMTask_D2_Age18_Exercise_Ontime_NoCtx,
        EventCGMTask_D2_Age40_Exercise_Ontime_NoCtx,
        EventCGMTask_D2_Age65_Exercise_Ontime_NoCtx,
    )

    # New Medium tasks (timing + key metrics like calories/carbs)
    from eventglucose.tasks.task_diet_ontime_profile_newmedium import (
        EventCGMTask_D1_Age18_Diet_Ontime_NewMedium,
        EventCGMTask_D1_Age40_Diet_Ontime_NewMedium,
        EventCGMTask_D1_Age65_Diet_Ontime_NewMedium,
        EventCGMTask_D2_Age18_Diet_Ontime_NewMedium,
        EventCGMTask_D2_Age40_Diet_Ontime_NewMedium,
        EventCGMTask_D2_Age65_Diet_Ontime_NewMedium,
    )

    from eventglucose.tasks.task_exercise_ontime_profile_newmedium import (
        EventCGMTask_D1_Age18_Exercise_Ontime_NewMedium,
        EventCGMTask_D1_Age40_Exercise_Ontime_NewMedium,
        EventCGMTask_D1_Age65_Exercise_Ontime_NewMedium,
        EventCGMTask_D2_Age18_Exercise_Ontime_NewMedium,
        EventCGMTask_D2_Age40_Exercise_Ontime_NewMedium,
        EventCGMTask_D2_Age65_Exercise_Ontime_NewMedium,
    )

    # New Detail tasks (full event details with rounded values)
    from eventglucose.tasks.task_diet_ontime_profile_newdetail import (
        EventCGMTask_D1_Age18_Diet_Ontime_NewDetail,
        EventCGMTask_D1_Age40_Diet_Ontime_NewDetail,
        EventCGMTask_D1_Age65_Diet_Ontime_NewDetail,
        EventCGMTask_D2_Age18_Diet_Ontime_NewDetail,
        EventCGMTask_D2_Age40_Diet_Ontime_NewDetail,
        EventCGMTask_D2_Age65_Diet_Ontime_NewDetail,
    )

    from eventglucose.tasks.task_exercise_ontime_profile_newdetail import (
        EventCGMTask_D1_Age18_Exercise_Ontime_NewDetail,
        EventCGMTask_D1_Age40_Exercise_Ontime_NewDetail,
        EventCGMTask_D1_Age65_Exercise_Ontime_NewDetail,
        EventCGMTask_D2_Age18_Exercise_Ontime_NewDetail,
        EventCGMTask_D2_Age40_Exercise_Ontime_NewDetail,
        EventCGMTask_D2_Age65_Exercise_Ontime_NewDetail,
    )

    # No-event tasks (exist in codebase but were not previously registered here)
    from eventglucose.tasks.eventglucose_tasks import EventCGMTask_withEvent_withLag
    from eventglucose.tasks.task_noevent_ontime_noctx import (
        EventCGMTask_NoEvent_Ontime_NoCtx,
        EventCGMTask_D1_Age18_NoEvent_Ontime_NoCtx,
        EventCGMTask_D1_Age40_NoEvent_Ontime_NoCtx,
        EventCGMTask_D1_Age65_NoEvent_Ontime_NoCtx,
        EventCGMTask_D2_Age18_NoEvent_Ontime_NoCtx,
        EventCGMTask_D2_Age40_NoEvent_Ontime_NoCtx,
        EventCGMTask_D2_Age65_NoEvent_Ontime_NoCtx,
    )

    # Build the registry
    task_classes = [
        # Base tasks
        EventCGMTask_Base,
        EventCGMTask_test,
        # EventCGMTask_Diet_TypeOne,
        # # Event-specific tasks
        # EventCGMTask_Diet5Min,
        # EventCGMTask_Diet5Min_Med5Min,
        # EventCGMTask_Exercise5Min,
        # # Subgroup tasks
        # EventCGMTask_D1_Age18,
        # EventCGMTask_D1_Age40,
        # EventCGMTask_D1_Age65,
        # EventCGMTask_D2_Age18,
        # EventCGMTask_D2_Age40,
        # EventCGMTask_D2_Age65,
        # Diet Profile tasks
        EventCGMTask_D1_Age18_Diet_Ontime_Profile,
        EventCGMTask_D1_Age40_Diet_Ontime_Profile,
        EventCGMTask_D1_Age65_Diet_Ontime_Profile,
        EventCGMTask_D2_Age18_Diet_Ontime_Profile,
        EventCGMTask_D2_Age40_Diet_Ontime_Profile,
        EventCGMTask_D2_Age65_Diet_Ontime_Profile,
        # Diet MediumEvent tasks
        EventCGMTask_D1_Age18_Diet_Ontime_MediumEvent,
        EventCGMTask_D1_Age40_Diet_Ontime_MediumEvent,
        EventCGMTask_D1_Age65_Diet_Ontime_MediumEvent,
        EventCGMTask_D2_Age18_Diet_Ontime_MediumEvent,
        EventCGMTask_D2_Age40_Diet_Ontime_MediumEvent,
        EventCGMTask_D2_Age65_Diet_Ontime_MediumEvent,
        # Diet DetailedEvent tasks
        EventCGMTask_D1_Age18_Diet_Ontime_DetailedEvent,
        EventCGMTask_D1_Age40_Diet_Ontime_DetailedEvent,
        EventCGMTask_D1_Age65_Diet_Ontime_DetailedEvent,
        EventCGMTask_D2_Age18_Diet_Ontime_DetailedEvent,
        EventCGMTask_D2_Age40_Diet_Ontime_DetailedEvent,
        EventCGMTask_D2_Age65_Diet_Ontime_DetailedEvent,
        # Diet NoCtx tasks
        EventCGMTask_D1_Age18_Diet_Ontime_NoCtx,
        EventCGMTask_D1_Age40_Diet_Ontime_NoCtx,
        EventCGMTask_D1_Age65_Diet_Ontime_NoCtx,
        EventCGMTask_D2_Age18_Diet_Ontime_NoCtx,
        EventCGMTask_D2_Age40_Diet_Ontime_NoCtx,
        EventCGMTask_D2_Age65_Diet_Ontime_NoCtx,
        # Exercise Profile tasks
        EventCGMTask_D1_Age18_Exercise_Ontime_Profile,
        EventCGMTask_D1_Age40_Exercise_Ontime_Profile,
        EventCGMTask_D1_Age65_Exercise_Ontime_Profile,
        EventCGMTask_D2_Age18_Exercise_Ontime_Profile,
        EventCGMTask_D2_Age40_Exercise_Ontime_Profile,
        EventCGMTask_D2_Age65_Exercise_Ontime_Profile,
        # Exercise MediumEvent tasks
        EventCGMTask_D1_Age18_Exercise_Ontime_MediumEvent,
        EventCGMTask_D1_Age40_Exercise_Ontime_MediumEvent,
        EventCGMTask_D1_Age65_Exercise_Ontime_MediumEvent,
        EventCGMTask_D2_Age18_Exercise_Ontime_MediumEvent,
        EventCGMTask_D2_Age40_Exercise_Ontime_MediumEvent,
        EventCGMTask_D2_Age65_Exercise_Ontime_MediumEvent,
        # Exercise DetailedEvent tasks
        EventCGMTask_D1_Age18_Exercise_Ontime_DetailedEvent,
        EventCGMTask_D1_Age40_Exercise_Ontime_DetailedEvent,
        EventCGMTask_D1_Age65_Exercise_Ontime_DetailedEvent,
        EventCGMTask_D2_Age18_Exercise_Ontime_DetailedEvent,
        EventCGMTask_D2_Age40_Exercise_Ontime_DetailedEvent,
        EventCGMTask_D2_Age65_Exercise_Ontime_DetailedEvent,
        # Exercise NoCtx tasks
        EventCGMTask_D1_Age18_Exercise_Ontime_NoCtx,
        EventCGMTask_D1_Age40_Exercise_Ontime_NoCtx,
        EventCGMTask_D1_Age65_Exercise_Ontime_NoCtx,
        EventCGMTask_D2_Age18_Exercise_Ontime_NoCtx,
        EventCGMTask_D2_Age40_Exercise_Ontime_NoCtx,
        EventCGMTask_D2_Age65_Exercise_Ontime_NoCtx,
        # Diet NewMedium tasks
        EventCGMTask_D1_Age18_Diet_Ontime_NewMedium,
        EventCGMTask_D1_Age40_Diet_Ontime_NewMedium,
        EventCGMTask_D1_Age65_Diet_Ontime_NewMedium,
        EventCGMTask_D2_Age18_Diet_Ontime_NewMedium,
        EventCGMTask_D2_Age40_Diet_Ontime_NewMedium,
        EventCGMTask_D2_Age65_Diet_Ontime_NewMedium,
        # Diet NewDetail tasks
        EventCGMTask_D1_Age18_Diet_Ontime_NewDetail,
        EventCGMTask_D1_Age40_Diet_Ontime_NewDetail,
        EventCGMTask_D1_Age65_Diet_Ontime_NewDetail,
        EventCGMTask_D2_Age18_Diet_Ontime_NewDetail,
        EventCGMTask_D2_Age40_Diet_Ontime_NewDetail,
        EventCGMTask_D2_Age65_Diet_Ontime_NewDetail,
        # Exercise NewMedium tasks
        EventCGMTask_D1_Age18_Exercise_Ontime_NewMedium,
        EventCGMTask_D1_Age40_Exercise_Ontime_NewMedium,
        EventCGMTask_D1_Age65_Exercise_Ontime_NewMedium,
        EventCGMTask_D2_Age18_Exercise_Ontime_NewMedium,
        EventCGMTask_D2_Age40_Exercise_Ontime_NewMedium,
        EventCGMTask_D2_Age65_Exercise_Ontime_NewMedium,
        # Exercise NewDetail tasks
        EventCGMTask_D1_Age18_Exercise_Ontime_NewDetail,
        EventCGMTask_D1_Age40_Exercise_Ontime_NewDetail,
        EventCGMTask_D1_Age65_Exercise_Ontime_NewDetail,
        EventCGMTask_D2_Age18_Exercise_Ontime_NewDetail,
        EventCGMTask_D2_Age40_Exercise_Ontime_NewDetail,
        EventCGMTask_D2_Age65_Exercise_Ontime_NewDetail,

        # No-event tasks
        EventCGMTask_withEvent_withLag,
        EventCGMTask_NoEvent_Ontime_NoCtx,
        EventCGMTask_D1_Age18_NoEvent_Ontime_NoCtx,
        EventCGMTask_D1_Age40_NoEvent_Ontime_NoCtx,
        EventCGMTask_D1_Age65_NoEvent_Ontime_NoCtx,
        EventCGMTask_D2_Age18_NoEvent_Ontime_NoCtx,
        EventCGMTask_D2_Age40_NoEvent_Ontime_NoCtx,
        EventCGMTask_D2_Age65_NoEvent_Ontime_NoCtx,
    ]

    return {cls.__name__: cls for cls in task_classes}


# ============================================================================
# MODEL TYPE REGISTRY - Map model types to all their configurations
# ============================================================================

def get_model_type_registry() -> Dict[str, List[str]]:
    """
    Return a dictionary mapping model types to lists of all their configurations.

    This allows users to specify a model type (e.g., "chronos") and automatically
    run all configurations for that type (e.g., chronos-tiny, chronos-small, etc.)

    Returns:
        Dict mapping model type names to lists of model configuration names
    """

    return {

        # -------------- Statistical & Naive Models --------------
        "naive": [
            "random",
            "oracle",
        ],

        # Statistical models
        "statistical": [
            "exp-smoothing",
            "r-ets",
            "r-arima",
        ],
        
        
        # # -------------- foundation models --------------
        # All foundation models (time series specific)
        # region start 
        "foundation-all": [
            "chronos-tiny",
            "chronos-mini",
            "chronos-small",
            "chronos-base",
            "chronos-large",
            "moirai-small",
            "moirai-base",
            "moirai-large",
            "lag-llama",
        ],

        # Foundation Models - Chronos
        "chronos": [
            "chronos-tiny",
            "chronos-mini",
            "chronos-small",
            "chronos-base",
            "chronos-large",
        ],

        # Foundation Models - Moirai
        "moirai": [
            "moirai-small",
            "moirai-base",
            "moirai-large",
        ],

        "lag-llama": ["lag-llama"], 
        # end



        # Multimodal models (only pred96 available for etth1)
        # ----------------------------------------------------------
        # running configs:
        "multimodal-all": [
            "timellm-etth1-nocontext-pred96",
            "timellm-etth1-context-pred96",
            "unitime-etth1-nocontext-pred96",
            "unitime-etth1-context-pred96",
        ],

        # TimeLLM / UniTime groups (official)
        "timellm-context": ["timellm-context-pred96"],
        "timellm-nocontext": ["timellm-nocontext-pred96"],
        
        "timellm": [
            "timellm-nocontext-pred96",
            "timellm-context-pred96",
        ],
        "unitime-nocontext": ["unitime-nocontext-pred96"],
        "unitime-context": ["unitime-context-pred96"],
        "unitime": [
            "unitime-nocontext-pred96",
            "unitime-context-pred96",
        ],
        
        # --------------- TimeLLM Models (official config) ---------------
        # "timellm-nocontext": [
        #     "timellm-etth1-nocontext-pred96",
        # ],
        # "timellm-context": [
        #     "timellm-etth1-context-pred96",
        # ],
        # "timellm": [
        #     "timellm-etth1-nocontext-pred96",
        #     "timellm-etth1-context-pred96",
        # ],

        # # --------------- UniTime Models (official config) ---------------
        # "unitime-nocontext": [
        #     "unitime-etth1-nocontext-pred96",
        # ],
        # "unitime-context": [
        #     "unitime-etth1-context-pred96",
        # ],
        # "unitime": [
        #     "unitime-etth1-nocontext-pred96",
        #     "unitime-etth1-context-pred96",
        # ],

        # ----------------------------------------------------------
        # DirectPrompt + API models
        # ----------------------------------------------------------
        "directprompt-all": [
            # OpenAI models (DirectPrompt)
            "gpt-4o-nocontext",
            "gpt-4o-context",
            "gpt-4o-mini-nocontext",
            "gpt-4o-mini-context",
            "gpt-5-mini-nocontext",
            "gpt-5-mini-context",

            # Anthropic Claude models (DirectPrompt)
            "claude-4.5-opus-nocontext",
            "claude-4.5-opus-context",
            "claude-4.5-sonnet-nocontext",
            "claude-4.5-sonnet-context",

            # OpenRouter models (DirectPrompt)
            "openrouter-llama-3-8b-instruct-nocontext",
            "openrouter-llama-3-8b-instruct-context",
            "openrouter-llama-3-70b-instruct-nocontext",
            "openrouter-llama-3-70b-instruct-context",
            "openrouter-mixtral-8x7b-instruct-nocontext",
            "openrouter-mixtral-8x7b-instruct-context",
            # (Optional) 405B via OpenRouter if available in your account
            "openrouter-llama-3.1-405b-instruct-nocontext",
            "openrouter-llama-3.1-405b-instruct-context",
            # New OpenRouter models
            "openrouter-qwen3-235b-a22b-instruct-nocontext",
            "openrouter-qwen3-235b-a22b-instruct-context",
            "openrouter-gemini-2.5-flash-nocontext",
            "openrouter-gemini-2.5-flash-context",
            "openrouter-claude-3.5-haiku-nocontext",
            "openrouter-claude-3.5-haiku-context",

            # Qwen local HF models (small → large)
            "qwen2.5-0.5b-instruct-nocontext",
            "qwen2.5-0.5b-instruct-context",
            "qwen2.5-7b-instruct-nocontext",
            "qwen2.5-7b-instruct-context",

            # Nixtla TimeGEN-1 API
            "timegen1",
        ],
        
        # Backwards-compatible group name used in docs/older scripts
        "gpt": [
            "gpt-4o-nocontext",
            "gpt-4o-context",
            "gpt-4o-mini-nocontext",
            "gpt-4o-mini-context",
            "gpt-5-mini-nocontext",
            "gpt-5-mini-context",
        ],
        # Claude models via Anthropic API
        "claude": [
            "claude-4.5-opus-nocontext",
            "claude-4.5-opus-context",
            "claude-4.5-sonnet-nocontext",
            "claude-4.5-sonnet-context",
        ],
        # Claude Code SDK (subscription-based)
        # Using Claude 4.5 models with specific snapshot versions
        "claude-sdk": [
            "claude-sdk-haiku-4.5-nocontext",
            "claude-sdk-haiku-4.5-context",
            "claude-sdk-sonnet-4.5-nocontext",
            "claude-sdk-sonnet-4.5-context",
            "claude-sdk-opus-4.5-nocontext",
            "claude-sdk-opus-4.5-context",
        ],
        # Convenience group: only the OpenRouter DirectPrompt models
        "openrouter": [
            "openrouter-llama-3-8b-instruct-nocontext",
            "openrouter-llama-3-8b-instruct-context",
            "openrouter-llama-3-70b-instruct-nocontext",
            "openrouter-llama-3-70b-instruct-context",
            "openrouter-mixtral-8x7b-instruct-nocontext",
            "openrouter-mixtral-8x7b-instruct-context",
            "openrouter-llama-3.1-405b-instruct-nocontext",
            "openrouter-llama-3.1-405b-instruct-context",
            # New OpenRouter models
            "openrouter-qwen3-235b-a22b-instruct-nocontext",
            "openrouter-qwen3-235b-a22b-instruct-context",
            "openrouter-gemini-2.5-flash-nocontext",
            "openrouter-gemini-2.5-flash-context",
            "openrouter-claude-3.5-haiku-nocontext",
            "openrouter-claude-3.5-haiku-context",
        ],

        "timegen1": ["timegen1"],
        "qwen": [
            "qwen2.5-0.5b-instruct-nocontext",
            "qwen2.5-0.5b-instruct-context",
            "qwen2.5-7b-instruct-nocontext",
            "qwen2.5-7b-instruct-context",
        ],
        # Convenience group for the smallest Qwen configs (fast local smoke tests)
        "qwen-small": [
            "qwen2.5-0.5b-instruct-nocontext",
            "qwen2.5-0.5b-instruct-context",
        ],


        # -------------- LLM Processes (LLMP) --------------
        # Keep aligned with the experiment specs in `experiments/llmp-models/*.json`
        # Note: experiment method name is "llmp". In this script, individual model names
        # are prefixed with "llmp-" (e.g. llmp-llama-3-8B-context). Passing "--model llmp"
        # expands to this list.
        "llmp-all": [
            "llmp-llama-3-8B-nocontext",
            "llmp-llama-3-8B-context",
            "llmp-llama-3-8B-instruct-nocontext",
            "llmp-llama-3-8B-instruct-context",
            "llmp-llama-3-70B-nocontext",
            "llmp-llama-3-70B-context",
            "llmp-llama-3-70B-instruct-nocontext",
            "llmp-llama-3-70B-instruct-context",
            "llmp-mixtral-8x7B-nocontext",
            "llmp-mixtral-8x7B-context",
            "llmp-mixtral-8x7B-instruct-nocontext",
            "llmp-mixtral-8x7B-instruct-context",
            # Qwen (local HF via `code/llm_processes/hf_api.py:llm_map`)
            "llmp-qwen2.5-0.5B-Instruct-nocontext",
            "llmp-qwen2.5-0.5B-Instruct-context",
            "llmp-qwen2.5-7B-Instruct-nocontext",
            "llmp-qwen2.5-7B-Instruct-context",
        ],
        'llama3':[
            'llmp-llama-3-8B-nocontext', 
            'llmp-llama-3-8B-context', 
            'llmp-llama-3-8B-instruct-nocontext', 
            'llmp-llama-3-8B-instruct-context', 
            'llmp-llama-3-70B-nocontext', 
            'llmp-llama-3-70B-context', 
            'llmp-llama-3-70B-instruct-nocontext', 
            'llmp-llama-3-70B-instruct-context'
            ],
        "llmp-qwen": [
            "llmp-qwen2.5-0.5B-Instruct-nocontext",
            "llmp-qwen2.5-0.5B-Instruct-context",
            "llmp-qwen2.5-7B-Instruct-nocontext",
            "llmp-qwen2.5-7B-Instruct-context",
        ],
        'llmp-sample':[
            # Quick local sanity checks (small models)
            'llmp-qwen2.5-0.5B-Instruct-nocontext',
            'llmp-qwen2.5-0.5B-Instruct-context',
            'llmp-llama-3-8B-nocontext', 
            'llmp-llama-3-8B-context', 
            ],
        'mixtral':[
            'llmp-mixtral-8x7B-nocontext', 
            'llmp-mixtral-8x7B-context', 
            'llmp-mixtral-8x7B-instruct-nocontext', 
            'llmp-mixtral-8x7B-instruct-context'
            ],
    }


# ============================================================================
# MODEL REGISTRY - Map string names to model factories
# ============================================================================

def get_model_registry(sleep_between_requests: float = 0.0) -> Dict[str, Callable]:
    """
    Return a dictionary mapping model names to factory functions.

    Factory functions take no arguments and return a forecaster instance
    or function that can be passed to evaluate_all_tasks.

    Args:
        sleep_between_requests: Sleep time in seconds between API requests for API-based models
    
    Returns:
        Dict mapping model name strings to factory callables
    """

    # --- Lazy import helpers (avoid importing heavy deps during --list-* commands) ---
    def _random():
        from eventglucose.baselines.naive import random_baseline
        return random_baseline

    def _oracle():
        from eventglucose.baselines.naive import oracle_baseline
        return oracle_baseline

    def _chronos(model_size: str):
        from eventglucose.baselines.chronos import ChronosForecaster
        return ChronosForecaster(model_size=model_size)

    def _moirai(model_size: str):
        from eventglucose.baselines.moirai import MoiraiForecaster
        return MoiraiForecaster(model_size=model_size)

    def _lag_llama():
        from eventglucose.baselines.lag_llama import lag_llama
        return lag_llama

    def _exp_smoothing():
        from eventglucose.baselines.statsmodels import ExponentialSmoothingForecaster
        return ExponentialSmoothingForecaster()

    def _r_ets():
        from eventglucose.baselines.r_forecast import R_ETS
        return R_ETS()

    def _r_arima():
        from eventglucose.baselines.r_forecast import R_Arima
        return R_Arima()

    def _timegen1():
        from eventglucose.baselines.timegen import timegen1
        return timegen1

    def _direct_prompt(model: str, use_context: bool, token_cost: dict, sleep_between_requests: float = 0.0, fail_on_invalid: bool = False, batch_size: int = None):
        from eventglucose.baselines.direct_prompt import DirectPrompt
        return DirectPrompt(
            model=model,
            use_context=use_context,
            token_cost=token_cost,
            sleep_between_requests=sleep_between_requests,
            fail_on_invalid=fail_on_invalid,
            batch_size=batch_size
        )

    def _timellm(use_context: bool, dataset: str, pred_len: int):
        from eventglucose.baselines.timellm import TimeLLMForecaster
        return TimeLLMForecaster(use_context=use_context, dataset=dataset, pred_len=pred_len)

    def _unitime(use_context: bool, dataset: str, pred_len: int):
        from eventglucose.baselines.unitime import UniTimeForecaster
        return UniTimeForecaster(use_context=use_context, dataset=dataset, pred_len=pred_len)

    def _llmp(llm_type: str, use_context: bool):
        from eventglucose.baselines.llm_processes import LLMPForecaster
        return LLMPForecaster(llm_type=llm_type, use_context=use_context)

    registry = {
        # Naive baselines
        "random": _random,
        "oracle": _oracle,

        # Chronos models
        "chronos-tiny": lambda: _chronos("tiny"),
        "chronos-mini": lambda: _chronos("mini"),
        "chronos-small": lambda: _chronos("small"),
        "chronos-base": lambda: _chronos("base"),
        "chronos-large": lambda: _chronos("large"),

        # Moirai models
        "moirai-small": lambda: _moirai("small"),
        "moirai-base": lambda: _moirai("base"),
        "moirai-large": lambda: _moirai("large"),

        # Lag-Llama
        "lag-llama": _lag_llama,

        # Statistical models
        "exp-smoothing": _exp_smoothing,
        "r-ets": _r_ets,
        "r-arima": _r_arima,

        # TimeGEN
        "timegen1": _timegen1,
    }

    # DirectPrompt models (LLM-based)
    openai_costs = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo": {"input": 0.003, "output": 0.006},
        "gpt-5-mini": {"input": 0.0003, "output": 0.0012},  # Estimated cost for GPT-5-mini
    }

    for model_name in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]:
        # Without context
        registry[f"{model_name}-nocontext"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m,
            use_context=False,
            token_cost=openai_costs.get(m, {"input": 0.0, "output": 0.0}),
            sleep_between_requests=s,
        )
        # With context
        registry[f"{model_name}-context"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m,
            use_context=True,
            token_cost=openai_costs.get(m, {"input": 0.0, "output": 0.0}),
            sleep_between_requests=s,
        )

    # GPT-5-mini with batch_size=5 (API max is 8, but we use 5 for consistency)
    registry["gpt-5-mini-nocontext"] = lambda s=sleep_between_requests: _direct_prompt(
        model="gpt-5-mini",
        use_context=False,
        token_cost=openai_costs.get("gpt-5-mini", {"input": 0.0, "output": 0.0}),
        sleep_between_requests=s,
        batch_size=5,
    )
    registry["gpt-5-mini-context"] = lambda s=sleep_between_requests: _direct_prompt(
        model="gpt-5-mini",
        use_context=True,
        token_cost=openai_costs.get("gpt-5-mini", {"input": 0.0, "output": 0.0}),
        sleep_between_requests=s,
        batch_size=5,
    )

    # OpenRouter DirectPrompt models (API)
    # These are referenced by experiment specs in `experiments/direct-prompt-models/*`
    # and are supported by DirectPrompt.get_client() when the model startswith "openrouter-".
    for model_name in [
        "openrouter-llama-3-8b-instruct",
        "openrouter-llama-3-70b-instruct",
        "openrouter-mixtral-8x7b-instruct",
        "openrouter-llama-3.1-405b-instruct",
        # New OpenRouter models
        "openrouter-qwen3-235b-a22b-instruct",  # Qwen3 235B A22B Instruct 2507
        "openrouter-gemini-2.5-flash",  # Google Gemini 2.5 Flash
        "openrouter-claude-3.5-haiku",  # Anthropic Claude 3.5 Haiku (cheap & fast)
    ]:
        registry[f"{model_name}-nocontext"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m,
            use_context=False,
            token_cost={"input": 0.0, "output": 0.0},
            sleep_between_requests=s,
        )
        registry[f"{model_name}-context"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m,
            use_context=True,
            token_cost={"input": 0.0, "output": 0.0},
            sleep_between_requests=s,
        )

    # Experiment direct-prompt configs sometimes specify these without the "openrouter-" prefix.
    # Route them to the OpenRouter client to keep CLI usage consistent with experiments.
    registry["llama-3.1-405b-instruct-nocontext"] = registry["openrouter-llama-3.1-405b-instruct-nocontext"]
    registry["llama-3.1-405b-instruct-context"] = registry["openrouter-llama-3.1-405b-instruct-context"]

    # Qwen DirectPrompt models (local HuggingFace via LLM_MAP)
    #
    # We register small Qwen2.5 instruct checkpoints for fast local smoke tests.
    # Canonical keys are cased like "qwen2.5-0.5B-Instruct" (see `dp_hf_api.LLM_MAP`);
    # we also accept lowercase aliases like "qwen2.5-0.5b-instruct".
    _qwen_sizes = ["0.5B", "1.5B", "3B", "7B"]
    for _size in _qwen_sizes:
        _qwen_canonical = f"qwen2.5-{_size}-Instruct"
        _qwen_aliases = [
            _qwen_canonical,
            f"qwen2.5-{_size.lower()}-instruct",
        ]
        for _alias in _qwen_aliases:
            registry[f"{_alias}-nocontext"] = lambda m=_qwen_canonical, s=sleep_between_requests: _direct_prompt(
                model=m,
                use_context=False,
                token_cost={"input": 0.0, "output": 0.0},
                sleep_between_requests=s,
            )
            registry[f"{_alias}-context"] = lambda m=_qwen_canonical, s=sleep_between_requests: _direct_prompt(
                model=m,
                use_context=True,
                token_cost={"input": 0.0, "output": 0.0},
                sleep_between_requests=s,
            )

    # Anthropic Claude DirectPrompt models (API)
    # Pricing as of Jan 2026 (estimated)
    claude_costs = {
        "claude-4.5-opus": {"input": 0.015, "output": 0.075},
        "claude-4.5-sonnet": {"input": 0.003, "output": 0.015},
    }
    for model_name in ["claude-4.5-opus", "claude-4.5-sonnet"]:
        registry[f"{model_name}-nocontext"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m,
            use_context=False,
            token_cost=claude_costs.get(m, {"input": 0.0, "output": 0.0}),
            sleep_between_requests=s,
        )
        registry[f"{model_name}-context"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m,
            use_context=True,
            token_cost=claude_costs.get(m, {"input": 0.0, "output": 0.0}),
            sleep_between_requests=s,
        )

    # Claude Code SDK models (subscription-based)
    # Tracks equivalent API costs for usage monitoring (actual cost: FREE with Pro/Max)
    # Note: batch_size=None lets DirectPrompt use n_samples as default batch size
    # Using specific snapshot versions for reproducibility:
    # - haiku-4.5: claude-haiku-4-5-20251001
    # - sonnet-4.5: claude-sonnet-4-5-20250929
    # - opus-4.5: claude-opus-4-5-20251101
    # Source: https://platform.claude.com/docs/en/about-claude/pricing
    ccsdk_costs = {
        "claude-sdk-haiku-4.5": {"input": 0.000001, "output": 0.000005},   # $1/$5 per M tokens
        "claude-sdk-sonnet-4.5": {"input": 0.000003, "output": 0.000015},  # $3/$15 per M tokens
        "claude-sdk-opus-4.5": {"input": 0.000005, "output": 0.000025},    # $5/$25 per M tokens
    }

    for model_name in ["claude-sdk-haiku-4.5", "claude-sdk-sonnet-4.5", "claude-sdk-opus-4.5"]:
        registry[f"{model_name}-nocontext"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m,
            use_context=False,
            token_cost=ccsdk_costs[m],  # Track equivalent API cost (actual: FREE)
            sleep_between_requests=s,
            fail_on_invalid=False,  # Don't fail on parsing errors
            batch_size=None,  # Use n_samples as batch size
        )
        registry[f"{model_name}-context"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m,
            use_context=True,
            token_cost=ccsdk_costs[m],  # Track equivalent API cost (actual: FREE)
            sleep_between_requests=s,
            fail_on_invalid=False,  # Don't fail on parsing errors
            batch_size=None,  # Use n_samples as batch size
        )

    # ---------------------------------------------------------------------
    # TimeLLM / UniTime
    #
    # "Official" experiment specs in `experiments/*` use dataset="etth1" and
    # pred_len=96. The underlying baseline implementations also hardcode
    # args.pred_len=96 internally, so exposing pred24/pred48 here is misleading.
    # ---------------------------------------------------------------------
    for pred_len in [96]:
        # Backwards-compatible names (kept)
        registry[f"timellm-nocontext-pred{pred_len}"] = lambda pl=pred_len: _timellm(
            use_context=False,
            dataset="etth1",
            pred_len=pl,
        )
        registry[f"timellm-context-pred{pred_len}"] = lambda pl=pred_len: _timellm(
            use_context=True,
            dataset="etth1",
            pred_len=pl,
        )
        registry[f"unitime-nocontext-pred{pred_len}"] = lambda pl=pred_len: _unitime(
            use_context=False,
            dataset="etth1",
            pred_len=pl,
        )
        registry[f"unitime-context-pred{pred_len}"] = lambda pl=pred_len: _unitime(
            use_context=True,
            dataset="etth1",
            pred_len=pl,
        )

        # Explicit "official" names (recommended going forward)
        registry[f"timellm-etth1-nocontext-pred{pred_len}"] = registry[f"timellm-nocontext-pred{pred_len}"]
        registry[f"timellm-etth1-context-pred{pred_len}"] = registry[f"timellm-context-pred{pred_len}"]
        registry[f"unitime-etth1-nocontext-pred{pred_len}"] = registry[f"unitime-nocontext-pred{pred_len}"]
        registry[f"unitime-etth1-context-pred{pred_len}"] = registry[f"unitime-context-pred{pred_len}"]

    # ---------------------------------------------------------------------
    # LLM Process (LLMP)
    #
    # Mirror the experiment specs in `experiments/llmp-models/*.json`.
    # (These strings are passed through to LLMPForecaster/llm_processes.)
    # ---------------------------------------------------------------------
    llmp_llms = [
        "llama-3-8B",
        "llama-3-8B-instruct",
        "llama-3-70B",
        "llama-3-70B-instruct",
        "mixtral-8x7B",
        "mixtral-8x7B-instruct",
        # Qwen (local HF)
        "qwen2.5-0.5B-Instruct",
        "qwen2.5-7B-Instruct",
    ]
    for llm_type in llmp_llms:
        registry[f"llmp-{llm_type}-nocontext"] = lambda lt=llm_type: _llmp(
            llm_type=lt,
            use_context=False,
        )
        registry[f"llmp-{llm_type}-context"] = lambda lt=llm_type: _llmp(
            llm_type=lt,
            use_context=True,
        )

    return registry


# ============================================================================
# CLI Functions
# ============================================================================

def list_tasks():
    """Print all available task names."""
    task_registry = get_task_registry()

    print("\n" + "=" * 80)
    print("AVAILABLE TASKS")
    print("=" * 80)
    print(f"\nTotal: {len(task_registry)} tasks\n")
    print("Tip: You can use task alias suffixes:")
    print("  - *_context    → expands to *_Profile + *_MediumEvent + *_DetailedEvent")
    print("  - *_newcontext → expands to *_NewMedium + *_NewDetail")
    print("  - *_allcontext → expands to all 5 context levels (Profile, MediumEvent, DetailedEvent, NewMedium, NewDetail)")
    print("  - *_nocontext  → expands to *_NoCtx\n")

    # Categorize tasks
    categories = {
        "Base Tasks": [],
        "Event-Specific Tasks": [],
        "Subgroup Tasks": [],
        "Diet Profile Tasks": [],
        "Diet Medium Event Tasks": [],
        "Diet Detailed Event Tasks": [],
        "Diet No Context Tasks": [],
        "Exercise Profile Tasks": [],
        "Exercise Medium Event Tasks": [],
        "Exercise Detailed Event Tasks": [],
        "Exercise No Context Tasks": [],
    }

    for name in sorted(task_registry.keys()):
        if "Base" in name or "test" in name or "TypeOne" in name:
            categories["Base Tasks"].append(name)
        elif "Exercise" in name and "DetailedEvent" in name:
            categories["Exercise Detailed Event Tasks"].append(name)
        elif "Exercise" in name and "MediumEvent" in name:
            categories["Exercise Medium Event Tasks"].append(name)
        elif "Exercise" in name and "Profile" in name:
            categories["Exercise Profile Tasks"].append(name)
        elif "Exercise" in name and "NoCtx" in name:
            categories["Exercise No Context Tasks"].append(name)
        elif "Diet" in name and "DetailedEvent" in name:
            categories["Diet Detailed Event Tasks"].append(name)
        elif "Diet" in name and "MediumEvent" in name:
            categories["Diet Medium Event Tasks"].append(name)
        elif "Diet" in name and "Profile" in name:
            categories["Diet Profile Tasks"].append(name)
        elif "Diet" in name and "NoCtx" in name:
            categories["Diet No Context Tasks"].append(name)
        elif "Diet5Min" in name or "Exercise5Min" in name:
            categories["Event-Specific Tasks"].append(name)
        elif any(f"_{age}" in name for age in ["Age18", "Age40", "Age65"]):
            categories["Subgroup Tasks"].append(name)

    for category, tasks in categories.items():
        if tasks:
            print(f"\n{category} ({len(tasks)}):")
            print("-" * 60)
            for task in tasks:
                print(f"  {task}")

    print("\n" + "=" * 80 + "\n")


def _task_alias_variants(base: str, variant: str) -> List[str]:
    """
    Map a task base prefix to concrete task class names.

    Example base:
        EventCGMTask_D1_Age18_Diet_Ontime

    Supported variants:
        - context    -> Profile, MediumEvent, DetailedEvent (original context levels)
        - newcontext -> NewMedium, NewDetail (new context levels)
        - allcontext -> all 5 context levels (Profile, MediumEvent, DetailedEvent, NewMedium, NewDetail)
        - nocontext  -> NoCtx
    """
    if variant == "context":
        return [
            f"{base}_Profile",
            f"{base}_MediumEvent",
            f"{base}_DetailedEvent",
        ]
    if variant == "newcontext":
        return [
            f"{base}_NewMedium",
            f"{base}_NewDetail",
        ]
    if variant == "allcontext":
        return [
            f"{base}_Profile",
            f"{base}_MediumEvent",
            f"{base}_DetailedEvent",
            f"{base}_NewMedium",
            f"{base}_NewDetail",
        ]
    if variant == "nocontext":
        return [f"{base}_NoCtx"]
    raise ValueError(f"Unknown task variant: {variant}")


def expand_task_names(task_specs: List[str]) -> List[str]:
    """
    Expand task spec strings into concrete task class names.

    Supported forms:
      - Exact task class name (e.g., EventCGMTask_Base)
      - Alias with suffix:
          * *_context    -> expands to Profile + MediumEvent + DetailedEvent
          * *_newcontext -> expands to NewMedium + NewDetail
          * *_allcontext -> expands to all 5 context levels
          * *_nocontext  -> expands to NoCtx
        Example:
          EventCGMTask_D1_Age18_Diet_Ontime_context
          EventCGMTask_D1_Age18_Diet_Ontime_newcontext
          EventCGMTask_D1_Age18_Diet_Ontime_allcontext
    """
    task_registry = get_task_registry()
    expanded: List[str] = []

    for spec in task_specs:
        if spec in task_registry:
            expanded.append(spec)
            continue

        m = re.match(r"^(.*?)(?:[_-])(context|newcontext|allcontext|nocontext)$", spec)
        if m:
            base, variant = m.group(1), m.group(2)
            expanded.extend(_task_alias_variants(base=base, variant=variant))
            continue

        # Keep as-is; validation later will print a clear message.
        expanded.append(spec)

    # Remove duplicates while preserving order
    seen = set()
    result: List[str] = []
    for item in expanded:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _is_model_nocontext_variant(model_name: str) -> bool:
    return "nocontext" in model_name


def _is_model_context_variant(model_name: str) -> bool:
    # Important: "nocontext" contains "context" substring.
    return ("context" in model_name) and ("nocontext" not in model_name)


def split_models_by_context(model_names: List[str]) -> Dict[str, List[str]]:
    """
    Split a model list into context / nocontext buckets.

    Models without a context marker (e.g., Chronos) are treated as context-agnostic
    and included in BOTH buckets.
    """
    context_models: List[str] = []
    nocontext_models: List[str] = []

    for m in model_names:
        if _is_model_context_variant(m):
            context_models.append(m)
        elif _is_model_nocontext_variant(m):
            nocontext_models.append(m)
        else:
            # context-agnostic -> include in both
            context_models.append(m)
            nocontext_models.append(m)

    return {"context": context_models, "nocontext": nocontext_models}


def expand_model_names(model_names: List[str]) -> List[str]:
    """
    Expand model type names to individual model configurations.

    Args:
        model_names: List of model names or model type names

    Returns:
        Expanded list with model types replaced by their configurations
    """
    model_type_registry = get_model_type_registry()
    expanded = []

    for name in model_names:
        if name in model_type_registry:
            # This is a model type, expand it
            expanded.extend(model_type_registry[name])
        else:
            # This is an individual model name
            expanded.append(name)

    # Remove duplicates while preserving order
    seen = set()
    result = []
    for item in expanded:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def list_model_types():
    """Print all available model type names and their expansions."""
    model_type_registry = get_model_type_registry()
    # Note: list_model_types doesn't need sleep_between_requests since it only lists types

    print("\n" + "=" * 80)
    print("AVAILABLE MODEL TYPES")
    print("=" * 80)
    print(f"\nTotal: {len(model_type_registry)} model types\n")
    print("Use model types to run all configurations for a model family.\n")

    # Categorize model types
    categories = {
        "Foundation Models": [],
        "DirectPrompt Models": [],
        "Multimodal Models": [],
        "Statistical Models": [],
        "Naive Baselines": [],
        "Comprehensive Groups": [],
    }

    for name in sorted(model_type_registry.keys()):
        if "chronos" in name or "moirai" in name or "foundation" in name or name in ["lag-llama", "timegen1"]:
            categories["Foundation Models"].append(name)
        elif "gpt" in name and "llmp" not in name:
            categories["DirectPrompt Models"].append(name)
        elif "claude" in name and "llmp" not in name:
            categories["DirectPrompt Models"].append(name)
        elif "openrouter" in name and "llmp" not in name:
            categories["DirectPrompt Models"].append(name)
        elif "timellm" in name or "unitime" in name or "multimodal" in name:
            categories["Multimodal Models"].append(name)
        elif "llmp" in name:
            categories["DirectPrompt Models"].append(name)
        elif "qwen" in name:
            categories["DirectPrompt Models"].append(name)
        elif "statistical" in name or name in ["r-ets", "r-arima", "exp-smoothing"]:
            categories["Statistical Models"].append(name)
        elif "naive" in name:
            categories["Naive Baselines"].append(name)
        elif "all" in name:
            categories["Comprehensive Groups"].append(name)

    for category, types in categories.items():
        if types:
            print(f"\n{category}:")
            print("-" * 60)
            for model_type in types:
                configs = model_type_registry[model_type]
                print(f"  {model_type:25} → {len(configs)} configs: {', '.join(configs[:3])}{'...' if len(configs) > 3 else ''}")

    print("\n" + "=" * 80)
    print("\nExample usage:")
    print("  --model-type chronos              # Run all 5 Chronos variants")
    print("  --model-type gpt-4o               # Run GPT-4o with and without context")
    print("  --model-type foundation-all       # Run all foundation models")
    print("  --model-type chronos moirai       # Run all Chronos + all Moirai models")
    print("=" * 80 + "\n")


def list_models():
    """Print all available model names."""
    model_registry = get_model_registry(sleep_between_requests=0.0)

    print("\n" + "=" * 80)
    print("AVAILABLE INDIVIDUAL MODELS")
    print("=" * 80)
    print(f"\nTotal: {len(model_registry)} individual model configurations\n")
    print("Note: Use --list-model-types to see model type groups.\n")

    # Categorize models
    categories = {
        "Naive Baselines": [],
        "Chronos Models": [],
        "Moirai Models": [],
        "Lag-Llama": [],
        "Statistical Models": [],
        "TimeGEN": [],
        "DirectPrompt (GPT)": [],
        "DirectPrompt (Claude)": [],
        "DirectPrompt (OpenRouter)": [],
        "DirectPrompt (Qwen)": [],
        "TimeLLM": [],
        "UniTime": [],
        "LLM Processes": [],
    }

    for name in sorted(model_registry.keys()):
        if name in ["random", "oracle"]:
            categories["Naive Baselines"].append(name)
        elif "chronos" in name:
            categories["Chronos Models"].append(name)
        elif "moirai" in name:
            categories["Moirai Models"].append(name)
        elif "lag-llama" in name:
            categories["Lag-Llama"].append(name)
        elif name in ["exp-smoothing", "r-ets", "r-arima"]:
            categories["Statistical Models"].append(name)
        elif "timegen" in name:
            categories["TimeGEN"].append(name)
        elif "gpt" in name and "llmp" not in name:
            categories["DirectPrompt (GPT)"].append(name)
        elif "claude" in name and "llmp" not in name:
            categories["DirectPrompt (Claude)"].append(name)
        elif "openrouter" in name and "llmp" not in name:
            categories["DirectPrompt (OpenRouter)"].append(name)
        elif "qwen" in name and "llmp" not in name:
            categories["DirectPrompt (Qwen)"].append(name)
        elif "timellm" in name:
            categories["TimeLLM"].append(name)
        elif "unitime" in name:
            categories["UniTime"].append(name)
        elif "llmp" in name:
            categories["LLM Processes"].append(name)

    for category, models in categories.items():
        if models:
            print(f"\n{category} ({len(models)}):")
            print("-" * 60)
            for model in models:
                print(f"  {model}")

    print("\n" + "=" * 80 + "\n")


def run_evaluation(
    task_names: List[str],
    model_names: List[str],
    n_instances: int = 10,
    n_samples: int = 25,
    output_folder: str | None = None,
    max_parallel: int = None,
    skip_cache_miss: bool = False,
    skip_done: bool = False,
    results_roots: List[str] | None = None,
    sleep_between_requests: float = 0.0,
):
    """
    Run evaluation for specified task and model combinations.

    Args:
        task_names: List of task names to evaluate
        model_names: List of model names or model types to use
        n_instances: Number of task instances to evaluate
        n_samples: Number of forecast samples per instance
        output_folder: Directory to save results
        max_parallel: Maximum parallel workers
        skip_cache_miss: Skip if cache miss
    """
    task_registry = get_task_registry()
    model_registry = get_model_registry(sleep_between_requests=sleep_between_requests)
    model_type_registry = get_model_type_registry()

    # Resolve output folder at runtime:
    # - If caller didn't pass output_folder, match CLI default: $LOCAL_RESULTS_FOLDER or ./results
    output_folder = output_folder or os.environ.get("LOCAL_RESULTS_FOLDER", "./results")

    # Expand task aliases (e.g., *_context, *_nocontext)
    original_task_names = task_names.copy()
    task_names = expand_task_names(task_names)

    # Expand model types to individual model names
    original_model_names = model_names.copy()
    model_names = expand_model_names(model_names)

    # Show expansion if any model types were used
    expanded_types = []
    for name in original_model_names:
        if name in model_type_registry:
            expanded_types.append(f"{name} → {len(model_type_registry[name])} configs")

    # Validate task names
    invalid_tasks = [t for t in task_names if t not in task_registry]
    if invalid_tasks:
        print(f"Error: Invalid task names: {invalid_tasks}")
        print("Use --list-tasks to see available tasks")
        print("\nIf you intended a context alias, use one of:")
        print("  - *_context    (expands to *_Profile, *_MediumEvent, *_DetailedEvent)")
        print("  - *_newcontext (expands to *_NewMedium, *_NewDetail)")
        print("  - *_allcontext (expands to all 5 context levels)")
        print("  - *_nocontext  (expands to *_NoCtx)")
        return

    # Validate model names (after expansion)
    invalid_models = [m for m in model_names if m not in model_registry]
    if invalid_models:
        print(f"Error: Invalid model names: {invalid_models}")
        print("Use --list-models to see available models")
        print("Use --list-model-types to see available model types")
        return

    # Get task classes
    task_classes = [task_registry[name] for name in task_names]

    print("\n" + "=" * 80)
    print(f"RUNNING EVALUATION: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 80)
    print(f"\nTasks ({len(task_classes)}):")
    for name in task_names:
        print(f"  - {name}")

    # Show alias expansions if any were used
    expanded_task_aliases = []
    for name in original_task_names:
        if re.match(r"^(.*?)(?:[_-])(context|newcontext|allcontext|nocontext)$", name):
            expanded_task_aliases.append(name)
    if expanded_task_aliases:
        print(f"\nTask Alias Expansion:")
        for alias in expanded_task_aliases:
            base, variant = re.match(r"^(.*?)(?:[_-])(context|newcontext|allcontext|nocontext)$", alias).group(1, 2)
            print(f"  - {alias} → {', '.join(_task_alias_variants(base, variant))}")

    if expanded_types:
        print(f"\nModel Type Expansion:")
        for expansion in expanded_types:
            print(f"  - {expansion}")

    print(f"\nModels to run ({len(model_names)} total):")
    for name in model_names:
        print(f"  - {name}")

    print(f"\nParameters:")
    print(f"  - n_instances: {n_instances}")
    print(f"  - n_samples: {n_samples}")
    print(f"  - output_folder: {output_folder}")
    print(f"  - max_parallel: {max_parallel}")
    if sleep_between_requests > 0:
        print(f"  - sleep_between_requests: {sleep_between_requests}s (for API-based models)")
    print("=" * 80 + "\n")

    # Run evaluations for each model
    results = {}

    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"Running model: {model_name}")
        print('=' * 60)

        # Get model instance
        model_factory = model_registry[model_name]
        model = model_factory()

        # Optionally skip tasks already done on disk (saves time when resuming runs)
        filtered_task_classes = task_classes
        if skip_done:
            try:
                from scripts.run_result_check import check_task_model_status
            except Exception as e:
                print(f"Warning: failed to import scripts.run_result_check ({e}); continuing without skip-done.")
                check_task_model_status = None  # type: ignore[assignment]

            if check_task_model_status is not None:
                # Default search roots to the same resolved output folder.
                # If the user doesn't pass --output, this will be $LOCAL_RESULTS_FOLDER.
                roots = results_roots if results_roots else [output_folder]
                pending: List[type] = []
                for tc in task_classes:
                    status, _report = check_task_model_status(
                        model_name=model_name,
                        task_name=tc.__name__,
                        results_roots=roots,
                        n_instances=n_instances,
                        n_samples=n_samples,
                    )
                    if status != "done":
                        pending.append(tc)

                if len(pending) != len(task_classes):
                    print(f"Skip-done: {len(task_classes) - len(pending)} task(s) already done for {model_name}.")
                filtered_task_classes = pending

        if not filtered_task_classes:
            print(f"✓ Skipping model {model_name}: all selected tasks are already done.")
            continue

        # Run evaluation
        output_path = Path(output_folder) / model_name

        try:
            from eventglucose.evaluation import evaluate_all_tasks
            result = evaluate_all_tasks(
                filtered_task_classes,
                model,
                n_instances=n_instances,
                n_samples=n_samples,
                output_folder=str(output_path),
                max_parallel=max_parallel,
                skip_cache_miss=skip_cache_miss,
            )

            results[model_name] = result

            print(f"\n✓ Completed: {model_name}")
            print(f"  Results saved to: {output_path}")

        except Exception as e:
            print(f"\n✗ Error running {model_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nCompleted models: {len(results)}/{len(model_names)}")
    print(f"Results directory: {output_folder}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run individual model and task combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # List commands
    parser.add_argument("--list-tasks", action="store_true",
                        help="List all available task names")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available individual model names")
    parser.add_argument("--list-model-types", action="store_true",
                        help="List all available model types (groups of models)")

    # Execution parameters
    parser.add_argument("--task", nargs="+", type=str,
                        help="Task name(s) to evaluate (can specify multiple)")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Run all available tasks (overrides --task)")
    parser.add_argument("--model", nargs="+", type=str,
                        help="Model name(s) or model type(s) to use (can specify multiple, will auto-expand types)")

    # Evaluation parameters
    parser.add_argument("--n-instances", type=int, default=10,
                        help="Number of task instances to evaluate (default: 10)")
    parser.add_argument("--n-samples", type=int, default=25,
                        help="Number of forecast samples per instance (default: 25)")
    parser.add_argument("--output", type=str,
                        default=os.environ.get("LOCAL_RESULTS_FOLDER", "./results"),
                        help="Output directory for results (default: $LOCAL_RESULTS_FOLDER or ./results)")
    parser.add_argument("--max-parallel", type=int, default=None,
                        help="Maximum parallel workers (default: None)")
    parser.add_argument("--skip-cache-miss", action="store_true",
                        help="Skip evaluation if cache miss")
    parser.add_argument(
        "--skip-done",
        action="store_true",
        help="Skip task×model pairs that are already complete on disk (based on data_summary.txt).",
    )
    parser.add_argument(
        "--results-roots",
        nargs="+",
        type=str,
        default=None,
        help=(
            "One or more results roots to search when --skip-done is enabled. "
            "Defaults to --output only (which itself defaults to $LOCAL_RESULTS_FOLDER). Useful if your results are split across folders "
            "(e.g. ./results and ./_WorkSpace/Result)."
        ),
    )
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.0,
        help=(
            "Sleep time in seconds between API requests for API-based models (e.g., GPT, OpenRouter). "
            "Useful to avoid rate limiting. Only applies to API-based models. Default: 0.0 (no sleep). "
            "Recommended values: 0.1-1.0 seconds depending on API rate limits."
        ),
    )

    args = parser.parse_args()

    # Handle list commands
    if args.list_tasks:
        list_tasks()
        return

    if args.list_models:
        list_models()
        return

    if args.list_model_types:
        list_model_types()
        return

    # Handle --all-tasks flag
    if args.all_tasks:
        task_registry = get_task_registry()
        task_names = sorted(task_registry.keys())
        print(f"\n--all-tasks flag detected: Running all {len(task_names)} tasks\n")
    else:
        task_names = args.task

    # Simple mode: require --model and either --task or --all-tasks
    if not args.all_tasks and not task_names:
        parser.error("--task is required unless you use --all-tasks.")
    if not args.model:
        parser.error("--model is required (or use --list-models/--list-model-types).")

    # Supports *_context / *_nocontext alias via run_evaluation (expand_task_names)
    run_evaluation(
        task_names=task_names or [],
        model_names=args.model or [],
        n_instances=args.n_instances,
        n_samples=args.n_samples,
        output_folder=args.output,
        max_parallel=args.max_parallel,
        skip_cache_miss=args.skip_cache_miss,
        skip_done=args.skip_done,
        results_roots=args.results_roots,
        sleep_between_requests=args.sleep_between_requests,
    )


if __name__ == "__main__":
    main()
