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
    #   *_context    -> expands to *_Profile + *_BasicEventInfo
    #   *_newcontext -> expands to *_StandardEventInfo + *_DetailedEventInfo
    #   *_allcontext -> expands to all 4 context levels
    #   *_nocontext  -> expands to *_NoCtx
    python code/scripts/run_individual.py \
        --task EventCGMTask_D1_Age18_Diet_Ontime_context \
        --model gpt-4o-context chronos-small \
        --n-instances 10 --n-samples 25

    # Run with model types (auto-expands to all configs)
    python code/scripts/run_individual.py \
        --task EventCGMTask_Base \
        --model chronos \
        --n-instances 10 \
        --n-samples 25
    # This runs all 5 Chronos variants: tiny, mini, small, base, large

    # Run comprehensive evaluation
    python code/scripts/run_individual.py \
        --task EventCGMTask_Base \
        --model foundation-all \
        --n-instances 20 \
        --n-samples 50
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

import matplotlib
matplotlib.use('Agg')


# ============================================================================
# Auto-load environment from env.sh
# ============================================================================
def load_env_from_shell():
    current_path = Path(__file__).resolve()

    for _ in range(3):
        current_path = current_path.parent
        env_file = current_path / "env.sh"

        if env_file.exists():
            is_subprocess = os.environ.get('_EVENTGLUCOSE_SUBPROCESS') == '1'

            if not is_subprocess:
                logging.info(f"Found env.sh at: {env_file}")

            try:
                cmd = f'source "{env_file}" && env'
                result = subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True,
                    text=True,
                    check=True,
                    cwd=str(current_path)
                )

                for line in result.stdout.splitlines():
                    if '=' in line:
                        key, value = line.split('=', 1)
                        if key not in os.environ:
                            os.environ[key] = value

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


logging.basicConfig(level=logging.INFO)
load_env_from_shell()
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# TASK REGISTRY
# ============================================================================

def get_task_registry() -> Dict[str, type]:
    from eventglucose.tasks.factory_task_Base import (
        EventCGMTask_Base,
    )

    from eventglucose.tasks.task_diet_ontime_profile import (
        EventCGMTask_D1_Age18_Diet_Ontime_Profile,
        EventCGMTask_D1_Age40_Diet_Ontime_Profile,
        EventCGMTask_D1_Age65_Diet_Ontime_Profile,
        EventCGMTask_D2_Age18_Diet_Ontime_Profile,
        EventCGMTask_D2_Age40_Diet_Ontime_Profile,
        EventCGMTask_D2_Age65_Diet_Ontime_Profile,
    )

    from eventglucose.tasks.task_diet_ontime_profile_basiceventinfo import (
        EventCGMTask_D1_Age18_Diet_Ontime_BasicEventInfo,
        EventCGMTask_D1_Age40_Diet_Ontime_BasicEventInfo,
        EventCGMTask_D1_Age65_Diet_Ontime_BasicEventInfo,
        EventCGMTask_D2_Age18_Diet_Ontime_BasicEventInfo,
        EventCGMTask_D2_Age40_Diet_Ontime_BasicEventInfo,
        EventCGMTask_D2_Age65_Diet_Ontime_BasicEventInfo,
    )

    from eventglucose.tasks.task_diet_ontime_profile_standardeventinfo import (
        EventCGMTask_D1_Age18_Diet_Ontime_StandardEventInfo,
        EventCGMTask_D1_Age40_Diet_Ontime_StandardEventInfo,
        EventCGMTask_D1_Age65_Diet_Ontime_StandardEventInfo,
        EventCGMTask_D2_Age18_Diet_Ontime_StandardEventInfo,
        EventCGMTask_D2_Age40_Diet_Ontime_StandardEventInfo,
        EventCGMTask_D2_Age65_Diet_Ontime_StandardEventInfo,
    )

    from eventglucose.tasks.task_diet_ontime_profile_detailedeventinfo import (
        EventCGMTask_D1_Age18_Diet_Ontime_DetailedEventInfo,
        EventCGMTask_D1_Age40_Diet_Ontime_DetailedEventInfo,
        EventCGMTask_D1_Age65_Diet_Ontime_DetailedEventInfo,
        EventCGMTask_D2_Age18_Diet_Ontime_DetailedEventInfo,
        EventCGMTask_D2_Age40_Diet_Ontime_DetailedEventInfo,
        EventCGMTask_D2_Age65_Diet_Ontime_DetailedEventInfo,
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

    from eventglucose.tasks.task_exercise_ontime_profile_basiceventinfo import (
        EventCGMTask_D1_Age18_Exercise_Ontime_BasicEventInfo,
        EventCGMTask_D1_Age40_Exercise_Ontime_BasicEventInfo,
        EventCGMTask_D1_Age65_Exercise_Ontime_BasicEventInfo,
        EventCGMTask_D2_Age18_Exercise_Ontime_BasicEventInfo,
        EventCGMTask_D2_Age40_Exercise_Ontime_BasicEventInfo,
        EventCGMTask_D2_Age65_Exercise_Ontime_BasicEventInfo,
    )

    from eventglucose.tasks.task_exercise_ontime_profile_standardeventinfo import (
        EventCGMTask_D1_Age18_Exercise_Ontime_StandardEventInfo,
        EventCGMTask_D1_Age40_Exercise_Ontime_StandardEventInfo,
        EventCGMTask_D1_Age65_Exercise_Ontime_StandardEventInfo,
        EventCGMTask_D2_Age18_Exercise_Ontime_StandardEventInfo,
        EventCGMTask_D2_Age40_Exercise_Ontime_StandardEventInfo,
        EventCGMTask_D2_Age65_Exercise_Ontime_StandardEventInfo,
    )

    from eventglucose.tasks.task_exercise_ontime_profile_detailedeventinfo import (
        EventCGMTask_D1_Age18_Exercise_Ontime_DetailedEventInfo,
        EventCGMTask_D1_Age40_Exercise_Ontime_DetailedEventInfo,
        EventCGMTask_D1_Age65_Exercise_Ontime_DetailedEventInfo,
        EventCGMTask_D2_Age18_Exercise_Ontime_DetailedEventInfo,
        EventCGMTask_D2_Age40_Exercise_Ontime_DetailedEventInfo,
        EventCGMTask_D2_Age65_Exercise_Ontime_DetailedEventInfo,
    )

    from eventglucose.tasks.task_exercise_ontime_noctx import (
        EventCGMTask_D1_Age18_Exercise_Ontime_NoCtx,
        EventCGMTask_D1_Age40_Exercise_Ontime_NoCtx,
        EventCGMTask_D1_Age65_Exercise_Ontime_NoCtx,
        EventCGMTask_D2_Age18_Exercise_Ontime_NoCtx,
        EventCGMTask_D2_Age40_Exercise_Ontime_NoCtx,
        EventCGMTask_D2_Age65_Exercise_Ontime_NoCtx,
    )

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

    task_classes = [
        EventCGMTask_Base,
        # Diet Profile
        EventCGMTask_D1_Age18_Diet_Ontime_Profile,
        EventCGMTask_D1_Age40_Diet_Ontime_Profile,
        EventCGMTask_D1_Age65_Diet_Ontime_Profile,
        EventCGMTask_D2_Age18_Diet_Ontime_Profile,
        EventCGMTask_D2_Age40_Diet_Ontime_Profile,
        EventCGMTask_D2_Age65_Diet_Ontime_Profile,
        # Diet BasicEventInfo
        EventCGMTask_D1_Age18_Diet_Ontime_BasicEventInfo,
        EventCGMTask_D1_Age40_Diet_Ontime_BasicEventInfo,
        EventCGMTask_D1_Age65_Diet_Ontime_BasicEventInfo,
        EventCGMTask_D2_Age18_Diet_Ontime_BasicEventInfo,
        EventCGMTask_D2_Age40_Diet_Ontime_BasicEventInfo,
        EventCGMTask_D2_Age65_Diet_Ontime_BasicEventInfo,
        # Diet StandardEventInfo
        EventCGMTask_D1_Age18_Diet_Ontime_StandardEventInfo,
        EventCGMTask_D1_Age40_Diet_Ontime_StandardEventInfo,
        EventCGMTask_D1_Age65_Diet_Ontime_StandardEventInfo,
        EventCGMTask_D2_Age18_Diet_Ontime_StandardEventInfo,
        EventCGMTask_D2_Age40_Diet_Ontime_StandardEventInfo,
        EventCGMTask_D2_Age65_Diet_Ontime_StandardEventInfo,
        # Diet DetailedEventInfo
        EventCGMTask_D1_Age18_Diet_Ontime_DetailedEventInfo,
        EventCGMTask_D1_Age40_Diet_Ontime_DetailedEventInfo,
        EventCGMTask_D1_Age65_Diet_Ontime_DetailedEventInfo,
        EventCGMTask_D2_Age18_Diet_Ontime_DetailedEventInfo,
        EventCGMTask_D2_Age40_Diet_Ontime_DetailedEventInfo,
        EventCGMTask_D2_Age65_Diet_Ontime_DetailedEventInfo,
        # Diet NoCtx
        EventCGMTask_D1_Age18_Diet_Ontime_NoCtx,
        EventCGMTask_D1_Age40_Diet_Ontime_NoCtx,
        EventCGMTask_D1_Age65_Diet_Ontime_NoCtx,
        EventCGMTask_D2_Age18_Diet_Ontime_NoCtx,
        EventCGMTask_D2_Age40_Diet_Ontime_NoCtx,
        EventCGMTask_D2_Age65_Diet_Ontime_NoCtx,
        # Exercise Profile
        EventCGMTask_D1_Age18_Exercise_Ontime_Profile,
        EventCGMTask_D1_Age40_Exercise_Ontime_Profile,
        EventCGMTask_D1_Age65_Exercise_Ontime_Profile,
        EventCGMTask_D2_Age18_Exercise_Ontime_Profile,
        EventCGMTask_D2_Age40_Exercise_Ontime_Profile,
        EventCGMTask_D2_Age65_Exercise_Ontime_Profile,
        # Exercise BasicEventInfo
        EventCGMTask_D1_Age18_Exercise_Ontime_BasicEventInfo,
        EventCGMTask_D1_Age40_Exercise_Ontime_BasicEventInfo,
        EventCGMTask_D1_Age65_Exercise_Ontime_BasicEventInfo,
        EventCGMTask_D2_Age18_Exercise_Ontime_BasicEventInfo,
        EventCGMTask_D2_Age40_Exercise_Ontime_BasicEventInfo,
        EventCGMTask_D2_Age65_Exercise_Ontime_BasicEventInfo,
        # Exercise StandardEventInfo
        EventCGMTask_D1_Age18_Exercise_Ontime_StandardEventInfo,
        EventCGMTask_D1_Age40_Exercise_Ontime_StandardEventInfo,
        EventCGMTask_D1_Age65_Exercise_Ontime_StandardEventInfo,
        EventCGMTask_D2_Age18_Exercise_Ontime_StandardEventInfo,
        EventCGMTask_D2_Age40_Exercise_Ontime_StandardEventInfo,
        EventCGMTask_D2_Age65_Exercise_Ontime_StandardEventInfo,
        # Exercise DetailedEventInfo
        EventCGMTask_D1_Age18_Exercise_Ontime_DetailedEventInfo,
        EventCGMTask_D1_Age40_Exercise_Ontime_DetailedEventInfo,
        EventCGMTask_D1_Age65_Exercise_Ontime_DetailedEventInfo,
        EventCGMTask_D2_Age18_Exercise_Ontime_DetailedEventInfo,
        EventCGMTask_D2_Age40_Exercise_Ontime_DetailedEventInfo,
        EventCGMTask_D2_Age65_Exercise_Ontime_DetailedEventInfo,
        # Exercise NoCtx
        EventCGMTask_D1_Age18_Exercise_Ontime_NoCtx,
        EventCGMTask_D1_Age40_Exercise_Ontime_NoCtx,
        EventCGMTask_D1_Age65_Exercise_Ontime_NoCtx,
        EventCGMTask_D2_Age18_Exercise_Ontime_NoCtx,
        EventCGMTask_D2_Age40_Exercise_Ontime_NoCtx,
        EventCGMTask_D2_Age65_Exercise_Ontime_NoCtx,
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
# MODEL TYPE REGISTRY
# ============================================================================

def get_model_type_registry() -> Dict[str, List[str]]:
    return {
        "naive": ["random", "oracle"],

        "statistical": ["exp-smoothing", "r-ets", "r-arima"],

        "linear": ["dlinear", "nlinear"],

        "transformer": ["itransformer", "autoformer", "causal"],
        "transformer-ctx": ["itransformer-ctx", "autoformer-ctx", "causal-ctx"],
        "transformer-all": [
            "itransformer", "autoformer", "causal",
            "itransformer-ctx", "autoformer-ctx", "causal-ctx",
        ],

        "foundation-all": [
            "chronos-tiny", "chronos-mini", "chronos-small", "chronos-base", "chronos-large",
            "moirai-small", "moirai-base", "moirai-large",
            "lag-llama",
        ],

        "chronos": [
            "chronos-tiny", "chronos-mini", "chronos-small", "chronos-base", "chronos-large",
        ],

        "moirai": ["moirai-small", "moirai-base", "moirai-large"],

        "lag-llama": ["lag-llama"],

        "multimodal-all": [
            "timellm-etth1-nocontext-pred96",
            "timellm-etth1-context-pred96",
            "unitime-etth1-nocontext-pred96",
            "unitime-etth1-context-pred96",
        ],

        "timellm-context": ["timellm-context-pred96"],
        "timellm-nocontext": ["timellm-nocontext-pred96"],
        "timellm": ["timellm-nocontext-pred96", "timellm-context-pred96"],
        "unitime-nocontext": ["unitime-nocontext-pred96"],
        "unitime-context": ["unitime-context-pred96"],
        "unitime": ["unitime-nocontext-pred96", "unitime-context-pred96"],

        "directprompt-all": [
            "gpt-4o-nocontext", "gpt-4o-context",
            "gpt-4o-mini-nocontext", "gpt-4o-mini-context",
            "gpt-5-mini-nocontext", "gpt-5-mini-context",
            "claude-4.5-opus-nocontext", "claude-4.5-opus-context",
            "claude-4.5-sonnet-nocontext", "claude-4.5-sonnet-context",
            "openrouter-llama-3-8b-instruct-nocontext", "openrouter-llama-3-8b-instruct-context",
            "openrouter-llama-3-70b-instruct-nocontext", "openrouter-llama-3-70b-instruct-context",
            "openrouter-mixtral-8x7b-instruct-nocontext", "openrouter-mixtral-8x7b-instruct-context",
            "openrouter-llama-3.1-405b-instruct-nocontext", "openrouter-llama-3.1-405b-instruct-context",
            "openrouter-qwen3-235b-a22b-instruct-nocontext", "openrouter-qwen3-235b-a22b-instruct-context",
            "openrouter-gemini-2.5-flash-nocontext", "openrouter-gemini-2.5-flash-context",
            "openrouter-claude-3.5-haiku-nocontext", "openrouter-claude-3.5-haiku-context",
            "qwen2.5-0.5b-instruct-nocontext", "qwen2.5-0.5b-instruct-context",
            "qwen2.5-7b-instruct-nocontext", "qwen2.5-7b-instruct-context",
            "timegen1",
        ],

        "gpt": [
            "gpt-4o-nocontext", "gpt-4o-context",
            "gpt-4o-mini-nocontext", "gpt-4o-mini-context",
            "gpt-5-mini-nocontext", "gpt-5-mini-context",
        ],
        "claude": [
            "claude-4.5-opus-nocontext", "claude-4.5-opus-context",
            "claude-4.5-sonnet-nocontext", "claude-4.5-sonnet-context",
        ],
        "claude-sdk": [
            "claude-sdk-haiku-4.5-nocontext", "claude-sdk-haiku-4.5-context",
            "claude-sdk-sonnet-4.5-nocontext", "claude-sdk-sonnet-4.5-context",
            "claude-sdk-opus-4.5-nocontext", "claude-sdk-opus-4.5-context",
        ],
        "openrouter": [
            "openrouter-llama-3-8b-instruct-nocontext", "openrouter-llama-3-8b-instruct-context",
            "openrouter-llama-3-70b-instruct-nocontext", "openrouter-llama-3-70b-instruct-context",
            "openrouter-mixtral-8x7b-instruct-nocontext", "openrouter-mixtral-8x7b-instruct-context",
            "openrouter-llama-3.1-405b-instruct-nocontext", "openrouter-llama-3.1-405b-instruct-context",
            "openrouter-qwen3-235b-a22b-instruct-nocontext", "openrouter-qwen3-235b-a22b-instruct-context",
            "openrouter-gemini-2.5-flash-nocontext", "openrouter-gemini-2.5-flash-context",
            "openrouter-claude-3.5-haiku-nocontext", "openrouter-claude-3.5-haiku-context",
        ],

        "timegen1": ["timegen1"],
        "qwen": [
            "qwen2.5-0.5b-instruct-nocontext", "qwen2.5-0.5b-instruct-context",
            "qwen2.5-7b-instruct-nocontext", "qwen2.5-7b-instruct-context",
        ],
        "qwen-small": [
            "qwen2.5-0.5b-instruct-nocontext", "qwen2.5-0.5b-instruct-context",
        ],

        "llmp-all": [
            "llmp-llama-3-8B-nocontext", "llmp-llama-3-8B-context",
            "llmp-llama-3-8B-instruct-nocontext", "llmp-llama-3-8B-instruct-context",
            "llmp-llama-3-70B-nocontext", "llmp-llama-3-70B-context",
            "llmp-llama-3-70B-instruct-nocontext", "llmp-llama-3-70B-instruct-context",
            "llmp-mixtral-8x7B-nocontext", "llmp-mixtral-8x7B-context",
            "llmp-mixtral-8x7B-instruct-nocontext", "llmp-mixtral-8x7B-instruct-context",
            "llmp-qwen2.5-0.5B-Instruct-nocontext", "llmp-qwen2.5-0.5B-Instruct-context",
            "llmp-qwen2.5-7B-Instruct-nocontext", "llmp-qwen2.5-7B-Instruct-context",
        ],
        "llama3": [
            "llmp-llama-3-8B-nocontext", "llmp-llama-3-8B-context",
            "llmp-llama-3-8B-instruct-nocontext", "llmp-llama-3-8B-instruct-context",
            "llmp-llama-3-70B-nocontext", "llmp-llama-3-70B-context",
            "llmp-llama-3-70B-instruct-nocontext", "llmp-llama-3-70B-instruct-context",
        ],
        "llmp-qwen": [
            "llmp-qwen2.5-0.5B-Instruct-nocontext", "llmp-qwen2.5-0.5B-Instruct-context",
            "llmp-qwen2.5-7B-Instruct-nocontext", "llmp-qwen2.5-7B-Instruct-context",
        ],
        "llmp-sample": [
            "llmp-qwen2.5-0.5B-Instruct-nocontext", "llmp-qwen2.5-0.5B-Instruct-context",
            "llmp-llama-3-8B-nocontext", "llmp-llama-3-8B-context",
        ],
        "mixtral": [
            "llmp-mixtral-8x7B-nocontext", "llmp-mixtral-8x7B-context",
            "llmp-mixtral-8x7B-instruct-nocontext", "llmp-mixtral-8x7B-instruct-context",
        ],
    }


# ============================================================================
# MODEL REGISTRY
# ============================================================================

def get_model_registry(sleep_between_requests: float = 0.0) -> Dict[str, Callable]:
    def _dlinear(model_type: str):
        from eventglucose.baselines.dlinear import DLinearForecaster
        return DLinearForecaster(model_type=model_type)

    def _transformer(model_type: str, use_context: bool):
        from eventglucose.baselines.transformers import TransformerForecaster
        return TransformerForecaster(model_type=model_type, use_context=use_context)

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

    def _direct_prompt(model: str, use_context: bool, token_cost: dict,
                       sleep_between_requests: float = 0.0,
                       fail_on_invalid: bool = False, batch_size: int = None):
        from eventglucose.baselines.direct_prompt import DirectPrompt
        return DirectPrompt(
            model=model,
            use_context=use_context,
            token_cost=token_cost,
            sleep_between_requests=sleep_between_requests,
            fail_on_invalid=fail_on_invalid,
            batch_size=batch_size,
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
        "random": _random,
        "oracle": _oracle,
        "chronos-tiny":  lambda: _chronos("tiny"),
        "chronos-mini":  lambda: _chronos("mini"),
        "chronos-small": lambda: _chronos("small"),
        "chronos-base":  lambda: _chronos("base"),
        "chronos-large": lambda: _chronos("large"),
        "moirai-small": lambda: _moirai("small"),
        "moirai-base":  lambda: _moirai("base"),
        "moirai-large": lambda: _moirai("large"),
        "lag-llama": _lag_llama,

        # DLinear / NLinear (LTSF-Linear)
        "dlinear": lambda: _dlinear("dlinear"),
        "nlinear": lambda: _dlinear("nlinear"),

        # Transformer models (no pretrained weights, trained on-the-fly)
        "itransformer":     lambda: _transformer("itransformer", False),
        "autoformer":       lambda: _transformer("autoformer",   False),
        "causal":           lambda: _transformer("causal",       False),
        "itransformer-ctx": lambda: _transformer("itransformer", True),
        "autoformer-ctx":   lambda: _transformer("autoformer",   True),
        "causal-ctx":       lambda: _transformer("causal",       True),

        "exp-smoothing": _exp_smoothing,
        "r-ets":   _r_ets,
        "r-arima": _r_arima,
        "timegen1": _timegen1,
    }

    openai_costs = {
        "gpt-4o":       {"input": 0.005,   "output": 0.015},
        "gpt-4o-mini":  {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo":{"input": 0.003,   "output": 0.006},
        "gpt-5-mini":   {"input": 0.0003,  "output": 0.0012},
    }

    for model_name in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]:
        registry[f"{model_name}-nocontext"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m, use_context=False,
            token_cost=openai_costs.get(m, {"input": 0.0, "output": 0.0}),
            sleep_between_requests=s,
        )
        registry[f"{model_name}-context"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m, use_context=True,
            token_cost=openai_costs.get(m, {"input": 0.0, "output": 0.0}),
            sleep_between_requests=s,
        )

    registry["gpt-5-mini-nocontext"] = lambda s=sleep_between_requests: _direct_prompt(
        model="gpt-5-mini", use_context=False,
        token_cost=openai_costs.get("gpt-5-mini", {"input": 0.0, "output": 0.0}),
        sleep_between_requests=s, batch_size=5,
    )
    registry["gpt-5-mini-context"] = lambda s=sleep_between_requests: _direct_prompt(
        model="gpt-5-mini", use_context=True,
        token_cost=openai_costs.get("gpt-5-mini", {"input": 0.0, "output": 0.0}),
        sleep_between_requests=s, batch_size=5,
    )

    for model_name in [
        "openrouter-llama-3-8b-instruct",
        "openrouter-llama-3-70b-instruct",
        "openrouter-mixtral-8x7b-instruct",
        "openrouter-llama-3.1-405b-instruct",
        "openrouter-qwen3-235b-a22b-instruct",
        "openrouter-gemini-2.5-flash",
        "openrouter-claude-3.5-haiku",
    ]:
        registry[f"{model_name}-nocontext"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m, use_context=False,
            token_cost={"input": 0.0, "output": 0.0}, sleep_between_requests=s,
        )
        registry[f"{model_name}-context"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m, use_context=True,
            token_cost={"input": 0.0, "output": 0.0}, sleep_between_requests=s,
        )

    registry["llama-3.1-405b-instruct-nocontext"] = registry["openrouter-llama-3.1-405b-instruct-nocontext"]
    registry["llama-3.1-405b-instruct-context"]   = registry["openrouter-llama-3.1-405b-instruct-context"]

    for _size in ["0.5B", "1.5B", "3B", "7B"]:
        _canonical = f"qwen2.5-{_size}-Instruct"
        for _alias in [_canonical, f"qwen2.5-{_size.lower()}-instruct"]:
            registry[f"{_alias}-nocontext"] = lambda m=_canonical, s=sleep_between_requests: _direct_prompt(
                model=m, use_context=False,
                token_cost={"input": 0.0, "output": 0.0}, sleep_between_requests=s,
            )
            registry[f"{_alias}-context"] = lambda m=_canonical, s=sleep_between_requests: _direct_prompt(
                model=m, use_context=True,
                token_cost={"input": 0.0, "output": 0.0}, sleep_between_requests=s,
            )

    claude_costs = {
        "claude-4.5-opus":   {"input": 0.015, "output": 0.075},
        "claude-4.5-sonnet": {"input": 0.003, "output": 0.015},
    }
    for model_name in ["claude-4.5-opus", "claude-4.5-sonnet"]:
        registry[f"{model_name}-nocontext"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m, use_context=False,
            token_cost=claude_costs.get(m, {"input": 0.0, "output": 0.0}),
            sleep_between_requests=s,
        )
        registry[f"{model_name}-context"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m, use_context=True,
            token_cost=claude_costs.get(m, {"input": 0.0, "output": 0.0}),
            sleep_between_requests=s,
        )

    ccsdk_costs = {
        "claude-sdk-haiku-4.5":  {"input": 0.000001, "output": 0.000005},
        "claude-sdk-sonnet-4.5": {"input": 0.000003, "output": 0.000015},
        "claude-sdk-opus-4.5":   {"input": 0.000005, "output": 0.000025},
    }
    for model_name in ["claude-sdk-haiku-4.5", "claude-sdk-sonnet-4.5", "claude-sdk-opus-4.5"]:
        registry[f"{model_name}-nocontext"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m, use_context=False, token_cost=ccsdk_costs[m],
            sleep_between_requests=s, fail_on_invalid=False, batch_size=None,
        )
        registry[f"{model_name}-context"] = lambda m=model_name, s=sleep_between_requests: _direct_prompt(
            model=m, use_context=True, token_cost=ccsdk_costs[m],
            sleep_between_requests=s, fail_on_invalid=False, batch_size=None,
        )

    for pred_len in [96]:
        registry[f"timellm-nocontext-pred{pred_len}"] = lambda pl=pred_len: _timellm(False, "etth1", pl)
        registry[f"timellm-context-pred{pred_len}"]   = lambda pl=pred_len: _timellm(True,  "etth1", pl)
        registry[f"unitime-nocontext-pred{pred_len}"] = lambda pl=pred_len: _unitime(False, "etth1", pl)
        registry[f"unitime-context-pred{pred_len}"]   = lambda pl=pred_len: _unitime(True,  "etth1", pl)
        registry[f"timellm-etth1-nocontext-pred{pred_len}"] = registry[f"timellm-nocontext-pred{pred_len}"]
        registry[f"timellm-etth1-context-pred{pred_len}"]   = registry[f"timellm-context-pred{pred_len}"]
        registry[f"unitime-etth1-nocontext-pred{pred_len}"] = registry[f"unitime-nocontext-pred{pred_len}"]
        registry[f"unitime-etth1-context-pred{pred_len}"]   = registry[f"unitime-context-pred{pred_len}"]

    llmp_llms = [
        "llama-3-8B", "llama-3-8B-instruct",
        "llama-3-70B", "llama-3-70B-instruct",
        "mixtral-8x7B", "mixtral-8x7B-instruct",
        "qwen2.5-0.5B-Instruct", "qwen2.5-7B-Instruct",
    ]
    for llm_type in llmp_llms:
        registry[f"llmp-{llm_type}-nocontext"] = lambda lt=llm_type: _llmp(lt, False)
        registry[f"llmp-{llm_type}-context"]   = lambda lt=llm_type: _llmp(lt, True)

    return registry


# ============================================================================
# CLI Functions
# ============================================================================

def list_tasks():
    task_registry = get_task_registry()

    print("\n" + "=" * 80)
    print("AVAILABLE TASKS")
    print("=" * 80)
    print(f"\nTotal: {len(task_registry)} tasks\n")
    print("Tip: You can use task alias suffixes:")
    print("  - *_context    → expands to *_Profile + *_BasicEventInfo")
    print("  - *_newcontext → expands to *_StandardEventInfo + *_DetailedEventInfo")
    print("  - *_allcontext → expands to all 4 context levels")
    print("  - *_nocontext  → expands to *_NoCtx\n")

    categories = {
        "Base Tasks": [],
        "Diet Profile Tasks": [],
        "Diet BasicEventInfo Tasks": [],
        "Diet StandardEventInfo Tasks": [],
        "Diet DetailedEventInfo Tasks": [],
        "Diet No Context Tasks": [],
        "Exercise Profile Tasks": [],
        "Exercise BasicEventInfo Tasks": [],
        "Exercise StandardEventInfo Tasks": [],
        "Exercise DetailedEventInfo Tasks": [],
        "Exercise No Context Tasks": [],
        "No-Event Tasks": [],
    }

    for name in sorted(task_registry.keys()):
        if "Base" in name or "test" in name:
            categories["Base Tasks"].append(name)
        elif "NoEvent" in name or (name == "EventCGMTask_withEvent_withLag"):
            categories["No-Event Tasks"].append(name)
        elif "Exercise" in name and "DetailedEventInfo" in name:
            categories["Exercise DetailedEventInfo Tasks"].append(name)
        elif "Exercise" in name and "StandardEventInfo" in name:
            categories["Exercise StandardEventInfo Tasks"].append(name)
        elif "Exercise" in name and "BasicEventInfo" in name:
            categories["Exercise BasicEventInfo Tasks"].append(name)
        elif "Exercise" in name and "Profile" in name:
            categories["Exercise Profile Tasks"].append(name)
        elif "Exercise" in name and "NoCtx" in name:
            categories["Exercise No Context Tasks"].append(name)
        elif "Diet" in name and "DetailedEventInfo" in name:
            categories["Diet DetailedEventInfo Tasks"].append(name)
        elif "Diet" in name and "StandardEventInfo" in name:
            categories["Diet StandardEventInfo Tasks"].append(name)
        elif "Diet" in name and "BasicEventInfo" in name:
            categories["Diet BasicEventInfo Tasks"].append(name)
        elif "Diet" in name and "Profile" in name:
            categories["Diet Profile Tasks"].append(name)
        elif "Diet" in name and "NoCtx" in name:
            categories["Diet No Context Tasks"].append(name)

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

    Supported variants:
        - context    -> Profile + BasicEventInfo
        - newcontext -> StandardEventInfo + DetailedEventInfo
        - allcontext -> all 4 context levels
        - nocontext  -> NoCtx
    """
    if variant == "context":
        return [f"{base}_Profile", f"{base}_BasicEventInfo"]
    if variant == "newcontext":
        return [f"{base}_StandardEventInfo", f"{base}_DetailedEventInfo"]
    if variant == "allcontext":
        return [
            f"{base}_Profile",
            f"{base}_BasicEventInfo",
            f"{base}_StandardEventInfo",
            f"{base}_DetailedEventInfo",
        ]
    if variant == "nocontext":
        return [f"{base}_NoCtx"]
    raise ValueError(f"Unknown task variant: {variant}")


def expand_task_names(task_specs: List[str]) -> List[str]:
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

        expanded.append(spec)

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
    return ("context" in model_name) and ("nocontext" not in model_name)


def expand_model_names(model_names: List[str]) -> List[str]:
    model_type_registry = get_model_type_registry()
    expanded = []

    for name in model_names:
        if name in model_type_registry:
            expanded.extend(model_type_registry[name])
        else:
            expanded.append(name)

    seen = set()
    result = []
    for item in expanded:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def list_model_types():
    model_type_registry = get_model_type_registry()

    print("\n" + "=" * 80)
    print("AVAILABLE MODEL TYPES")
    print("=" * 80)
    print(f"\nTotal: {len(model_type_registry)} model types\n")

    categories = {
        "Foundation Models": [],
        "LTSF-Linear Models": [],
        "Transformer Models (on-the-fly)": [],
        "DirectPrompt Models": [],
        "Multimodal Models": [],
        "Statistical Models": [],
        "Naive Baselines": [],
        "Comprehensive Groups": [],
    }

    for name in sorted(model_type_registry.keys()):
        if name in ("linear",):
            categories["LTSF-Linear Models"].append(name)
        elif name in ("transformer", "transformer-ctx", "transformer-all"):
            categories["Transformer Models (on-the-fly)"].append(name)
        elif "chronos" in name or "moirai" in name or "foundation" in name or name in ["lag-llama", "timegen1"]:
            categories["Foundation Models"].append(name)
        elif "gpt" in name and "llmp" not in name:
            categories["DirectPrompt Models"].append(name)
        elif "claude" in name and "llmp" not in name:
            categories["DirectPrompt Models"].append(name)
        elif "openrouter" in name and "llmp" not in name:
            categories["DirectPrompt Models"].append(name)
        elif "timellm" in name or "unitime" in name or "multimodal" in name:
            categories["Multimodal Models"].append(name)
        elif "llmp" in name or "llama" in name or "mixtral" in name:
            categories["DirectPrompt Models"].append(name)
        elif "qwen" in name:
            categories["DirectPrompt Models"].append(name)
        elif name in ["r-ets", "r-arima", "exp-smoothing", "statistical"]:
            categories["Statistical Models"].append(name)
        elif name == "naive":
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
    print("  --model chronos              # Run all 5 Chronos variants")
    print("  --model gpt-4o-context       # Run GPT-4o with context")
    print("  --model foundation-all       # Run all foundation models")
    print("=" * 80 + "\n")


def list_models():
    model_registry = get_model_registry(sleep_between_requests=0.0)

    print("\n" + "=" * 80)
    print("AVAILABLE INDIVIDUAL MODELS")
    print("=" * 80)
    print(f"\nTotal: {len(model_registry)} individual model configurations\n")

    categories = {
        "Naive Baselines": [],
        "Chronos Models": [],
        "Moirai Models": [],
        "Lag-Llama": [],
        "DLinear / NLinear": [],
        "Transformer (on-the-fly)": [],
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
        elif name in ["dlinear", "nlinear"]:
            categories["DLinear / NLinear"].append(name)
        elif name in ["itransformer", "autoformer", "causal",
                      "itransformer-ctx", "autoformer-ctx", "causal-ctx"]:
            categories["Transformer (on-the-fly)"].append(name)
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


def _patch_task_presample(base_class: type, pre_sampled_folder: str) -> type:
    """
    Return a subclass of base_class that loads from a pre-sampled folder and
    selects rows deterministically by seed (seed 1 → row 0, seed 2 → row 1, …).

    Uses _PreSampleMeta so the patched class is picklable across spawn-mode
    multiprocessing workers — no need to force max_parallel=1.
    """
    from eventglucose.tasks.eventglucose_tasks import make_presample_cls
    return make_presample_cls(base_class, pre_sampled_folder)


def _patch_task_format(base_class: type, prompt_time_format: str, prompt_output_format: str) -> type:
    """
    Return a subclass of base_class with prompt format overrides injected into get_task_config().

    The patched class is registered in the base class's own module so that
    Python's multiprocessing pickle can locate it by (module, qualname).
    """
    unique_name = f"_Patched_{base_class.__name__}_{prompt_time_format}_{prompt_output_format}"

    _time_fmt = prompt_time_format
    _out_fmt = prompt_output_format

    def _get_task_config(self):
        cfg = base_class.get_task_config(self)
        cfg["prompt_time_format"] = _time_fmt
        cfg["prompt_output_format"] = _out_fmt
        return cfg

    PatchedClass = type(unique_name, (base_class,), {"get_task_config": _get_task_config})

    target_module = sys.modules[base_class.__module__]
    setattr(target_module, unique_name, PatchedClass)
    PatchedClass.__module__ = base_class.__module__
    PatchedClass.__qualname__ = unique_name
    PatchedClass.__name__ = base_class.__name__

    return PatchedClass


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
    prompt_time_format: str | None = None,
    prompt_output_format: str | None = None,
    pre_sampled_dir: str | None = None,
):
    task_registry = get_task_registry()
    model_registry = get_model_registry(sleep_between_requests=sleep_between_requests)
    model_type_registry = get_model_type_registry()

    output_folder = output_folder or os.environ.get("LOCAL_RESULTS_FOLDER", "./results")

    original_task_names = task_names.copy()
    task_names = expand_task_names(task_names)
    original_model_names = model_names.copy()
    model_names = expand_model_names(model_names)

    expanded_types = [
        f"{name} → {len(model_type_registry[name])} configs"
        for name in original_model_names
        if name in model_type_registry
    ]

    invalid_tasks = [t for t in task_names if t not in task_registry]
    if invalid_tasks:
        print(f"Error: Invalid task names: {invalid_tasks}")
        print("Use --list-tasks to see available tasks")
        print("\nIf you intended a context alias, use one of:")
        print("  - *_context    (expands to *_Profile, *_BasicEventInfo)")
        print("  - *_newcontext (expands to *_StandardEventInfo, *_DetailedEventInfo)")
        print("  - *_allcontext (expands to all 4 context levels)")
        print("  - *_nocontext  (expands to *_NoCtx)")
        return

    invalid_models = [m for m in model_names if m not in model_registry]
    if invalid_models:
        print(f"Error: Invalid model names: {invalid_models}")
        print("Use --list-models to see available models")
        print("Use --list-model-types to see available model types")
        return

    task_classes = [task_registry[name] for name in task_names]

    # Apply pre-sampled folder override if specified (applied before prompt patching)
    if pre_sampled_dir is not None:
        task_classes = [_patch_task_presample(tc, pre_sampled_dir) for tc in task_classes]
        print(f"\nPre-sampled mode: loading fixed instances from {pre_sampled_dir}")
        print("  Row selection: seed 1 → row 0, seed 2 → row 1, … (deterministic)")

    # Apply prompt format overrides if specified
    if prompt_time_format is not None or prompt_output_format is not None:
        time_fmt = prompt_time_format or "time_value_pairs"
        out_fmt = prompt_output_format or "timestamp"
        task_classes = [_patch_task_format(tc, time_fmt, out_fmt) for tc in task_classes]
        print(f"\nPrompt format override: input={time_fmt}, output={out_fmt}")

    print("\n" + "=" * 80)
    print(f"RUNNING EVALUATION: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 80)
    print(f"\nTasks ({len(task_classes)}):")
    for name in task_names:
        print(f"  - {name}")

    expanded_task_aliases = [
        name for name in original_task_names
        if re.match(r"^(.*?)(?:[_-])(context|newcontext|allcontext|nocontext)$", name)
    ]
    if expanded_task_aliases:
        print(f"\nTask Alias Expansion:")
        for alias in expanded_task_aliases:
            base, variant = re.match(r"^(.*?)(?:[_-])(context|newcontext|allcontext|nocontext)$", alias).group(1, 2)
            print(f"  - {alias} → {', '.join(_task_alias_variants(base, variant))}")

    if expanded_types:
        print(f"\nModel Type Expansion:")
        for expansion in expanded_types:
            print(f"  - {expansion}")

    print(f"\nModels ({len(model_names)}):")
    for name in model_names:
        print(f"  - {name}")

    print(f"\nParameters:")
    print(f"  - n_instances:  {n_instances}")
    print(f"  - n_samples:    {n_samples}")
    print(f"  - output_folder:{output_folder}")
    print(f"  - max_parallel: {max_parallel}")
    if sleep_between_requests > 0:
        print(f"  - sleep_between_requests: {sleep_between_requests}s")
    print("=" * 80 + "\n")

    results = {}

    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"Running model: {model_name}")
        print('=' * 60)

        model_factory = model_registry[model_name]
        model = model_factory()

        filtered_task_classes = task_classes
        if skip_done:
            try:
                from scripts.run_result_check import check_task_model_status
            except Exception as e:
                print(f"Warning: failed to import scripts.run_result_check ({e}); continuing without skip-done.")
                check_task_model_status = None

            if check_task_model_status is not None:
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
    print(f"\nCompleted: {len(results)}/{len(model_names)} models")
    print(f"Results directory: {output_folder}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run individual model and task combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--list-tasks",       action="store_true", help="List all available task names")
    parser.add_argument("--list-models",      action="store_true", help="List all available individual model names")
    parser.add_argument("--list-model-types", action="store_true", help="List all available model types")

    parser.add_argument("--task",      nargs="+", type=str, help="Task name(s) to evaluate")
    parser.add_argument("--all-tasks", action="store_true",  help="Run all available tasks")
    parser.add_argument("--model",     nargs="+", type=str, help="Model name(s) or type(s)")

    parser.add_argument("--n-instances",  type=int,   default=10,   help="Number of instances (default: 10)")
    parser.add_argument("--n-samples",    type=int,   default=25,   help="Samples per instance (default: 25)")
    parser.add_argument("--output",       type=str,
                        default=os.environ.get("LOCAL_RESULTS_FOLDER", "./results"),
                        help="Output directory (default: $LOCAL_RESULTS_FOLDER or ./results)")
    parser.add_argument("--max-parallel", type=int,   default=None, help="Max parallel workers")
    parser.add_argument("--skip-cache-miss", action="store_true",   help="Skip on cache miss")
    parser.add_argument("--skip-done",       action="store_true",   help="Skip already-completed task/model pairs")
    parser.add_argument("--results-roots",   nargs="+", type=str,   default=None,
                        help="Extra result roots to search when --skip-done is active")
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.0,
        help="Sleep between API requests to avoid rate limiting (default: 0.0)",
    )
    parser.add_argument(
        "--prompt-time-format",
        type=str,
        default=None,
        choices=["time_value_pairs", "value_sequence", "llmtime", "markdown_table"],
        help="Override the input time series format in the prompt. If not set, the task default is used.",
    )
    parser.add_argument(
        "--prompt-output-format",
        type=str,
        default=None,
        choices=["timestamp", "index"],
        help="Override the forecast output format in the prompt. If not set, the task default is used.",
    )
    parser.add_argument(
        "--pre-sampled-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Path to a folder of pre-sampled publish_clean PKLs "
            "(e.g. _WorkSpace/Data/EventGlucose/publish_clean). "
            "Tasks load fixed rows instead of resampling; seed 1 → row 0, seed 2 → row 1, … "
            "n_instances should not exceed the number of rows in the PKL (typically 10)."
        ),
    )

    args = parser.parse_args()

    if args.list_tasks:
        list_tasks()
        return
    if args.list_models:
        list_models()
        return
    if args.list_model_types:
        list_model_types()
        return

    if args.all_tasks:
        task_registry = get_task_registry()
        task_names = sorted(task_registry.keys())
        print(f"\n--all-tasks: running all {len(task_names)} tasks\n")
    else:
        task_names = args.task

    if not args.all_tasks and not task_names:
        parser.error("--task is required unless you use --all-tasks.")
    if not args.model:
        parser.error("--model is required (or use --list-models/--list-model-types).")

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
        prompt_time_format=args.prompt_time_format,
        prompt_output_format=args.prompt_output_format,
        pre_sampled_dir=args.pre_sampled_dir,
    )


if __name__ == "__main__":
    main()
