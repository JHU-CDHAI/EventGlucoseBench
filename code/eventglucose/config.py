"""
Configuration variables for the benchmark

All configuration is sourced from environment variables.
Before using this module, ensure you have sourced env.sh:
    source env.sh

"""

import os
import sys
import subprocess
from pathlib import Path


# ============================================================================
# Storage Paths (sourced from env.sh via CIK_* environment variables)
# ============================================================================

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _as_workspace_path(p: str) -> Path:
    """
    Convert an env-provided path to an absolute Path.

    env.sh commonly sets *relative* paths like `_WorkSpace/Model`. In notebooks,
    the process CWD is often `notebooks/`, which would otherwise resolve these
    relative paths incorrectly. We resolve relative paths against the workspace
    root (repo root).
    """
    pp = Path(p).expanduser()
    if not pp.is_absolute():
        pp = (_WORKSPACE_ROOT / pp).resolve()
    return pp


def _load_env_from_shell() -> bool:
    """
    Best-effort auto-loader for env.sh.

    This mirrors `code/scripts/run_individual.py`:
    - Find env.sh by walking up from this file
    - Source it in a bash subprocess
    - Merge exported variables into this process environment (without overriding)

    This is critical for Jupyter/VS Code kernels which often don't inherit
    `source env.sh` from your interactive terminal.
    """
    # Avoid re-loading repeatedly within the same process.
    if os.environ.get("_EVENTGLUCOSE_SUBPROCESS") == "1":
        return True

    current_path = Path(__file__).resolve()
    for _ in range(3):
        current_path = current_path.parent
        env_file = current_path / "env.sh"
        if not env_file.exists():
            continue

        try:
            cmd = f'source "{env_file}" && env'
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                check=True,
                cwd=str(current_path),
            )
            for line in result.stdout.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key not in os.environ:
                        os.environ[key] = value

            # Keep consistent with existing scripts: used as a lightweight sentinel
            # to prevent repeated loads and noisy logs.
            os.environ["_EVENTGLUCOSE_SUBPROCESS"] = "1"
            return True
        except Exception:
            return False

    return False


def _require_env(var_name: str, description: str) -> str:
    """Get required environment variable or raise helpful error."""
    value = os.environ.get(var_name)
    if not value:
        # Jupyter kernels often don't see your terminal's `source env.sh`.
        _load_env_from_shell()
        value = os.environ.get(var_name)
    if not value:
        print(f"ERROR: Required environment variable '{var_name}' is not set.", file=sys.stderr)
        print(f"       ({description})", file=sys.stderr)
        print(f"\nPlease run: source env.sh", file=sys.stderr)
        raise EnvironmentError(f"Missing required environment variable: {var_name}")
    return value

# Model storage (maps to LOCAL_MODELINSTANCE_STORE from env.sh)
MODEL_STORAGE_PATH = _as_workspace_path(_require_env("CIK_MODEL_STORE", "Model weights storage"))
MODEL_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

# Data storage (maps to LOCAL_WORKSPACE from env.sh)
DATA_STORAGE_PATH = _as_workspace_path(_require_env("CIK_DATA_STORE", "Main data storage"))
DATA_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

# Dataset-specific storage paths
DOMINICK_STORAGE_PATH = os.environ.get(
    "CIK_DOMINICK_STORE", str(DATA_STORAGE_PATH / "dominicks")
)
TRAFFIC_STORAGE_PATH = os.environ.get(
    "CIK_TRAFFIC_DATA_STORE", str(DATA_STORAGE_PATH / "traffic_data")
)
HF_CACHE_DIR = os.environ.get("HF_HOME", str(DATA_STORAGE_PATH / "hf_cache"))

# Legacy compatibility (for old code that uses PROJ_* variables)
PROJ_DATA_FOLDER = str(DATA_STORAGE_PATH)
PROJ_MODEL_OUTPUT_FOLDER = str(MODEL_STORAGE_PATH)

# ============================================================================
# Evaluation Configuration (sourced from env.sh)
# ============================================================================
# CIK_RESULT_CACHE: Cache for model predictions to avoid recomputation
# CIK_METRIC_SCALING_CACHE: Cache for metric scaling factors
# CIK_METRIC_COMPUTE_VARIANCE: If set to "true", compute metric variance estimates

DEFAULT_N_SAMPLES = 50

RESULT_CACHE_PATH = _as_workspace_path(
    os.environ.get("CIK_RESULT_CACHE", str(DATA_STORAGE_PATH / "_Inference_Cache"))
)
RESULT_CACHE_PATH.mkdir(parents=True, exist_ok=True)

METRIC_SCALING_CACHE_PATH = str(
    _as_workspace_path(
        os.environ.get("CIK_METRIC_SCALING_CACHE", str(DATA_STORAGE_PATH / "_Metric_Scaling_Cache"))
    )
)

COMPUTE_METRIC_VARIANCE = os.environ.get("CIK_METRIC_COMPUTE_VARIANCE", "").lower() == "true"

# ============================================================================
# API Configuration (sourced from env.sh)
# ============================================================================

# OpenAI Configuration
# CIK_OPENAI_USE_AZURE: Use Azure OpenAI instead of OpenAI API ("True"/"False")
# CIK_OPENAI_API_KEY: OpenAI or Azure OpenAI API key (required for OpenAI models)
# CIK_OPENAI_API_VERSION: Azure API version (only for Azure)
# CIK_OPENAI_AZURE_ENDPOINT: Azure endpoint URL (only for Azure)
OPENAI_USE_AZURE = os.environ.get("CIK_OPENAI_USE_AZURE", "False").lower() == "true"
OPENAI_API_KEY = os.environ.get("CIK_OPENAI_API_KEY")  # No default - sourced from env.sh
OPENAI_API_VERSION = os.environ.get("CIK_OPENAI_API_VERSION") or None
OPENAI_AZURE_ENDPOINT = os.environ.get("CIK_OPENAI_AZURE_ENDPOINT") or None

# Llama-405b Configuration (for vLLM server)
# CIK_LLAMA31_405B_URL: vLLM server URL
# CIK_LLAMA31_405B_API_KEY: API key for Llama server
LLAMA31_405B_URL = os.environ.get("CIK_LLAMA31_405B_URL") or None
LLAMA31_405B_API_KEY = os.environ.get("CIK_LLAMA31_405B_API_KEY") or None

# Nixtla TimeGEN Configuration
# CIK_NIXTLA_BASE_URL: Azure API URL for TimeGEN
# CIK_NIXTLA_API_KEY: Nixtla API key
NIXTLA_BASE_URL = os.environ.get("CIK_NIXTLA_BASE_URL") or None
NIXTLA_API_KEY = os.environ.get("CIK_NIXTLA_API_KEY")  # No default - sourced from env.sh

# Anthropic Claude Configuration
# CIK_ANTHROPIC_API_KEY: Anthropic API key for Claude models
ANTHROPIC_API_KEY = os.environ.get("CIK_ANTHROPIC_API_KEY") or None


# ============================================================================
# Task Configuration
# ============================================================================
DATA_LTS_FOLDER = f"{PROJ_DATA_FOLDER}/EventGlucose/@task"

SCENARIAO_FEATURE_INCLUDE_DICT = {'Diet5Min': [], 'Med5Min': [], 'Exercise5Min': []}