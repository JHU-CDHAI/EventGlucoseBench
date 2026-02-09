
import os
import sys
import subprocess
from pathlib import Path

# ============================================================================
# ============================================================================
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

def _as_workspace_path(p: str) -> Path:
    pp = Path(p).expanduser()
    if not pp.is_absolute():
        pp = (_WORKSPACE_ROOT / pp).resolve()
    return pp

def _load_env_from_shell() -> bool:
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

            os.environ["_EVENTGLUCOSE_SUBPROCESS"] = "1"
            return True
        except Exception:
            return False

    return False

def _require_env(var_name: str, description: str) -> str:
    """Get required environment variable or raise helpful error."""
    value = os.environ.get(var_name)
    if not value:
        _load_env_from_shell()
        value = os.environ.get(var_name)
    if not value:
        print(f"ERROR: Required environment variable '{var_name}' is not set.", file=sys.stderr)
        print(f"       ({description})", file=sys.stderr)
        print(f"\nPlease run: source env.sh", file=sys.stderr)
        raise EnvironmentError(f"Missing required environment variable: {var_name}")
    return value

MODEL_STORAGE_PATH = _as_workspace_path(_require_env("CIK_MODEL_STORE", "Model weights storage"))
MODEL_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

DATA_STORAGE_PATH = _as_workspace_path(_require_env("CIK_DATA_STORE", "Main data storage"))
DATA_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

DOMINICK_STORAGE_PATH = os.environ.get(
    "CIK_DOMINICK_STORE", str(DATA_STORAGE_PATH / "dominicks")
)
TRAFFIC_STORAGE_PATH = os.environ.get(
    "CIK_TRAFFIC_DATA_STORE", str(DATA_STORAGE_PATH / "traffic_data")
)
HF_CACHE_DIR = os.environ.get("HF_HOME", str(DATA_STORAGE_PATH / "hf_cache"))

PROJ_DATA_FOLDER = str(DATA_STORAGE_PATH)
PROJ_MODEL_OUTPUT_FOLDER = str(MODEL_STORAGE_PATH)

# ============================================================================
# ============================================================================
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
# ============================================================================
OPENAI_USE_AZURE = os.environ.get("CIK_OPENAI_USE_AZURE", "False").lower() == "true"
OPENAI_API_KEY = os.environ.get("CIK_OPENAI_API_KEY")
OPENAI_API_VERSION = os.environ.get("CIK_OPENAI_API_VERSION") or None
OPENAI_AZURE_ENDPOINT = os.environ.get("CIK_OPENAI_AZURE_ENDPOINT") or None

LLAMA31_405B_URL = os.environ.get("CIK_LLAMA31_405B_URL") or None
LLAMA31_405B_API_KEY = os.environ.get("CIK_LLAMA31_405B_API_KEY") or None

NIXTLA_BASE_URL = os.environ.get("CIK_NIXTLA_BASE_URL") or None
NIXTLA_API_KEY = os.environ.get("CIK_NIXTLA_API_KEY")

ANTHROPIC_API_KEY = os.environ.get("CIK_ANTHROPIC_API_KEY") or None

# ============================================================================
# ============================================================================
DATA_LTS_FOLDER = f"{PROJ_DATA_FOLDER}/EventGlucose/@task"

SCENARIAO_FEATURE_INCLUDE_DICT = {'Diet5Min': [], 'Med5Min': [], 'Exercise5Min': []}