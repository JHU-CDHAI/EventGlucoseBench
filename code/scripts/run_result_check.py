"""
Utilities for checking whether model/task results are already saved on disk.

This mirrors the results layout produced by `code/scripts/run_individual.py`:
    Standard layout:
      {results_root}/{model_name}/{task_name}/{instance_id}/data_summary.txt
    Legacy single-instance layout (rare):
      {results_root}/{model_name}/{task_name}/data_summary.txt

Each `instance_id` directory (typically 1..N) is one "instance" (default: 10 instances).
Each `complete_data.pkl` contains prediction samples with a recorded `n_samples`
(default: 5 samples per instance).
"""

from __future__ import annotations

import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal


# ============================================================================
# Auto-load environment from env.sh (same approach as run_individual.py)
# ============================================================================
def load_env_from_shell() -> bool:
    """
    Automatically find and load environment variables from env.sh.

    This allows the script to work without manually running 'source env.sh' first.
    """
    current_path = Path(__file__).resolve()

    for _ in range(3):
        current_path = current_path.parent
        env_file = current_path / "env.sh"
        if env_file.exists():
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
            except subprocess.CalledProcessError:
                return False
    return False


# Load environment and ensure `code/` is on sys.path for optional imports.
load_env_from_shell()
sys.path.insert(0, str(Path(__file__).parent.parent))


def _infer_model_name(model_or_name: Union[str, Callable[..., Any], Any]) -> str:
    """
    Infer the model directory name used under results root.

    Preferred usage is passing the model name string (e.g., "chronos-small").
    If a callable/object is provided, we try common attributes.
    """
    if isinstance(model_or_name, str):
        return model_or_name

    for attr in ("model_name", "name", "id"):
        v = getattr(model_or_name, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()

    v = getattr(model_or_name, "__name__", None)
    if isinstance(v, str) and v.strip():
        return v.strip()

    raise ValueError(
        "Could not infer model name from `model_or_name`. "
        "Pass the model name string used by run_individual.py (e.g., 'chronos-small')."
    )


def get_default_results_root() -> Path:
    """
    Match run_individual.py default: $LOCAL_RESULTS_FOLDER or ./results
    """
    return Path(os.environ.get("LOCAL_RESULTS_FOLDER", "./results")).expanduser().resolve()


def _read_stored_n_samples(instance_dir: Path) -> Tuple[Optional[int], Optional[str]]:
    """
    Returns (n_samples, error_message) by parsing `data_summary.txt`.

    We intentionally avoid unpickling `complete_data.pkl` by default to keep this checker
    robust across environments (some pickle loads may crash due to dependency mismatches).
    """
    summary_path = instance_dir / "data_summary.txt"
    if not summary_path.exists():
        return None, "missing_summary"
    try:
        summary_text = summary_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"^Number of samples:\s*(\d+)\s*$", summary_text, flags=re.MULTILINE)
        if m:
            return int(m.group(1)), None
        return None, "unparseable_summary"
    except Exception as e:
        return None, f"summary_read_error: {e}"


def _list_instance_dirs(task_dir: Path) -> List[Path]:
    """
    Return instance directories for a task.

    Supported layouts:
      1) Standard: task_dir/<int>/data_summary.txt
      2) Legacy:   task_dir/data_summary.txt (treated as one instance)
    """
    if not task_dir.exists():
        return []

    numeric_subdirs: List[Tuple[int, Path]] = []
    for p in task_dir.iterdir():
        if not p.is_dir():
            continue
        try:
            idx = int(p.name)
        except ValueError:
            continue
        numeric_subdirs.append((idx, p))

    if numeric_subdirs:
        return [p for _, p in sorted(numeric_subdirs, key=lambda t: t[0])]

    if (task_dir / "data_summary.txt").exists():
        return [task_dir]

    return []


def _resolve_model_location(model_name: str, results_roots: List[Path]) -> Path:
    """
    Choose the first results root that contains a directory named `model_name`.
    Falls back to the first root if none contain it.
    """
    if not results_roots:
        raise ValueError("results_roots must be non-empty")
    model_name = str(model_name).strip()
    for root in results_roots:
        if (root / model_name).is_dir():
            return root
    return results_roots[0]


Status = Literal["done", "not_enough", "no"]


def check_task_model_status(
    *,
    model_name: str,
    task_name: str,
    results_roots: Optional[List[Union[str, Path]]] = None,
    n_instances: int = 10,
    n_samples: int = 5,
) -> Tuple[Status, Dict[str, Any]]:
    """
    Check whether a model/task already has results saved on disk.

    Returns:
      (status, report)
        - status: "done" / "not_enough" / "no" (same semantics as root-level run_checker.py)
        - report: diagnostics (resolved root, missing counts, etc.)
    """
    roots = (
        [Path(p).expanduser().resolve() for p in results_roots]
        if results_roots
        else [get_default_results_root()]
    )
    chosen_root = _resolve_model_location(model_name, roots)
    task_dir = chosen_root / model_name / task_name

    if not task_dir.exists():
        return "no", {
            "results_root": str(chosen_root),
            "model_name": model_name,
            "task_name": task_name,
            "task_dir": str(task_dir),
            "reason": "missing_task_dir",
        }

    instance_dirs = _list_instance_dirs(task_dir)
    if not instance_dirs:
        return "no", {
            "results_root": str(chosen_root),
            "model_name": model_name,
            "task_name": task_name,
            "task_dir": str(task_dir),
            "reason": "no_instances_found",
        }

    if len(instance_dirs) < int(n_instances):
        return "not_enough", {
            "results_root": str(chosen_root),
            "model_name": model_name,
            "task_name": task_name,
            "task_dir": str(task_dir),
            "found_instances": len(instance_dirs),
            "required_instances": int(n_instances),
        }

    for inst_dir in instance_dirs[: int(n_instances)]:
        if (inst_dir / "error").exists():
            return "no", {
                "results_root": str(chosen_root),
                "model_name": model_name,
                "task_name": task_name,
                "task_dir": str(task_dir),
                "reason": f"error_marker: {inst_dir.name}",
            }

        stored_n, err = _read_stored_n_samples(inst_dir)
        if stored_n is None:
            return "no", {
                "results_root": str(chosen_root),
                "model_name": model_name,
                "task_name": task_name,
                "task_dir": str(task_dir),
                "reason": f"missing_or_unparseable_summary: {inst_dir.name}",
                "error": err,
            }
        if stored_n < int(n_samples):
            return "not_enough", {
                "results_root": str(chosen_root),
                "model_name": model_name,
                "task_name": task_name,
                "task_dir": str(task_dir),
                "reason": f"insufficient_samples: {inst_dir.name}",
                "stored_n_samples": stored_n,
                "required_n_samples": int(n_samples),
            }

    return "done", {
        "results_root": str(chosen_root),
        "model_name": model_name,
        "task_name": task_name,
        "task_dir": str(task_dir),
    }


def check_default_instances_and_samples(
    *,
    model_or_name: Union[str, Callable[..., Any], Any],
    task_name: str,
    results_root: Optional[Union[str, Path]] = None,
    n_instances: int = 10,
    n_samples: int = 5,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Check whether a model/task already has results saved on disk.

    Expected layout:
        {results_root}/{model_name}/{task_name}/{seed}/complete_data.pkl

    For each seed in 1..n_instances, it verifies:
    - the seed directory exists
    - `complete_data.pkl` exists and is readable (or summary is readable)
    - the stored `n_samples` is at least `n_samples`

    Args:
        model_or_name: model name string (recommended) or a callable/object used to infer the model name
        task_name: task class name folder (e.g., "EventCGMTask_D1_Age18_Diet_Ontime_NoCtx")
        results_root: base results directory (default: $LOCAL_RESULTS_FOLDER or ./results)
        n_instances: expected number of seeds/instances (default: 10)
        n_samples: expected samples per instance (default: 5)

    Returns:
        (ok, report) where ok is True iff all required seeds exist and have >= n_samples stored.
    """
    model_name = _infer_model_name(model_or_name)
    roots: List[Union[str, Path]] = [results_root] if results_root is not None else None  # type: ignore[assignment]
    status, report = check_task_model_status(
        model_name=model_name,
        task_name=task_name,
        results_roots=roots,  # type: ignore[arg-type]
        n_instances=n_instances,
        n_samples=n_samples,
    )
    return status == "done", report

"""
python run_result_check.py --matrix-csv task_model_matrix.csv --results-root _Workspace/Result --out-csv task_model_checklist.csv --n-instances 10 --n-samples 5
"""