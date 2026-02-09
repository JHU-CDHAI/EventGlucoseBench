"""
2025-10-28: This File and Functions are not used, maybe from original CIK codebase
It Will be deleted after all corresponding experiment scripts and notebooks are checked
"""
import logging
import os
import matplotlib

# Force non-interactive backend for batch runs to avoid GUI windows and reduce memory usage.
# Enable with: export CIK_NO_PLOT_DISPLAY=1
if os.environ.get("CIK_NO_PLOT_DISPLAY", "0") == "1":
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
import pickle
import traceback
from pprint import pprint

from collections import defaultdict
from functools import partial
from pathlib import Path

from . import ALL_TASKS
from .config import DEFAULT_N_SAMPLES
from .utils.cache import ResultCache, CacheMissError

# Fix CUDA multiprocessing issue: use 'spawn' instead of 'fork'
# This is required for CUDA to work correctly with multiprocessing on Linux
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

logger = logging.getLogger("Evaluation")


def plot_forecast_univariate(task, samples, path, return_fig=False):
    """
    Plot the first variable of a forecast.

    Parameters:
    -----------
    task: a BaseTask
        The task associated with the forecast
    samples: np.array
        The forecast of shape [samples, time dimension, number of variables]
    path: Pathlike
        Directory in which to save the figure

    """
    samples = samples[:, :, -1]

    future_timesteps = task.future_time.index
    if isinstance(future_timesteps, pd.PeriodIndex):
        future_timesteps = future_timesteps.to_timestamp()

    fig = task.plot()
    ax = fig.gca()

    for zorder, quant, color, label in [
        [1, 0.05, (0.75, 0.75, 1), "5%-95%"],
        [2, 0.10, (0.25, 0.25, 1), "10%-90%"],
        [3, 0.25, (0, 0, 0.75), "25%-75%"],
    ]:
        lower_quantile = np.quantile(samples, quant, axis=0).astype(float)
        upper_quantile = np.quantile(samples, 1 - quant, axis=0).astype(float)

        ax.fill_between(
            future_timesteps,
            lower_quantile,
            upper_quantile,
            facecolor=color,
            interpolate=True,
            label=label,
            zorder=zorder,
        )

        if quant == 0.05:
            min_quantile_value = np.min(lower_quantile)
            max_quantile_value = np.max(upper_quantile)

    ax.plot(
        future_timesteps,
        np.quantile(samples, 0.5, axis=0),
        color=(0.5, 0.5, 0.5),
        linewidth=3,
        label="50%",
        zorder=4,
    )

    target_name = task.past_time.columns[-1]
    # For glucose tasks, maintain the 0-450 mg/dL range
    # Check if this is a glucose-related task by looking at the column name
    if 'glucose' in target_name.lower() or 'mg' in target_name.lower():
        ax.set_ylim(0, 450)
    else:
        # For non-glucose tasks, use data-driven limits
        ax.set_ylim(
            np.min(
                [
                    np.min(task.past_time[target_name]),
                    np.min(task.future_time[target_name]),
                    min_quantile_value,
                ]
            ),
            np.max(
                [
                    np.max(task.past_time[target_name]),
                    np.max(task.future_time[target_name]),
                    max_quantile_value,
                ]
            ),
        )

    handles, labels = ax.get_legend_handles_labels()
    order = [2, 0, 1, 3, 4, 5, 6]
    ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order])

    ax.set_title(task.name)

    if path:
        fig.savefig(path / "forecast.pdf")
        fig.savefig(path / "forecast.png", bbox_inches="tight")

    # Always close unless explicitly requested to return the figure (prevents memory growth in loops).
    if return_fig:
        return fig
    plt.close(fig)


def save_context(task, path):
    """
    Save the context of a task to a file for future reference.

    """
    with open(path / "context", "w") as f:
        f.write(
            f"""
Background:
{task.background}

Constraints:
{task.constraints}

Scenario:
{task.scenario}
"""
        )


def save_prompt(task, path):
    """
    Save the actual full prompt sent to the model.

    This captures task.prompt which contains the complete prompt with all context levels
    (e.g., nutritional info for NewDetail, calories/carbs for NewMedium, etc.)
    """
    prompt = getattr(task, 'prompt', None)
    with open(path / "prompt", "w") as f:
        f.write(
            f"""
Prompt:
{prompt if prompt is not None else "(No prompt - this task may not use LLM-based methods)"}
"""
        )


def save_output(extra_info, path):
    """
    Save the raw LLM outputs in a readable format.

    Each sample's raw output is saved with a separator for easy reading.
    """
    llm_outputs = extra_info.get('llm_outputs', [])
    with open(path / "output", "w") as f:
        f.write(
            f"""
Number of responses:
{len(llm_outputs)}

"""
        )
        for i, output in enumerate(llm_outputs):
            f.write(
                f"""Response {i + 1}:
{output}

--------------------------------------------------------------------------------
"""
            )


def save_evaluation(evaluation, path):
    """
    Save the content of the task evaluation content to a file for future reference.
    """
    with open(path / "evaluation", "w") as f:
        pprint(evaluation, f)


def save_extra_info(extra_info, path):
    """
    Save the content of the model extra information content to a file for future reference.
    """
    with open(path / "extra_info", "w") as f:
        pprint(extra_info, f)


def save_complete_data(task, samples, evaluation, extra_info, path):
    """
    Save all data (input, context, predictions, evaluation) in a single pickle file.

    Parameters:
    -----------
    task : BaseTask
        The task instance containing input data and context
    samples : np.ndarray
        The prediction samples from the model
    evaluation : dict
        The evaluation results/metrics
    extra_info : dict
        Additional model-specific information
    path : Path
        Directory where to save the pickle file
    """
    # Prepare complete data dictionary
    complete_data = {
        # Input data
        'input_data': {
            'past_time': task.past_time,
            'future_time': task.future_time,
            'c_cov': getattr(task, 'c_cov', None),
            'selected_event': getattr(task, 'selected_event', None),
            'region_of_interest': getattr(task, 'region_of_interest', None),
        },

        # Context information
        'context': {
            'background': task.background,
            'constraints': getattr(task, 'constraints', None),
            'scenario': task.scenario,
            'prompt': getattr(task, 'prompt', None),  # Full prompt sent to model
            'task_name': task.name if hasattr(task, 'name') else task.__class__.__name__,
            'task_class': task.__class__.__name__,
        },

        # Model outputs
        'predictions': {
            'samples': samples,
            'samples_shape': samples.shape,
            'n_samples': samples.shape[0] if len(samples.shape) > 0 else 0,
            'prediction_length': samples.shape[1] if len(samples.shape) > 1 else 0,
        },

        # Evaluation results
        'evaluation': evaluation,

        # Additional model information
        'extra_info': extra_info,

        # Metadata
        'metadata': {
            'task_attributes': {attr: getattr(task, attr, None)
                              for attr in ['seed', 'random_state']
                              if hasattr(task, attr)},
        }
    }

    # Save to pickle file
    with open(path / "complete_data.pkl", "wb") as f:
        pickle.dump(complete_data, f)

    # Also save a summary text file for quick inspection
    with open(path / "data_summary.txt", "w") as f:
        f.write(f"Task: {complete_data['context']['task_name']}\n")
        f.write(f"Task Class: {complete_data['context']['task_class']}\n")
        f.write(f"Prediction samples shape: {complete_data['predictions']['samples_shape']}\n")
        f.write(f"Number of samples: {complete_data['predictions']['n_samples']}\n")
        f.write(f"Prediction length: {complete_data['predictions']['prediction_length']}\n")
        f.write(f"Past data shape: {complete_data['input_data']['past_time'].shape}\n")
        f.write(f"Future data shape: {complete_data['input_data']['future_time'].shape}\n")

        if complete_data['input_data']['selected_event']:
            f.write(f"Selected event: {complete_data['input_data']['selected_event']}\n")

        f.write(f"\nEvaluation results:\n")
        if evaluation is None:
            f.write(f"  Evaluation skipped (partial samples)\n")
        elif isinstance(evaluation, dict):
            for key, value in evaluation.items():
                f.write(f"  {key}: {value}\n")
        else:
            f.write(f"  score: {evaluation}\n")


def evaluate_task(
    task_cls,
    seed,
    method_callable,
    n_samples,
    output_folder=None,
):
    if output_folder:
        task_folder = output_folder / task_cls.__name__
        seed_folder = task_folder / f"{seed}"
        seed_folder.mkdir(parents=True, exist_ok=True)

    try:
        # Instantiate the task
        ############################################################
        task = task_cls(seed=seed) # CORE: task is an instance of the task class
        # task = task_cls(fixed_config=fixed_config)
        task.random_instance()
        ############################################################

        logger.info(f"Method {method_callable} - Task {task.name} - Seed {seed}")
        samples = method_callable(task_instance=task, n_samples=n_samples)
        if isinstance(samples, tuple):
            samples, extra_info = samples
        else:
            extra_info = {}
        
        # Add sleep between task instances for API-based models to avoid rate limiting
        # Check if method_callable is a DirectPrompt instance (or wrapped in ResultCache)
        actual_method = getattr(method_callable, 'method', method_callable)
        if hasattr(actual_method, '_is_api_based') and actual_method._is_api_based():
            sleep_between_instances = getattr(actual_method, 'sleep_between_requests', 0.0)
            if sleep_between_instances > 0:
                import time
                logger.info(f"Sleeping {sleep_between_instances}s between task instances to avoid rate limiting...")
                time.sleep(sleep_between_instances)
        
        # Check if we have partial samples - skip evaluation but save data
        has_partial_samples = extra_info.get("partial_samples", False)
        
        if has_partial_samples:
            logger.warning(
                f"Skipping evaluation due to partial samples: "
                f"{extra_info.get('n_samples_obtained', 'unknown')}/{extra_info.get('n_samples_requested', 'unknown')}"
            )
            evaluation = None
            result = {
                "seed": seed,
                "error": f"Partial samples: {extra_info.get('n_samples_obtained', 'unknown')}/{extra_info.get('n_samples_requested', 'unknown')}",
            }
        else:
            evaluation = task.evaluate(samples)
            result = {
                "seed": seed,
                "score": (
                    evaluation["metric"] if isinstance(evaluation, dict) else evaluation
                ),
            }

        if output_folder:
            # Save context (always save)
            save_context(task=task, path=seed_folder)

            # Save the actual full prompt sent to model (always save)
            save_prompt(task=task, path=seed_folder)

            # Save extra_info content (always save)
            save_extra_info(extra_info=extra_info, path=seed_folder)

            # Save raw LLM outputs in readable format (if available)
            save_output(extra_info=extra_info, path=seed_folder)

            # Save complete data in a single pickle file (always save, even with partial samples)
            save_complete_data(task=task, samples=samples, evaluation=evaluation,
                             extra_info=extra_info, path=seed_folder)

            # Only save evaluation and plots if we have full samples
            if not has_partial_samples:
                # Save forecast plots
                plot_forecast_univariate(task=task, samples=samples, path=seed_folder)
                # Save metric content
                save_evaluation(evaluation=evaluation, path=seed_folder)
            else:
                # Save a note about skipped evaluation
                with open(seed_folder / "evaluation_skipped", "w") as f:
                    f.write(f"Evaluation skipped due to partial samples.\n")
                    f.write(f"Requested: {extra_info.get('n_samples_requested', 'unknown')}\n")
                    f.write(f"Obtained: {extra_info.get('n_samples_obtained', 'unknown')}\n")

        return (task_cls.__name__, result)

    except CacheMissError:
        logger.info(f"Skipping over cache miss.")
        return (
            task_cls.__name__,
            {
                "seed": seed,
                "error": f"Cache miss - Method {method_callable} - Task {task_cls.__name__} - Seed {seed}",
            },
        )

    except Exception as e:
        logger.error(f"Error evaluating task {task_cls.__name__} - Seed {seed}: {e}")
        logger.error(traceback.format_exc())
        if output_folder:
            with open(seed_folder / "error", "w") as f:
                f.write(str(e))
                f.write("\n")
                f.write(traceback.format_exc())
        return (task_cls.__name__, {"seed": seed, "error": str(e)})


def evaluate_all_tasks(
    ALL_TASKS,
    method_callable,
    n_instances = 5, # number of instances.
    n_samples=DEFAULT_N_SAMPLES,
    output_folder=None,
    use_cache=True,
    cache_name=None,
    max_parallel=None,
    skip_cache_miss=False,
):
    """
    Evaluate a method on all tasks.

    Parameters:
    -----------
    method_callable: callable
        The method to evaluate. Must take a task instance and return samples.
    seeds: int
        Number of seeds to evaluate on each task.
    n_samples: int
        Number of samples to generate.
    output_folder: Pathlike
        Directory in which to save the results.
    use_cache: bool
        Whether to use a cache to store/load results.
    cache_name: str
        Name of the cache.
    max_parallel: int
        Number of parallel processes to use.
    skip_cache_miss: bool
        Whether to skip computing tasks that are not found in the cache (useful for report generation).

    """

    logger.info(
        f"Evaluating method {method_callable} with {n_instances} seeds and {n_samples} samples on {len(ALL_TASKS)} tasks."
    )

    if output_folder:
        logger.info(f"Saving outputs to {output_folder}")
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    else:
        logger.info("No output folder provided. Results will not be saved.")

    if use_cache:
        method_callable = ResultCache(
            method_callable, method_name=cache_name, raise_on_miss=skip_cache_miss
        )

    tasks_to_evaluate = []
    for task_cls in ALL_TASKS: # 60 tasks 
        for seed in range(1, n_instances + 1): # 5 seeds 
            tasks_to_evaluate.append((task_cls, seed))

    func = partial(
        evaluate_task,
        method_callable=method_callable,
        n_samples=n_samples,
        output_folder=output_folder,
    )

    if max_parallel == 1:
        # No parallelism, just evaluate tasks in a loop
        results_list = [func(task_cls, seed) for task_cls, seed in tasks_to_evaluate]
    else:
        # Use multiprocessing to parallelize the evaluation
        with multiprocessing.Pool(processes=max_parallel) as pool:
            results_list = pool.starmap(func, tasks_to_evaluate)

    # Collect results
    results = defaultdict(list)
    for task_name, result in results_list:
        results[task_name].append(result)

    return results
