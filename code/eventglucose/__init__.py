__version__ = "0.0.1"

from fractions import Fraction

from .base import BaseTask

from .tasks.glucose_cgm_task import (
    __TASKS__ as GLUCOSE_TASKS,
    __CLUSTERS__ as GLUCOSE_CLUSTERS,
)

# All tasks that are officially included in the benchmark
# TEMPORARILY MODIFIED: Only glucose tasks for testing
ALL_TASKS = (
    GLUCOSE_TASKS  # Only glucose tasks
)

# TEMPORARILY MODIFIED: Only glucose clusters for testing
WEIGHT_CLUSTERS = (
    GLUCOSE_CLUSTERS  # Only glucose clusters
)


def get_task_weight(task: BaseTask) -> Fraction:
    for cluster in WEIGHT_CLUSTERS:
        if task in cluster.tasks:
            return Fraction(cluster.weight) / len(cluster.tasks)


TASK_NAME_TO_WEIGHT = {task.__name__: get_task_weight(task) for task in ALL_TASKS}
