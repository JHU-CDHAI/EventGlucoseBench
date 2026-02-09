import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from ..config import DATA_STORAGE_PATH
from ..base import UnivariateCRPSTask
from . import WeightCluster

from .eventglucose_tasks import EventCGMTask_withEvent_withLag
from ..config import DATA_LTS_FOLDER



class EventCGMTask_Base(EventCGMTask_withEvent_withLag):

    __version__ = "0.2.0"

    def get_task_config(self) -> Dict:
        """Return task configuration for EventCGMTask_Base.

        Includes all event types: single Diet, single Exercise, and combo Diet-Med.
        Includes all subgroups (both Type 1 and Type 2 diabetes, all age groups).
        """
        return {
            "data_lts_folder": DATA_LTS_FOLDER,
            "eventtype": ["Diet5Min", "Diet5Min-Med5Min", "Exercise5Min"],
            "subgroup": [],  # Empty list means no subgroup filtering (all D1/D2 ages)
            "event_columns": ["Diet5Min", "Med5Min", "Exercise5Min"],
            "lag": 0,
            "context_length": 289,
            "prediction_length": 24,
            "use_calendar_covs": False,
            # Ensure DirectPrompt-compatible tasks always have a non-None `self.prompt`
            # (see `EventCGMTask_withEvent_withLag.random_instance()`).
            "prompt_level": "noctx",
        }

# ============================================================================
# Cluster Registry
# ============================================================================

__cluster__ = [
    EventCGMTask_Base
]
