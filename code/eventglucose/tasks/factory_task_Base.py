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

class EventCGMTask_test(EventCGMTask_withEvent_withLag):

    __version__ = "0.2.0"

    def get_task_config(self) -> Dict:
        """Return task configuration for EventCGMTask_test."""
        return {
            "data_lts_folder": "/home/xzhi2/GlucoCIK/_Data/_Data/ProjB-Bench-2-EventGlucose/task_dataset",
            "eventtype": ["Diet5Min"],
            "subgroup": ["D1-Age18", "D1-Age65"],
            "event_columns": ["Diet5Min"],
            "lag": 0,
            # context_length: number of historical time steps used as input for predictions
            "context_length": 289,
            "prediction_length": 24,
            "use_calendar_covs": True,
            # Keep the smoke-test task runnable with DirectPrompt out of the box.
            "prompt_level": "noctx",
        }
        


# class EventCGMTask_Diet_TypeOne(EventCGMTask_withEvent_withLag):

#     __version__ = "0.2.0"

#     def get_task_config(self) -> Dict:
#         """Return task configuration for EventCGMTask_Diet_TypeOne.

#         Filters for:
#         - Diet events (both single Diet and Diet-Med combo)
#         - Type 1 diabetes patients only (D1 subgroups, all age groups)
#         """
#         return {
#             "data_lts_folder": DATA_LTS_FOLDER,
#             "eventtype": ["Diet5Min", "Diet5Min-Med5Min"],
#             "subgroup": ["D1-Age18", "D1-Age40", "D1-Age65"],  # Type 1 diabetes, all ages
#             "event_columns": ["Diet5Min", "Med5Min"],
#             "lag": 0,
#             "context_length": 289,
#             "prediction_length": 24,
#             "use_calendar_covs": False,
#             "prompt_config": None
#         }


# # ============================================================================
# # Event-specific Task Classes (all subgroups)
# # ============================================================================

# class EventCGMTask_Diet5Min(EventCGMTask_withEvent_withLag):
#     """Task class for single Diet events only, all subgroups."""

#     __version__ = "0.2.0"

#     def get_task_config(self) -> Dict:
#         """Return task configuration for Diet5Min events (all patients)."""
#         return {
#             "data_lts_folder": DATA_LTS_FOLDER,
#             "eventtype": ["Diet5Min"],
#             "subgroup": [],  # All subgroups
#             "event_columns": ["Diet5Min"],
#             "lag": 0,
#             "context_length": 289,
#             "prediction_length": 24,
#             "use_calendar_covs": False,
#             "prompt_config": None
#         }


# class EventCGMTask_Diet5Min_Med5Min(EventCGMTask_withEvent_withLag):
#     """Task class for Diet-Med combo events only, all subgroups."""

#     __version__ = "0.2.0"

#     def get_task_config(self) -> Dict:
#         """Return task configuration for Diet5Min-Med5Min combo events (all patients)."""
#         return {
#             "data_lts_folder": DATA_LTS_FOLDER,
#             "eventtype": ["Diet5Min-Med5Min"],
#             "subgroup": [],  # All subgroups
#             "event_columns": ["Diet5Min", "Med5Min"],
#             "lag": 0,
#             "context_length": 289,
#             "prediction_length": 24,
#             "use_calendar_covs": False,
#             "prompt_config": None
#         }


# class EventCGMTask_Exercise5Min(EventCGMTask_withEvent_withLag):
#     """Task class for Exercise events only, all subgroups."""

#     __version__ = "0.2.0"

#     def get_task_config(self) -> Dict:
#         """Return task configuration for Exercise5Min events (all patients)."""
#         return {
#             "data_lts_folder": DATA_LTS_FOLDER,
#             "eventtype": ["Exercise5Min"],
#             "subgroup": [],  # All subgroups
#             "event_columns": ["Exercise5Min"],
#             "lag": 0,
#             "context_length": 289,
#             "prediction_length": 24,
#             "use_calendar_covs": False,
#             "prompt_config": None
#         }


# # ============================================================================
# # Subgroup-specific Task Classes (all event types)
# # ============================================================================

# class EventCGMTask_D1_Age18(EventCGMTask_withEvent_withLag):
#     """Task class for Type 1 diabetes, Age 18, all event types."""

#     __version__ = "0.2.0"

#     def get_task_config(self) -> Dict:
#         """Return task configuration for D1-Age18 patients (all events)."""
#         return {
#             "data_lts_folder": DATA_LTS_FOLDER,
#             "eventtype": ["Diet5Min", "Diet5Min-Med5Min", "Exercise5Min"],
#             "subgroup": ["D1-Age18"],
#             "event_columns": ["Diet5Min", "Med5Min", "Exercise5Min"],
#             "lag": 0,
#             "context_length": 289,
#             "prediction_length": 24,
#             "use_calendar_covs": False,
#             "prompt_config": None
#         }


# class EventCGMTask_D1_Age40(EventCGMTask_withEvent_withLag):
#     """Task class for Type 1 diabetes, Age 40, all event types."""

#     __version__ = "0.2.0"

#     def get_task_config(self) -> Dict:
#         """Return task configuration for D1-Age40 patients (all events)."""
#         return {
#             "data_lts_folder": DATA_LTS_FOLDER,
#             "eventtype": ["Diet5Min", "Diet5Min-Med5Min", "Exercise5Min"],
#             "subgroup": ["D1-Age40"],
#             "event_columns": ["Diet5Min", "Med5Min", "Exercise5Min"],
#             "lag": 0,
#             "context_length": 289,
#             "prediction_length": 24,
#             "use_calendar_covs": False,
#             "prompt_config": None
#         }


# class EventCGMTask_D1_Age65(EventCGMTask_withEvent_withLag):
#     """Task class for Type 1 diabetes, Age 65, all event types."""

#     __version__ = "0.2.0"

#     def get_task_config(self) -> Dict:
#         """Return task configuration for D1-Age65 patients (all events)."""
#         return {
#             "data_lts_folder": DATA_LTS_FOLDER,
#             "eventtype": ["Diet5Min", "Diet5Min-Med5Min", "Exercise5Min"],
#             "subgroup": ["D1-Age65"],
#             "event_columns": ["Diet5Min", "Med5Min", "Exercise5Min"],
#             "lag": 0,
#             "context_length": 289,
#             "prediction_length": 24,
#             "use_calendar_covs": False,
#             "prompt_config": None
#         }


# class EventCGMTask_D2_Age18(EventCGMTask_withEvent_withLag):
#     """Task class for Type 2 diabetes, Age 18, all event types."""

#     __version__ = "0.2.0"

#     def get_task_config(self) -> Dict:
#         """Return task configuration for D2-Age18 patients (all events)."""
#         return {
#             "data_lts_folder": DATA_LTS_FOLDER,
#             "eventtype": ["Diet5Min", "Diet5Min-Med5Min", "Exercise5Min"],
#             "subgroup": ["D2-Age18"],
#             "event_columns": ["Diet5Min", "Med5Min", "Exercise5Min"],
#             "lag": 0,
#             "context_length": 289,
#             "prediction_length": 24,
#             "use_calendar_covs": False,
#             "prompt_config": None
#         }


# class EventCGMTask_D2_Age40(EventCGMTask_withEvent_withLag):
#     """Task class for Type 2 diabetes, Age 40, all event types."""

#     __version__ = "0.2.0"

#     def get_task_config(self) -> Dict:
#         """Return task configuration for D2-Age40 patients (all events)."""
#         return {
#             "data_lts_folder": DATA_LTS_FOLDER,
#             "eventtype": ["Diet5Min", "Diet5Min-Med5Min", "Exercise5Min"],
#             "subgroup": ["D2-Age40"],
#             "event_columns": ["Diet5Min", "Med5Min", "Exercise5Min"],
#             "lag": 0,
#             "context_length": 289,
#             "prediction_length": 24,
#             "use_calendar_covs": False,
#             "prompt_config": None
#         }


# class EventCGMTask_D2_Age65(EventCGMTask_withEvent_withLag):
#     """Task class for Type 2 diabetes, Age 65, all event types."""

#     __version__ = "0.2.0"

#     def get_task_config(self) -> Dict:
#         """Return task configuration for D2-Age65 patients (all events)."""
#         return {
#             "data_lts_folder": DATA_LTS_FOLDER,
#             "eventtype": ["Diet5Min", "Diet5Min-Med5Min", "Exercise5Min"],
#             "subgroup": ["D2-Age65"],
#             "event_columns": ["Diet5Min", "Med5Min", "Exercise5Min"],
#             "lag": 0,
#             "context_length": 289,
#             "prediction_length": 24,
#             "use_calendar_covs": False,
#             "prompt_config": None
#         }


# ============================================================================
# Cluster Registry
# ============================================================================

__cluster__ = [
    EventCGMTask_Base,
    EventCGMTask_test,
    # EventCGMTask_Diet_TypeOne,
    # # Event-specific tasks
    # EventCGMTask_Diet5Min,
    # EventCGMTask_Diet5Min_Med5Min,
    # EventCGMTask_Exercise5Min,
    # # Subgroup-specific tasks
    # EventCGMTask_D1_Age18,
    # EventCGMTask_D1_Age40,
    # EventCGMTask_D1_Age65,
    # EventCGMTask_D2_Age18,
    # EventCGMTask_D2_Age40,
    # EventCGMTask_D2_Age65,
]
