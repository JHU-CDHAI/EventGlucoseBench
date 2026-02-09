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



############################################################
# Standard Event Info Level: Profile + event timing + key metrics (calories/carbs)
############################################################



class EventCGMTask_D1_Age18_Diet_Ontime_StandardEventInfo(EventCGMTask_withEvent_withLag):

    __version__ = "0.1.0"

    def get_task_config(self) -> Dict:
        return {
            "data_lts_folder": DATA_LTS_FOLDER,
            "eventtype": ["Diet5Min"],
            "subgroup": ["D1-Age18"],
            "lag": 0,
            "context_length": 289,
            "prediction_length": 24,
            "use_calendar_covs": False,
            'prompt_level': 'standard_event_info',
            'prompt_max_digits': 1,
        }


class EventCGMTask_D1_Age40_Diet_Ontime_StandardEventInfo(EventCGMTask_withEvent_withLag):

    __version__ = "0.1.0"

    def get_task_config(self) -> Dict:
        return {
            "data_lts_folder": DATA_LTS_FOLDER,
            "eventtype": ["Diet5Min"],
            "subgroup": ["D1-Age40"],
            "lag": 0,
            "context_length": 289,
            "prediction_length": 24,
            "use_calendar_covs": False,
            'prompt_level': 'standard_event_info',
            'prompt_max_digits': 1,
        }


class EventCGMTask_D1_Age65_Diet_Ontime_StandardEventInfo(EventCGMTask_withEvent_withLag):

    __version__ = "0.1.0"

    def get_task_config(self) -> Dict:
        return {
            "data_lts_folder": DATA_LTS_FOLDER,
            "eventtype": ["Diet5Min"],
            "subgroup": ["D1-Age65"],
            "lag": 0,
            "context_length": 289,
            "prediction_length": 24,
            "use_calendar_covs": False,
            'prompt_level': 'standard_event_info',
            'prompt_max_digits': 1,
        }


class EventCGMTask_D2_Age18_Diet_Ontime_StandardEventInfo(EventCGMTask_withEvent_withLag):
    __version__ = "0.1.0"

    def get_task_config(self) -> Dict:
        return {
            "data_lts_folder": DATA_LTS_FOLDER,
            "eventtype": ["Diet5Min"],
            "subgroup": ["D2-Age18"],
            "lag": 0,
            "context_length": 289,
            "prediction_length": 24,
            "use_calendar_covs": False,
            'prompt_level': 'standard_event_info',
            'prompt_max_digits': 1,
        }


class EventCGMTask_D2_Age40_Diet_Ontime_StandardEventInfo(EventCGMTask_withEvent_withLag):
    __version__ = "0.1.0"

    def get_task_config(self) -> Dict:
        return {
            "data_lts_folder": DATA_LTS_FOLDER,
            "eventtype": ["Diet5Min"],
            "subgroup": ["D2-Age40"],
            "lag": 0,
            "context_length": 289,
            "prediction_length": 24,
            "use_calendar_covs": False,
            'prompt_level': 'standard_event_info',
            'prompt_max_digits': 1,
        }


class EventCGMTask_D2_Age65_Diet_Ontime_StandardEventInfo(EventCGMTask_withEvent_withLag):
    __version__ = "0.1.0"

    def get_task_config(self) -> Dict:
        return {
            "data_lts_folder": DATA_LTS_FOLDER,
            "eventtype": ["Diet5Min"],
            "subgroup": ["D2-Age65"],
            "lag": 0,
            "context_length": 289,
            "prediction_length": 24,
            "use_calendar_covs": False,
            'prompt_level': 'standard_event_info',
            'prompt_max_digits': 1,
        }



# ============================================================================
# Cluster Registry
# ============================================================================

__cluster__ = [
    EventCGMTask_D1_Age18_Diet_Ontime_StandardEventInfo,
    EventCGMTask_D1_Age40_Diet_Ontime_StandardEventInfo,
    EventCGMTask_D1_Age65_Diet_Ontime_StandardEventInfo,
    EventCGMTask_D2_Age18_Diet_Ontime_StandardEventInfo,
    EventCGMTask_D2_Age40_Diet_Ontime_StandardEventInfo,
    EventCGMTask_D2_Age65_Diet_Ontime_StandardEventInfo,
]
