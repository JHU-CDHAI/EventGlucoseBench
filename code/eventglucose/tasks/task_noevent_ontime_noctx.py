import numpy as np
import pandas as pd
from typing import Dict, Optional

from ..config import DATA_LTS_FOLDER
from . import WeightCluster
from .eventglucose_tasks import EventCGMTask_withEvent_withLag, load_df_lts_data


class EventCGMTask_NoEvent_Ontime_NoCtx(EventCGMTask_withEvent_withLag):
    """
    "No-event" variant (Option B):
    - still reads from the existing event-filtered task dataset files
    - but does NOT anchor the forecast boundary to `prediction_time_idx`
    - generates NO event description/prompt context
    - uses zero intervention covariates (Diet/Med/Exercise set to 0)

    This is meant to create a baseline forecasting setup while reusing the same
    pre-extracted window format in the stored .pkl rows (target_sequence, start_time, ...).
    """

    __version__ = "0.2.0"

    def get_task_config(self) -> Dict:
        return {
            "data_lts_folder": DATA_LTS_FOLDER,
            # Empty eventtype => no filter in select_files(); reuse whatever windows exist.
            "eventtype": [],
            "subgroup": [],
            "event_columns": [],
            "lag": 0,
            "context_length": 289,
            "prediction_length": 24,
            "use_calendar_covs": False,
            "prompt_level": "noctx",
            "prompt_max_digits": 1,
        }

    def _get_target_and_start(self, item) -> tuple[np.ndarray, pd.Timestamp]:
        """
        Robustly fetch (target_sequence, start_time) from a row.
        """
        # Most EventGlucose task_dataset rows use these keys
        y = self._safe_seq(item.get("target_sequence", None))
        start_time = item.get("start_time", None)

        # Back-compat fallbacks
        if y is None:
            y = self._safe_seq(item.get("target", None))
        if start_time is None:
            start_time = item.get("start", None)

        if y is None:
            raise ValueError("Could not parse target sequence from row (expected 'target_sequence' or 'target').")
        if start_time is None:
            raise ValueError("Could not find start time in row (expected 'start_time' or 'start').")

        return y, pd.to_datetime(start_time)

    def random_instance(self):
        """
        Sample a random cut point t within a row's target_sequence and create
        past/future windows, with NO event descriptions and zeroed event covariates.
        """
        self.task_config = self.get_task_config()
        task_config = self.task_config

        df = load_df_lts_data(task_config)
        self.cfg = task_config
        self.lag = int(task_config.get("lag", 0))

        C = int(task_config["context_length"])
        H = int(task_config["prediction_length"])

        # Pick a random row
        if self.random is not None and hasattr(self.random, "randint"):
            idx_to_select = int(self.random.randint(0, len(df)))
        else:
            idx_to_select = int(np.random.randint(0, len(df)))
        item = df.iloc[idx_to_select]

        y, start_time = self._get_target_and_start(item)
        n = int(len(y))
        if n < C + H:
            raise ValueError(f"Series too short for C={C}, H={H}: len={n}")

        # 5-min cadence index over the full sequence
        full_idx = pd.date_range(start=start_time, periods=n, freq="5min")
        y_series = pd.Series(y, index=full_idx, name="glucose_mg_dl")

        # Choose a random valid cut point t (no anchoring to prediction_time_idx)
        t_min, t_max = C, n - H
        if t_max <= t_min:
            raise ValueError(f"Invalid cut range: C={C}, H={H}, len={n}")
        if self.random is not None and hasattr(self.random, "randint"):
            t = int(self.random.randint(t_min, t_max))
        else:
            t = int(np.random.randint(t_min, t_max))

        past_series, future_series = self._split_at_t(y_series, t, C, H)

        # Assign past/future dataframes
        self.past_time = past_series.to_frame()
        self.future_time = future_series.to_frame()

        # Zero intervention covariates (Diet/Med/Exercise) to match "no-event" setup.
        zeros_past = np.zeros((C, 1), dtype=float)
        zeros_fut = np.zeros((H, 1), dtype=float)

        # Calendar covariates (zero columns if use_calendar_covs is False)
        cal_past = self._calendar_covs(past_series.index)
        cal_fut = self._calendar_covs(future_series.index)

        self.c_cov = {
            "past": np.concatenate([cal_past, zeros_past, zeros_past, zeros_past], axis=1),
            "future": np.concatenate([cal_fut, zeros_fut, zeros_fut, zeros_fut], axis=1),
        }

        # No event description
        self.selected_event = None
        self.background = (
            "This is continuous glucose monitoring (CGM) data measuring blood glucose levels every 5 minutes. "
            "The task is to forecast future glucose levels based on historical patterns."
        )
        fut_start, fut_end = future_series.index[0], future_series.index[-1]
        self.scenario = (
            f"No known intervention events are provided during the forecast window "
            f"({fut_start.strftime('%Y-%m-%d %H:%M')} to {fut_end.strftime('%Y-%m-%d %H:%M')})."
        )

        # Keep default ROI settings
        self.region_of_interest = None
        self.roi_weight = 0.5

        # Generate prompt if configured (mirrors EventCGMTask_withEvent_withLag)
        prompt_level: Optional[str] = task_config.get("prompt_level", None)
        if prompt_level is not None:
            from ..prompts.make_prompt import MakePrompt

            prompt_config = {
                "time_format": task_config.get("prompt_time_format", "time_value_pairs"),
                "max_history_points": task_config.get("prompt_max_history", None),
                "max_digits": task_config.get("prompt_max_digits", 1),
                "include_units": task_config.get("prompt_include_units", True),
            }
            prompt_maker = MakePrompt(**prompt_config)
            fn_prompt = prompt_maker.get_prompt_fn(level=prompt_level)
            self.prompt = fn_prompt(self)
        else:
            self.prompt = None

        return self


class EventCGMTask_D1_Age18_NoEvent_Ontime_NoCtx(EventCGMTask_NoEvent_Ontime_NoCtx):
    __version__ = "0.2.0"

    def get_task_config(self) -> Dict:
        cfg = super().get_task_config()
        cfg["subgroup"] = ["D1-Age18"]
        return cfg


class EventCGMTask_D1_Age40_NoEvent_Ontime_NoCtx(EventCGMTask_NoEvent_Ontime_NoCtx):
    __version__ = "0.2.0"

    def get_task_config(self) -> Dict:
        cfg = super().get_task_config()
        cfg["subgroup"] = ["D1-Age40"]
        return cfg


class EventCGMTask_D1_Age65_NoEvent_Ontime_NoCtx(EventCGMTask_NoEvent_Ontime_NoCtx):
    __version__ = "0.2.0"

    def get_task_config(self) -> Dict:
        cfg = super().get_task_config()
        cfg["subgroup"] = ["D1-Age65"]
        return cfg


class EventCGMTask_D2_Age18_NoEvent_Ontime_NoCtx(EventCGMTask_NoEvent_Ontime_NoCtx):
    __version__ = "0.2.0"

    def get_task_config(self) -> Dict:
        cfg = super().get_task_config()
        cfg["subgroup"] = ["D2-Age18"]
        return cfg


class EventCGMTask_D2_Age40_NoEvent_Ontime_NoCtx(EventCGMTask_NoEvent_Ontime_NoCtx):
    __version__ = "0.2.0"

    def get_task_config(self) -> Dict:
        cfg = super().get_task_config()
        cfg["subgroup"] = ["D2-Age40"]
        return cfg


class EventCGMTask_D2_Age65_NoEvent_Ontime_NoCtx(EventCGMTask_NoEvent_Ontime_NoCtx):
    __version__ = "0.2.0"

    def get_task_config(self) -> Dict:
        cfg = super().get_task_config()
        cfg["subgroup"] = ["D2-Age65"]
        return cfg


# ============================================================================
# Cluster Registry
# ============================================================================

__cluster__ = [
    EventCGMTask_D1_Age18_NoEvent_Ontime_NoCtx,
    EventCGMTask_D1_Age40_NoEvent_Ontime_NoCtx,
    EventCGMTask_D1_Age65_NoEvent_Ontime_NoCtx,
    EventCGMTask_D2_Age18_NoEvent_Ontime_NoCtx,
    EventCGMTask_D2_Age40_NoEvent_Ontime_NoCtx,
    EventCGMTask_D2_Age65_NoEvent_Ontime_NoCtx,
]

