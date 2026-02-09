"""
2025-10-28: This Old Version GlucoseCGMTask
It Will be deleted after all corresponding experiment scripts and notebooks are updated
"""

import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from ..config import DATA_STORAGE_PATH
from ..base import UnivariateCRPSTask
from . import WeightCluster



class GlucoseCGMTask(UnivariateCRPSTask):
    """
    CiK-style task for CGM (glucose) forecasting with:
      • fixed-size random windows (past=289, future=24 by default),
      • known-in-advance covariates via `c_cov` (calendar + Diet/Med/Exercise),
      • text context via `background` and `scenario`.
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.1.1"

    def __init__(
        self,
        seed: int = None,
        fixed_config: Optional[dict] = None,
        data_path: Optional[str] = None,):

        
        # Defaults; override via fixed_config if you like
        self.cfg = {
            "context_length": 289,   # past steps
            "prediction_length": 24, # future steps
            "use_calendar_covs": True,
        }
        if fixed_config:
            self.cfg.update(fixed_config)

        # print(data_path)
        self.data_path = Path(data_path) 
        # print(self.data_path)

        # Create proper fixed_config for parent class
        parent_fixed_config = None
        if fixed_config is not None:
            parent_fixed_config = {
                "past_time": fixed_config.get("past_time"),
                "future_time": fixed_config.get("future_time"),
                "constraints": fixed_config.get("constraints"),
                "background": fixed_config.get("background"),
                "scenario": fixed_config.get("scenario"),
                "region_of_interest": fixed_config.get("region_of_interest", None),
                "roi_weight": fixed_config.get("roi_weight", 0.5),
                "metric_constraint": fixed_config.get("metric_constraint", None),
            }
            # Only include keys if any task-specific data is provided
            if any(k in fixed_config for k in ["past_time", "future_time", "constraints", "background", "scenario"]):
                # Keep parent_fixed_config as is since we have task data
                pass
            else:
                # No task data provided, so don't pass fixed_config to parent
                parent_fixed_config = None

        super().__init__(seed=seed, fixed_config=parent_fixed_config)

    # -------------------------
    # Loading & safe parsing
    # -------------------------
    @staticmethod
    def _safe_seq(x):
        """Return np.ndarray[float] from a stringified list or array-like; None if not parseable."""
        if isinstance(x, str):
            try:
                return np.array(ast.literal_eval(x), dtype=float)
            except Exception:
                return None
        if isinstance(x, (list, np.ndarray, pd.Series)):
            return np.asarray(x, dtype=float)
        return None

    def load_data(self) -> pd.DataFrame:
        """Load the dataframe (pickle expected)."""
        df = pd.read_pickle(self.data_path)
        required = {"item_id", "target", "start"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return df

    # -------------------------
    # Feature builders
    # -------------------------
    def _calendar_covs(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """Day-of-week + hour-of-day one-hots (known-in-advance covariates)."""
        if not self.cfg["use_calendar_covs"]:
            return np.zeros((len(idx), 0), dtype=float)
        dow = np.eye(7, dtype=float)[idx.dayofweek.values]  # (N, 7)
        hr  = np.eye(24, dtype=float)[idx.hour.values]      # (N, 24)
        return np.concatenate([dow, hr], axis=1)            # (N, 31)

    @staticmethod
    def _dense_from_sparse_dict(obj, length: int) -> np.ndarray:
        """
        Your Diet/Med/Exercise columns often look like sparse dicts keyed by index:
          {330: {...}, 590: {...}, ...}
        Convert to a dense 0/1 array of length `length` where present indices are 1.
        """
        if obj is None:
            return np.zeros(length, dtype=float)

        # If it's a string, try to parse to a dict
        if isinstance(obj, str) and obj.strip():
            try:
                obj = ast.literal_eval(obj)
            except Exception:
                return np.zeros(length, dtype=float)

        # Dict -> mark keys
        if isinstance(obj, dict):
            arr = np.zeros(length, dtype=float)
            for k in obj.keys():
                try:
                    i = int(k)
                    if 0 <= i < length:
                        arr[i] = 1.0
                except Exception:
                    continue
            return arr

        # Already dense?
        try:
            a = np.asarray(obj, dtype=float)
            if len(a) == length:
                return a
        except Exception:
            pass

        return np.zeros(length, dtype=float)

    @staticmethod
    def _split_at_t(series: pd.Series, t: int, C: int, H: int) -> Tuple[pd.Series, pd.Series]:
        """Return (past, future) where past = [t-C, t), future = [t, t+H)."""
        past = series.iloc[t - C:t]
        fut  = series.iloc[t:t + H]
        return past, fut

    # -------------------------
    # Main instance builder
    # -------------------------
    def random_instance(self):
        """
        Random patient, random cut t with past=C and future=H windowing.
        Sets:
          self.past_time  -> DataFrame (C, 1)
          self.future_time-> DataFrame (H, 1)
          self.c_cov      -> {"past": (C,K), "future": (H,K)}
          self.background, self.scenario -> strings
        """
        df = self.load_data()
        # Random patient/item (uses self.random if provided by base)
        item = df.sample(n=1, random_state=self.random).iloc[0]

        # ---- Numeric target ----
        y = self._safe_seq(item["target"])
        if y is None:
            raise ValueError("Could not parse 'target' as numeric sequence.")
        n = len(y)

        C = int(self.cfg["context_length"])
        H = int(self.cfg["prediction_length"])
        if n < C + H:
            raise ValueError(f"Series too short for C={C}, H={H}: len={n}")

        # 5-min cadence index over the *entire* series
        start_time = pd.to_datetime(item["start"])
        full_idx = pd.date_range(start=start_time, periods=n, freq="5min")
        y_series = pd.Series(y, index=full_idx, name="glucose_mg_dl")

        # ---- Interventions (sparse -> dense) over the full series ----
        diet_full = self._dense_from_sparse_dict(item.get("Diet5Min", None), n)
        med_full  = self._dense_from_sparse_dict(item.get("Med5Min", None),  n)
        ex_full   = self._dense_from_sparse_dict(item.get("Exercise5Min", None), n)

        # ---- Choose a random valid cut t (C past, H future) ----
        # Use a seed derived from self.random if available, else non-deterministic
        try:
            seed = int(self.random)
        except Exception:
            try:
                # if self.random is a Random() with randint
                seed = int(self.random.randint(0, 2**32 - 1))
            except Exception:
                seed = None
        rng = np.random.default_rng(seed)

        t_min, t_max = C, n - H
        t = int(rng.integers(t_min, t_max))  # t in [C, n-H)

        # ---- Slice past/future windows ----
        past_series, future_series = self._split_at_t(y_series, t, C, H)

        past_slice = slice(t - C, t)
        fut_slice  = slice(t, t + H)

        past_diet = diet_full[past_slice]
        fut_diet  = diet_full[fut_slice]
        past_med  = med_full[past_slice]
        fut_med   = med_full[fut_slice]
        past_ex   = ex_full[past_slice]
        fut_ex    = ex_full[fut_slice]

        # ---- Calendar covariates (known-in-advance) ----
        cal_past = self._calendar_covs(past_series.index)   # (C, 31)
        cal_fut  = self._calendar_covs(future_series.index) # (H, 31)

        # Stack covariates in a consistent order
        self.c_cov = {
            "past":  np.concatenate(
                [cal_past,
                 past_diet.reshape(-1, 1),
                 past_med.reshape(-1, 1),
                 past_ex.reshape(-1, 1)], axis=1
            ),
            "future": np.concatenate(
                [cal_fut,
                 fut_diet.reshape(-1, 1),
                 fut_med.reshape(-1, 1),
                 fut_ex.reshape(-1, 1)], axis=1
            ),
        }

        # ---- Human-readable text context for THIS window ----
        gender = item.get("Gender", "Unknown")
        yob = item.get("YearOfBirth", np.nan)
        disease = item.get("DiseaseType", "Unknown")
        tz = item.get("UserTimeZone", "Unknown")

        if pd.notna(yob):
            try:
                age = pd.Timestamp.now().year - int(yob)
                age_str = f"{age} years old"
            except Exception:
                age_str = "age unknown"
        else:
            age_str = "age unknown"

        self.background = (
            "This is continuous glucose monitoring (CGM) data from a patient. "
            f"Patient demographics: Gender: {gender}, {age_str}, "
            f"Disease Type: {disease}, Time Zone: {tz}."
            f"the person eat at ..., take medicine at ..., activate at ...."
        )

        fut_start, fut_end = future_series.index[0], future_series.index[-1]

        def _count_pos(a): return int(np.sum(a > 0))
        parts = []
        if _count_pos(fut_diet): parts.append(f"meals at {_count_pos(fut_diet)} time points")
        if _count_pos(fut_med):  parts.append(f"medication at {_count_pos(fut_med)} time points")
        if _count_pos(fut_ex):   parts.append(f"exercise at {_count_pos(fut_ex)} time points")

        when = f"({fut_start:%Y-%m-%d %H:%M} to {fut_end:%Y-%m-%d %H:%M})"
        self.scenario = (
            f"During the forecast period {when}: " + ", ".join(parts) + "."
            if parts else f"No specific interventions are planned during the forecast period {when}."
        )

        # ---- Required fields for CiK-style baselines/metrics ----
        # Ensure frequency is preserved when creating DataFrames
        past_series.index.freq = "5min"
        future_series.index.freq = "5min"
        self.past_time   = past_series.to_frame(name="glucose_mg_dl")   # (289, 1) by default
        self.future_time = future_series.to_frame(name="glucose_mg_dl") # (24, 1) by default
        self.constraints = None
        
        # ROI: Could be event-specific periods (e.g., 2 hours post-meal)
        # For now, no specific ROI defined
        self.region_of_interest = None
        self.roi_weight = 0.5  # Default ROI weight
        self.metric_constraint = None  # No constraints on glucose predictions

        self.patient_metadata = {
            "patient_id": item.get("PatientID", "Unknown"),
            "gender": gender,
            "year_of_birth": yob,
            "disease_type": disease,
            "timezone": tz,
            "cut_index": t,
        }
        
        return self
    
    @property
    def seasonal_period(self) -> int:
        """
        Return the seasonal period for glucose CGM data.
        For 5-minute glucose data, we use 288 periods per day (24*60/5 = 288).
        """
        return 288  # 5-min intervals, 288 per day

    # -------------------------
    # Convenience helpers
    # -------------------------
    def get_intervention_context(self) -> Dict[str, np.ndarray]:
        """Return interventions split into past/future arrays (0/1 flags)."""
        return {
            "past_diet": self.c_cov["past"][:, -3],   # last 3 columns are diet/med/exercise
            "future_diet": self.c_cov["future"][:, -3],
            "past_medication": self.c_cov["past"][:, -2],
            "future_medication": self.c_cov["future"][:, -2],
            "past_exercise": self.c_cov["past"][:, -1],
            "future_exercise": self.c_cov["future"][:, -1],
        }

    def get_numpy_payload(self) -> Dict[str, np.ndarray]:
        """Handy shapes/arrays for quick debugging in notebooks."""
        return {
            "past_y": self.past_time.values.squeeze(-1),       # (C,)
            "future_h": np.array([len(self.future_time)], int),
            "past_cov": self.c_cov["past"],                    # (C, K)
            "future_cov": self.c_cov["future"],                # (H, K)
        }
    
    def evaluate_forecast(self, samples, task_instance=None) -> float:
        """Evaluate forecast using CRPS metric (compatibility method)."""
        # Handle case where models return (samples, extra_info) tuple
        if isinstance(samples, tuple):
            samples = samples[0]  # Extract the actual samples array
        
        result = self.evaluate(samples)
        if isinstance(result, dict):
            return result.get('crps', result.get('metric', float('nan')))
        return result
        
    def iter_all_instances(self, stride: int = 1):
        """
        Yields entries compatible with roi_crps(entry, forecast) for evaluation.
        Each entry corresponds to one (patient, cut t) window:
        - past_time / future_time (as pandas DataFrames)
        - metadata needed by the metric (roi info, constraints, scaling, weight)
        """
        import ast, numpy as np, pandas as pd

        df = self.load_data()
        C = int(self.cfg["context_length"])
        H = int(self.cfg["prediction_length"])

        for _, item in df.iterrows():
            # parse glucose sequence
            tgt = item["target"]
            y = np.array(ast.literal_eval(tgt), dtype=float) if isinstance(tgt, str) else np.asarray(tgt, float)
            n = len(y)
            if n < C + H:
                continue

            start_time = pd.to_datetime(item["start"])
            idx = pd.date_range(start=start_time, periods=n, freq="5min")

            # slide all valid cuts t (optionally downsample with stride)
            for t in range(C, n - H, stride):
                past = pd.Series(y[t - C:t], index=idx[t - C:t], name="glucose_mg_dl").to_frame()
                fut  = pd.Series(y[t:t + H],  index=idx[t:t + H],  name="glucose_mg_dl").to_frame()

                # minimal entry dict expected by roi_crps (see HF helper)
                entry = {
                    "name": "glucose_cgm",
                    "seed": int(hash((item.get("item_id", "NA"), t)) & 0x7FFFFFFF),
                    "past_time": past.to_json(),    # the HF helper reads JSON then np
                    "future_time": fut.to_json(),
                    # metric controls — keep defaults unless you want ROI/constraints
                    "region_of_interest": [],       # e.g., [i0, i1, ...] if you define an RoI
                    "metric_scaling": 1.0,          # per-task scaling (HF helper multiplies by this)
                    "constraint_min": -np.inf,
                    "constraint_max":  np.inf,
                    "constraint_variable_max_index": [],
                    "constraint_variable_max_values": [],
                    "weight": "1",                  # as Fraction; equal weights per instance
                }
                yield entry


class GlucoseCGMTask_withEvent_withLag(GlucoseCGMTask):

    
    def __init__(
        self,
        seed: int = None,
        event_columns: list = None,
        lag: int = 0,
        only_selected_event: bool = False,
        fixed_config: Optional[dict] = None,
        data_path: Optional[str] = None,
    ):
        """
        Initialize lag-adjusted event-focused CGM task.
        
        Parameters:
        -----------
        seed : int
            Random seed for reproducibility
        event_columns : list
            List of event column names to consider. Must have length >= 1.
            Options: ['Diet5Min', 'Med5Min', 'Exercise5Min']
        lag : int
            Lag adjustment for event positioning:
            - lag=0: event at observation point (boundary between past/future)
            - lag>0: event occurs `lag` timesteps after observation point (in future)
            - lag<0: event occurs `lag` timesteps before observation point (in past)
        only_selected_event : bool
            If True, only show the selected event type in covariates and scenario.
            Other event types will be hidden from the model. Default: False.
        fixed_config : dict
            Configuration overrides (context_length, prediction_length, etc.)
        data_path : str
            Path to the glucose data pickle file
        """
        if event_columns is None:
            event_columns = ['Diet5Min', 'Med5Min', 'Exercise5Min']
        
        if not isinstance(event_columns, list) or len(event_columns) < 1:
            raise ValueError("event_columns must be a list with at least one event type")
        
        # Validate event column names
        valid_events = {'Diet5Min', 'Med5Min', 'Exercise5Min'}
        invalid = set(event_columns) - valid_events
        if invalid:
            raise ValueError(f"Invalid event columns: {invalid}. Must be from {valid_events}")
        
        # Store child-specific attributes BEFORE calling super()
        self.event_columns = event_columns
        self.lag = int(lag)  # Store lag parameter
        self.only_selected_event = only_selected_event
        
        # Call parent constructor first - it will set up basic config and data_path
        super().__init__(seed=seed, fixed_config=fixed_config, data_path=data_path)
        
        # Now override/extend config if needed (after parent has set it up)
        # Parent already handles fixed_config updates, no need to duplicate
    
    
    def _find_valid_events_with_lag(self, item: pd.Series, C: int, H: int, n: int) -> list:
        """
        Find all valid event points considering lag adjustment.
        
        Simple logic:
        1. Find event at time `event_idx`
        2. Calculate observation point: `obs_point = event_idx + lag`
           - lag > 0: obs_point is AFTER the event (event in past relative to obs)
           - lag < 0: obs_point is BEFORE the event (event in future relative to obs)
           - lag = 0: obs_point is AT the event
        3. Check if obs_point has enough context: C <= obs_point < n - H
        
        Parameters:
        -----------
        item : pd.Series
            Data row containing event columns
        C : int
            Context length (past window size)
        H : int
            Horizon length (future window size)
        n : int
            Total length of the time series
            
        Returns:
        --------
        list: List of tuples (event_index, event_type) for valid events
        """
        valid_events = []
        
        for event_col in self.event_columns:
            event_data = item.get(event_col, {})
            
            # Parse event dictionary
            if isinstance(event_data, str) and event_data.strip():
                try:
                    event_data = ast.literal_eval(event_data)
                except:
                    continue
            
            if not isinstance(event_data, dict):
                continue
            
            # Find events with sufficient context considering lag
            for event_idx_str in event_data.keys():
                try:
                    event_idx = int(event_idx_str)
                    obs_point = event_idx + self.lag  # CORRECTED: was event_idx - self.lag
                    
                    # Simple constraint: obs_point needs C before and H after
                    if C <= obs_point < n - H:
                        valid_events.append((event_idx, event_col))
                        
                except:
                    continue
        
        return valid_events
    
    def random_instance(self):
        """
        Generate a random instance with event positioned according to lag parameter.
        
        The method:
        1. Loads all data and filters to series with valid events (considering lag)
        2. Randomly selects a patient with at least one valid event
        3. Randomly selects one event from that patient
        4. Creates a window with observation point at event_index - lag
        
        Sets:
          self.past_time  -> DataFrame (C, 1) ending at observation point
          self.future_time-> DataFrame (H, 1) starting at observation point
          self.c_cov      -> {"past": (C,K), "future": (H,K)}
          self.background, self.scenario -> strings with event and lag context
          self.selected_event -> dict with event details including lag info
        """
        df = self.load_data()
        C = int(self.cfg["context_length"])
        H = int(self.cfg["prediction_length"])
        
        # Find all rows with valid events considering lag
        valid_rows = []
        for idx, row in df.iterrows():
            y = self._safe_seq(row["target"])
            if y is None:
                continue
            n = len(y)
            if n < C + H:
                continue
            
            valid_events = self._find_valid_events_with_lag(row, C, H, n)
            if valid_events:
                # Make a copy of the row to avoid issues with views
                valid_rows.append((idx, row.copy(), valid_events))
        
        if not valid_rows:
            raise ValueError(f"No valid events found with sufficient context for event columns: {self.event_columns} with lag: {self.lag}")
        
        # Random selection using the provided seed
        if self.random is not None:
            if hasattr(self.random, 'choice'):
                # Use index-based selection to avoid numpy array conversion issues
                idx_to_select = self.random.choice(len(valid_rows))
                selected = valid_rows[idx_to_select]
            else:
                # Use index-based selection
                rng = np.random.default_rng(self._seed if hasattr(self, '_seed') else None) 
                idx_to_select = rng.integers(0, len(valid_rows))
                selected = valid_rows[idx_to_select]
        else:
            idx_to_select = np.random.randint(0, len(valid_rows))
            selected = valid_rows[idx_to_select]
        
        idx, item, valid_events = selected
        
        # Select a random event from this patient's valid events
        if self.random is not None:
            if hasattr(self.random, 'choice'):
                # Use index-based selection to avoid numpy array conversion issues
                event_to_select = self.random.choice(len(valid_events))
                event_idx, event_type = valid_events[event_to_select]
            else:
                rng = np.random.default_rng(self._seed if hasattr(self, '_seed') else None)
                event_to_select = rng.integers(0, len(valid_events))
                event_idx, event_type = valid_events[event_to_select]
        else:
            event_to_select = np.random.randint(0, len(valid_events))
            event_idx, event_type = valid_events[event_to_select]
        
        # Calculate observation point based on lag
        obs_point = event_idx + self.lag  # CORRECTED: was event_idx - self.lag
        
        # Store selected event info with lag details
        self.selected_event = {
            'index': event_idx,
            'type': event_type,
            'patient_id': item.get('PatientID', 'Unknown'),
            'lag': self.lag,
            'obs_point': obs_point,
            'event_relative_to_obs': 'at boundary' if self.lag == 0 else 
                                   f'{abs(self.lag)} steps in {"past" if self.lag > 0 else "future"}'
        }
        
        # ---- Parse and prepare data (similar to parent class) ----
        y = self._safe_seq(item["target"])
        n = len(y)
        
        # Create time series
        start_time = pd.to_datetime(item["start"])
        full_idx = pd.date_range(start=start_time, periods=n, freq="5min")
        y_series = pd.Series(y, index=full_idx, name="glucose_mg_dl")
        
        # Parse interventions
        diet_full = self._dense_from_sparse_dict(item.get("Diet5Min", None), n)
        med_full = self._dense_from_sparse_dict(item.get("Med5Min", None), n)
        ex_full = self._dense_from_sparse_dict(item.get("Exercise5Min", None), n)
        
        # ---- Create windows around the observation point (not the event) ----
        # Past window: [obs_point - C, obs_point)
        # Future window: [obs_point, obs_point + H)
        t = obs_point  # The observation point (different from event_idx when lag != 0)
        
        past_series, future_series = self._split_at_t(y_series, t, C, H)
        
        past_slice = slice(t - C, t)
        fut_slice = slice(t, t + H)
        
        past_diet = diet_full[past_slice]
        fut_diet = diet_full[fut_slice]
        past_med = med_full[past_slice]
        fut_med = med_full[fut_slice]
        past_ex = ex_full[past_slice]
        fut_ex = ex_full[fut_slice]
        
        # ---- Calendar covariates ----
        cal_past = self._calendar_covs(past_series.index)
        cal_fut = self._calendar_covs(future_series.index)
        
        # Ensure arrays are properly shaped before concatenation
        past_diet_col = np.asarray(past_diet).reshape(-1, 1)
        past_med_col = np.asarray(past_med).reshape(-1, 1)
        past_ex_col = np.asarray(past_ex).reshape(-1, 1)
        
        fut_diet_col = np.asarray(fut_diet).reshape(-1, 1)
        fut_med_col = np.asarray(fut_med).reshape(-1, 1)
        fut_ex_col = np.asarray(fut_ex).reshape(-1, 1)
        
        # Filter out non-selected events if only_selected_event is True
        if self.only_selected_event:
            selected_event_type = event_type.replace('5Min', '')  # 'Diet', 'Med', or 'Exercise'
            
            if selected_event_type != 'Diet':
                past_diet_col = np.zeros_like(past_diet_col)
                fut_diet_col = np.zeros_like(fut_diet_col)
            if selected_event_type != 'Med':
                past_med_col = np.zeros_like(past_med_col)
                fut_med_col = np.zeros_like(fut_med_col)
            if selected_event_type != 'Exercise':
                past_ex_col = np.zeros_like(past_ex_col)
                fut_ex_col = np.zeros_like(fut_ex_col)
        
        self.c_cov = {
            "past": np.concatenate(
                [cal_past,
                 past_diet_col,
                 past_med_col,
                 past_ex_col], axis=1
            ),
            "future": np.concatenate(
                [cal_fut,
                 fut_diet_col,
                 fut_med_col,
                 fut_ex_col], axis=1
            ),
        }
        
        # ---- Create background and scenario with lag information ----
        gender = item.get("Gender", "Unknown")
        yob = item.get("YearOfBirth", np.nan)
        disease_type = item.get("DiseaseType", "Unknown")
        tz = item.get("UserTimeZone", "Unknown")
        patient_id = item.get("PatientID", "Unknown")
        
        # Convert demographic info to readable format
        if pd.notna(yob):
            try:
                age = pd.Timestamp.now().year - int(yob)
                age_str = f"{age} years old"
            except:
                age_str = "age unknown"
        else:
            age_str = "age unknown"
        
        # Map gender codes to readable text
        gender_map = {1: "Male", 2: "Female", "1": "Male", "2": "Female"}
        gender_text = gender_map.get(gender, f"Gender code {gender}")
        
        # Map disease type codes to readable text
        disease_map = {1.0: "Type 1 diabetes", 2.0: "Type 2 diabetes", "1.0": "Type 1 diabetes", "2.0": "Type 2 diabetes"}
        disease_text = disease_map.get(disease_type, f"diabetes (type {disease_type})")
        
        # BACKGROUND: Static domain and patient context (no event-specific info)
        self.background = (
            "This is continuous glucose monitoring (CGM) data measuring blood glucose levels every 5 minutes. "
            f"The patient is a {gender_text}, {age_str}, with {disease_text}, located in {tz} timezone. "
            "The task is to forecast future glucose levels based on historical patterns and intervention events."
        )
        
        # SCENARIO: Event-specific information with lag context
        event_name = event_type.replace('5Min', '').lower()
        event_time = full_idx[event_idx]
        obs_time = full_idx[obs_point]
        fut_start, fut_end = future_series.index[0], future_series.index[-1]
        
        # Count future interventions
        def _count_pos(a): return int(np.sum(a > 0))
        future_interventions = []
        
        # If only_selected_event is True, only include the selected event type in scenario
        if self.only_selected_event:
            selected_event_type = event_type.replace('5Min', '')  # 'Diet', 'Med', or 'Exercise'
            if selected_event_type == 'Diet' and _count_pos(fut_diet):
                future_interventions.append(f"{_count_pos(fut_diet)} meal event{'s' if _count_pos(fut_diet) > 1 else ''}")
            elif selected_event_type == 'Med' and _count_pos(fut_med):
                future_interventions.append(f"{_count_pos(fut_med)} medication event{'s' if _count_pos(fut_med) > 1 else ''}")
            elif selected_event_type == 'Exercise' and _count_pos(fut_ex):
                future_interventions.append(f"{_count_pos(fut_ex)} exercise event{'s' if _count_pos(fut_ex) > 1 else ''}")
        else:
            # Include all event types as before
            if _count_pos(fut_diet): 
                future_interventions.append(f"{_count_pos(fut_diet)} meal event{'s' if _count_pos(fut_diet) > 1 else ''}")
            if _count_pos(fut_med): 
                future_interventions.append(f"{_count_pos(fut_med)} medication event{'s' if _count_pos(fut_med) > 1 else ''}")
            if _count_pos(fut_ex): 
                future_interventions.append(f"{_count_pos(fut_ex)} exercise event{'s' if _count_pos(fut_ex) > 1 else ''}")
        
        # Create scenario description with lag information
        scenario_parts = []
        
        if self.lag == 0:
            scenario_parts.append(f"A {event_name} event occurs at {event_time:%Y-%m-%d %H:%M}, marking the start of the forecast period.")
        elif self.lag > 0:
            lag_minutes = self.lag * 5  # 5-minute intervals
            lag_hours = lag_minutes // 60
            lag_mins = lag_minutes % 60
            lag_str = f"{lag_hours}h {lag_mins}min" if lag_hours > 0 else f"{lag_mins} minutes"
            scenario_parts.append(f"A {event_name} event occurred at {event_time:%Y-%m-%d %H:%M} ({lag_str} before the forecast start at {obs_time:%H:%M}).")
        else:
            lag_minutes = abs(self.lag) * 5  # 5-minute intervals  
            lag_hours = lag_minutes // 60
            lag_mins = lag_minutes % 60
            lag_str = f"{lag_hours}h {lag_mins}min" if lag_hours > 0 else f"{lag_mins} minutes"
            scenario_parts.append(f"A {event_name} event occurred at {event_time:%Y-%m-%d %H:%M} ({lag_str} after the forecast start at {obs_time:%H:%M}).")
        
        if future_interventions:
            intervention_text = ", ".join(future_interventions[:-1])
            if len(future_interventions) > 1:
                intervention_text += f" and {future_interventions[-1]}"
            else:
                intervention_text = future_interventions[0]
            scenario_parts.append(f"During the 2-hour forecast window ({fut_start:%H:%M} to {fut_end:%H:%M}), there will also be: {intervention_text}.")
        else:
            scenario_parts.append(f"No additional interventions are scheduled during the 2-hour forecast window ({fut_start:%H:%M} to {fut_end:%H:%M}).")
        
        self.scenario = " ".join(scenario_parts)
        
        # ---- Set required fields ----
        # Ensure frequency is preserved when creating DataFrames
        past_series.index.freq = "5min"
        future_series.index.freq = "5min"
        self.past_time = past_series.to_frame(name="glucose_mg_dl")
        self.future_time = future_series.to_frame(name="glucose_mg_dl")
        self.constraints = None
        self.region_of_interest = None
        self.roi_weight = 0.5  # Default ROI weight
        self.metric_constraint = None  # No constraints on glucose predictions
        
        self.patient_metadata = {
            "patient_id": item.get("PatientID", "Unknown"),
            "gender": gender,
            "year_of_birth": yob,
            "disease_type": disease_type,
            "timezone": tz,
            "cut_index": t,  # This is the observation point, not the event
            "event_details": {
                "type": event_type,
                "index": event_idx,
                "time": event_time.isoformat(),
                "lag": self.lag,
                "obs_point": obs_point,
                "obs_time": obs_time.isoformat()
            }
        }
        
        return self
    
    @property
    def seasonal_period(self) -> int:
        """
        Return the seasonal period for glucose CGM data.
        For 5-minute glucose data, we use 288 periods per day (24*60/5 = 288).
        """
        return 288  # 5-min intervals, 288 per day
    
    def evaluate_forecast(self, samples, task_instance=None) -> float:
        """Evaluate forecast using CRPS metric (compatibility method)."""
        # Handle case where models return (samples, extra_info) tuple
        if isinstance(samples, tuple):
            samples = samples[0]  # Extract the actual samples array
        
        result = self.evaluate(samples)
        if isinstance(result, dict):
            return result.get('crps', result.get('metric', float('nan')))
        return result


# Task registration for automatic discovery
__TASKS__ = [
    GlucoseCGMTask,
    GlucoseCGMTask_withEvent_withLag,
]

__CLUSTERS__ = [
    WeightCluster(
        weight=1,
        tasks=[
            GlucoseCGMTask,
            GlucoseCGMTask_withEvent_withLag,
        ],
    ),
]