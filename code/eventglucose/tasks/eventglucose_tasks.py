import ast
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import os
from ..config import DATA_STORAGE_PATH
from ..base import UnivariateCRPSTask
from . import WeightCluster
import json
import ast


    
BASE_PROMPT = """
I have a time series forecasting task for you.

Here is some context about the task. Make sure to factor in any background knowledge,
satisfy any constraints, and respect any scenarios.
<context>
{context}
</context>

Here is a historical time series in (timestamp, value) format:
<history>
{history}
</history>

Now please predict the value at the following timestamps: {pred_time}.

Return the forecast in (timestamp, value) format in between <forecast> and </forecast> tags.
Do not include any other information (e.g., comments) in the forecast.

Example:
<history>
(t1, v1)
(t2, v2)
(t3, v3)
</history>
<forecast>
(t4, v4)
(t5, v5)
</forecast>

"""


def select_files(
    base_dir,
    include: dict[str, list[str]] | None = None,
    suffix: str = ".pkl",
):
    """
    Return a list of file paths under `base_dir` whose names match the requested
    event types and subgroups.

    Filenames are expected like:
        WellDoc-testod-<EventType>-<D#>-Age<##>.pkl
      e.g. WellDoc-testod-Diet5Min-D1-Age40.pkl

    Parameters
    ----------
    base_dir : str | Path
        Directory containing the data files.
    include : dict with keys:
        - "eventtype": list[str] of event types to include, e.g. ["Diet5Min", "Med5Min", "Exercise5Min"]
        - "subgroup": list[str] of subgroup labels, e.g. ["D1-Age18", "D2-Age65"]
        If a key is missing or its list is empty, it's treated as "no filter" for that dimension.
    suffix : str
        File suffix to match (default ".pkl").

    Returns
    -------
    list[Path]
        Matching file paths.
    """
    base_dir = Path(base_dir)
    include = include or {}
    want_events = set(map(str.lower, include.get("eventtype", [])))
    want_groups = set(map(str.lower, include.get("subgroup", [])))

    matches = []
    for p in base_dir.glob(f"*{suffix}"):
        parts = p.name.split("-")
        # Expected formats:
        # Single: [WellDoc, testod, Diet5Min, D#, Age##.pkl]
        # Combo:  [WellDoc, testod, Diet5Min, Med5Min, D#, Age##.pkl]
        if len(parts) < 5:
            continue  # skip unexpected names

        # Find where subgroup starts (D1, D2, etc.)
        subgroup_start_idx = None
        for i in range(2, len(parts)):
            if parts[i].startswith(('D1', 'D2')):
                subgroup_start_idx = i
                break

        if subgroup_start_idx is None or subgroup_start_idx + 1 >= len(parts):
            continue  # malformed filename

        # Event type = everything between 'testod' and subgroup
        event = "-".join(parts[2:subgroup_start_idx])  # e.g. "Diet5Min" or "Diet5Min-Med5Min"

        # Subgroup = D# + Age##
        age_part = parts[subgroup_start_idx + 1].rsplit('.', 1)[0]  # Remove .pkl suffix
        subgroup = f"{parts[subgroup_start_idx]}-{age_part}"  # e.g. "D1-Age40"

        # Apply filters (case-insensitive). Empty sets mean "no filter".
        event_ok = (not want_events) or (event.lower() in want_events)
        group_ok = (not want_groups) or (subgroup.lower() in want_groups)

        if event_ok and group_ok:
            p = str(p).split("/")[-1]
            matches.append(p)

    return sorted(matches)


# ---- Example usage ----
# include = {
#     "eventtype": ["Diet5Min", "Exercise5Min"],
#     "subgroup": ["D1-Age18", "D2-Age65"]
# }
# files = select_files("/path/to/task_dataset", include)
# for f in files:
#     print(f)


def convert_event_info_to_dict(events):
    events = events.copy()
    # If the cell isn't a dict with 'event_info', leave it alone
    if not isinstance(events, dict) or 'event_info' not in events:
        return events

    event_info = events['event_info']

    # Normalize None -> empty list
    if event_info is None:
        events['event_info'] = []
        return events

    if len(event_info) == 0:
        events['event_info'] = []
        return events

    converted = []
    for e in event_info:
        if isinstance(e, str):
            # try to parse a stringified dict; if it fails, keep original string
                e = json.loads(e)

        elif isinstance(e, dict):
            pass

        converted.append(e)

    # Prefer a list of dicts; avoids awkward NumPy object arrays
    events['event_info'] = np.array(converted)
    return events


def load_df_lts_data(task_config: Dict) -> pd.DataFrame:
    """Load the dataframe (pickle expected)."""

    if 'data_lts_folder' in task_config:
        if 'subgroup' in task_config or 'eventtype' in task_config:
            print('data_lts_folder', task_config['data_lts_folder'])
            data_lts_files = select_files(task_config['data_lts_folder'], include=task_config, suffix=".pkl")
            print(data_lts_files)
        else:
            data_lts_files = os.listdir(task_config['data_lts_folder'])
        df_li = []
        for data_lts_file in data_lts_files:
            full_path = os.path.join(task_config['data_lts_folder'], data_lts_file,)
            try:
                print(full_path)
                df = pd.read_pickle(full_path)
                df['events'] = df['events'].apply(lambda x: convert_event_info_to_dict(x))  
                df_li.append(df)
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                continue
        df = pd.concat(df_li).reset_index(drop=True)

    elif 'data_path' in task_config :
        data_path = task_config.get("data_path")
        df = pd.read_pickle(data_path)

    else:
        raise ValueError("data_lts_folder or data_path must be provided")
    

    
    print(df.columns)
    # required = {"item_id", "target", "start_time"}
    # missing = required - set(df.columns)
    # if missing:
    #     raise ValueError(f"Missing required columns: {missing}")
    return df




class EventCGMTask_withEvent_withLag(UnivariateCRPSTask):
    """
    Event-aware CGM forecasting task with lag adjustment.

    This task extends the base univariate CRPS task to:
    - Focus on windows around intervention events (meals, medication, exercise)
    - Support lag adjustment for temporal event positioning
    - Optionally filter to show only selected event types
    - Provide event-specific context in background/scenario descriptions
    """

    _context_sources = UnivariateCRPSTask._context_sources + ["c_cov"]
    _skills = UnivariateCRPSTask._skills + ["instruction following"]
    __version__ = "0.2.0"

    def get_task_config(self) -> Dict:
        """
        Get task configuration.

        Returns instance-level task_config if set, otherwise returns empty dict.
        Subclasses can override this to provide class-level configs.
        """
        return getattr(self, 'task_config', {})

    # -------------------------
    # Data Loading & Parsing
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

    

    # -------------------------
    # Feature Builders
    # -------------------------
    def _calendar_covs(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """Day-of-week + hour-of-day one-hots (known-in-advance covariates)."""
        if not self.cfg["use_calendar_covs"]:
            return np.zeros((len(idx), 0), dtype=float)
        dow = np.eye(7, dtype=float)[idx.dayofweek.values]  # (N, 7)
        hr = np.eye(24, dtype=float)[idx.hour.values]       # (N, 24)
        return np.concatenate([dow, hr], axis=1)            # (N, 31)

    @staticmethod
    def _dense_from_sparse_dict(obj, length: int) -> np.ndarray:
        """
        Convert sparse event dictionaries to dense 0/1 arrays.
        
        Diet/Med/Exercise columns often look like sparse dicts keyed by index:
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
        fut = series.iloc[t:t + H]
        return past, fut

    # -------------------------
    # Main Instance Builder
    # -------------------------
    def random_instance(self):
        """
        Generate a random instance with event positioned according to lag parameter.

        The method:
        1. Loads pre-filtered data (already filtered by event type and subgroup)
        2. Randomly selects a row (each row is a pre-extracted valid window)
        3. Uses prediction_time_idx from the pkl file as the base event time
        4. Applies lag to calculate observation point: obs_point = prediction_time_idx + lag
        5. Creates past/future windows and covariates centered around obs_point

        Lag semantics:
          - lag = 0:  observation point AT the event (prediction_time_idx)
          - lag > 0:  observation point AFTER the event (event in the past)
          - lag < 0:  observation point BEFORE the event (event in the future)

        Sets:
          self.past_time  -> DataFrame (C, 1) ending at observation point
          self.future_time-> DataFrame (H, 1) starting at observation point
          self.c_cov      -> {"past": (C,K), "future": (H,K)}
          self.background, self.scenario -> strings with event and lag context
          self.selected_event -> dict with event details including lag info
        """
        self.task_config = self.get_task_config()
        task_config = self.task_config 
        df = load_df_lts_data(task_config)

        self.lag = task_config["lag"]
        self.cfg = task_config
        self.prediction_length = task_config["prediction_length"]

        C = int(task_config["context_length"])
        H = int(task_config["prediction_length"])

        # Directly select a random row since data is already filtered and validated
        if self.random is not None:
            if hasattr(self.random, 'choice'):
                idx_to_select = self.random.choice(len(df))
            else:
                rng = np.random.default_rng(self._seed if hasattr(self, '_seed') else None)
                idx_to_select = rng.integers(0, len(df))
        else:
            idx_to_select = np.random.randint(0, len(df))

        item = df.iloc[idx_to_select]

        # Get the pre-computed prediction time index from the pkl file
        prediction_time_idx = item.get("prediction_time_idx")
        if prediction_time_idx is None:
            raise ValueError("Column 'prediction_time_idx' not found in data. This column is required.")

        prediction_time_idx = int(prediction_time_idx)

        # Calculate observation point based on lag
        # lag > 0: observation point is AFTER the event (event happened in the past)
        # lag < 0: observation point is BEFORE the event (event will happen in the future)
        # lag = 0: observation point is AT the event
        obs_point = prediction_time_idx + self.lag

        # Extract event information from the row
        # Use event_columns if specified, otherwise fall back to eventtype config
        event_type = task_config.get("event_columns", task_config.get("eventtype", ["Diet5Min"]))[0]

        # Store selected event info
        self.selected_event = {
            'prediction_time_idx': prediction_time_idx,
            'type': event_type,
            'patient_id': item.get('PatientID', 'Unknown'),
            'lag': self.lag,
            'obs_point': obs_point,
            'event_relative_to_obs': 'at boundary' if self.lag == 0 else
                                   f'{abs(self.lag)} steps in {"past" if self.lag > 0 else "future"}'
        }

        # Extract detailed event info from the events column
        events_data = item.get('events', {})
        if isinstance(events_data, dict) and 'event_info' in events_data:
            event_info_arr = events_data.get('event_info', [])
            if event_info_arr is not None and len(event_info_arr) > 0:
                # Get the first event's details (typically there's one event per sample)
                event_info = event_info_arr[0] if len(event_info_arr) > 0 else {}
                if isinstance(event_info, dict):
                    self.selected_event['event_info'] = event_info

        # Parse target sequence
        y = self._safe_seq(item["target_sequence"])
        if y is None:
            raise ValueError("Could not parse 'target_sequence' as numeric sequence.")
        n = len(y)
        
        # Create time series
        start_time = pd.to_datetime(item["start_time"])
        full_idx = pd.date_range(start=start_time, periods=n, freq="5min")
        y_series = pd.Series(y, index=full_idx, name="glucose_mg_dl")

        # Parse intervention sequences (already aligned with target_sequence)
        diet_full = self._safe_seq(item.get("Diet5Min", None))
        if diet_full is None or len(diet_full) != n:
            diet_full = np.zeros(n, dtype=float)

        med_full = self._safe_seq(item.get("Med5Min", None))
        if med_full is None or len(med_full) != n:
            med_full = np.zeros(n, dtype=float)

        ex_full = self._safe_seq(item.get("Exercise5Min", None))
        if ex_full is None or len(ex_full) != n:
            ex_full = np.zeros(n, dtype=float)

        # Split at observation point (event_idx + lag)
        t = obs_point
        past_series, future_series = self._split_at_t(y_series, t, C, H)

        past_slice = slice(t - C, t)
        fut_slice = slice(t, t + H)

        past_diet = diet_full[past_slice]
        fut_diet = diet_full[fut_slice]
        past_med = med_full[past_slice]
        fut_med = med_full[fut_slice]
        past_ex = ex_full[past_slice]
        fut_ex = ex_full[fut_slice]
        
        # Calendar covariates
        cal_past = self._calendar_covs(past_series.index)
        cal_fut = self._calendar_covs(future_series.index)
        
        # Reshape intervention arrays
        past_diet_col = np.asarray(past_diet).reshape(-1, 1)
        past_med_col = np.asarray(past_med).reshape(-1, 1)
        past_ex_col = np.asarray(past_ex).reshape(-1, 1)
        
        fut_diet_col = np.asarray(fut_diet).reshape(-1, 1)
        fut_med_col = np.asarray(fut_med).reshape(-1, 1)
        fut_ex_col = np.asarray(fut_ex).reshape(-1, 1)

        # Stack covariates
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
        
        # Create background and scenario descriptions
        gender = item.get("Gender", "Unknown")
        yob = item.get("YearOfBirth", np.nan)
        disease_type = item.get("DiseaseType", "Unknown")
        tz = item.get("UserTimeZone", "Unknown")
        
        # Process demographics
        if pd.notna(yob):
            try:
                age = pd.Timestamp.now().year - int(yob)
                age_str = f"{age} years old"
            except:
                age_str = "age unknown"
        else:
            age_str = "age unknown"
        
        # Map codes to readable text
        gender_map = {1: "Male", 2: "Female", "1": "Male", "2": "Female"}
        gender_text = gender_map.get(gender, f"Gender code {gender}")
        
        disease_map = {1.0: "Type 1 diabetes", 2.0: "Type 2 diabetes", 
                      "1.0": "Type 1 diabetes", "2.0": "Type 2 diabetes"}
        disease_text = disease_map.get(disease_type, f"diabetes (type {disease_type})")
        
        # BACKGROUND: Static patient context
        self.background = (
            "This is continuous glucose monitoring (CGM) data measuring blood glucose levels every 5 minutes. "
            f"The patient is a {gender_text}, {age_str}, with {disease_text}, located in {tz} timezone. "
            "The task is to forecast future glucose levels based on historical patterns and intervention events."
        )
        
        # SCENARIO: Event-specific information with lag context
        event_name = event_type.replace('5Min', '').lower()
        event_time = full_idx[prediction_time_idx]
        obs_time = full_idx[obs_point]
        fut_start, fut_end = future_series.index[0], future_series.index[-1]
        
        # Count future interventions
        def _count_pos(a): return int(np.sum(a > 0))
        future_interventions = []

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
            lag_minutes = self.lag * 5
            lag_hours = lag_minutes // 60
            lag_mins = lag_minutes % 60
            lag_str = f"{lag_hours}h {lag_mins}min" if lag_hours > 0 else f"{lag_mins} minutes"
            scenario_parts.append(f"A {event_name} event occurred at {event_time:%Y-%m-%d %H:%M} ({lag_str} before the forecast start at {obs_time:%H:%M}).")
        else:
            lag_minutes = abs(self.lag) * 5
            lag_hours = lag_minutes // 60
            lag_mins = lag_minutes % 60
            lag_str = f"{lag_hours}h {lag_mins}min" if lag_hours > 0 else f"{lag_mins} minutes"
            scenario_parts.append(f"A {event_name} event will occur at {event_time:%Y-%m-%d %H:%M} ({lag_str} after the forecast start at {obs_time:%H:%M}).")
        
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
        
        # Set required fields
        past_series.index.freq = "5min"
        future_series.index.freq = "5min"
        self.past_time = past_series.to_frame(name="glucose_mg_dl")
        self.future_time = future_series.to_frame(name="glucose_mg_dl")
        self.constraints = None
        self.region_of_interest = None
        self.roi_weight = 0.5
        self.metric_constraint = None



        task_config = self.task_config 

        
        
        # Store metadata
        self.patient_metadata = {
            "patient_id": item.get("PatientID", "Unknown"),
            "gender": gender,
            "year_of_birth": yob,
            "disease_type": disease_type,
            "timezone": tz,
            "cut_index": t,
            "event_details": {
                "type": event_type,
                "prediction_time_idx": prediction_time_idx,
                "time": event_time.isoformat(),
                "lag": self.lag,
                "obs_point": obs_point,
                "obs_time": obs_time.isoformat()
            }
        }


        ############################################################
        # Prompt generation: Support both new string-based config and legacy function-based config

        # New approach: Task config provides 'prompt_level' string
        prompt_level = task_config.get("prompt_level", None)

        if prompt_level is not None:
            # Create prompt function from level string
            from ..prompts.make_prompt import MakePrompt

            # Extract optional prompt configuration
            prompt_config = {
                'time_format': task_config.get('prompt_time_format', 'time_value_pairs'),
                'max_history_points': task_config.get('prompt_max_history', None),
                'max_digits': task_config.get('prompt_max_digits', 1),
                'include_units': task_config.get('prompt_include_units', True),
            }

            # Create prompt maker and generate prompt
            prompt_maker = MakePrompt(**prompt_config)
            fn_prompt = prompt_maker.get_prompt_fn(level=prompt_level)
            prompt = fn_prompt(self) # prompt text
            self.prompt = prompt

        # Legacy approach: Task config provides 'fn_prompt' function directly
        elif task_config.get("fn_prompt", None) is not None:
            fn_prompt = task_config["fn_prompt"]
            prompt = fn_prompt(self)
            self.prompt = prompt

        # No prompt configuration provided
        else:
            prompt = None
            self.prompt = None
        ############################################################
        
        return self
    
    
    # -------------------------
    # Properties
    # -------------------------
    @property
    def seasonal_period(self) -> int:
        """
        Return the seasonal period for glucose CGM data.
        For 5-minute glucose data, we use 288 periods per day (24*60/5 = 288).
        """
        return 288

    # -------------------------
    # Evaluation Methods
    # -------------------------
    def evaluate_forecast(self, samples, task_instance=None) -> float:
        """
        Evaluate forecast using CRPS metric.
        
        Parameters:
        -----------
        samples : np.ndarray or tuple
            Forecast samples of shape (n_samples, prediction_length, 1)
            or tuple where first element is the samples array
        task_instance : optional
            Task instance (for compatibility)
            
        Returns:
        --------
        float: CRPS score
        """
        # Handle case where models return (samples, extra_info) tuple
        if isinstance(samples, tuple):
            samples = samples[0]
        
        result = self.evaluate(samples)
        if isinstance(result, dict):
            return result.get('crps', result.get('metric', float('nan')))
        return result

    # -------------------------
    # Convenience Methods
    # -------------------------
    def get_intervention_context(self) -> Dict[str, np.ndarray]:
        """Return interventions split into past/future arrays (0/1 flags)."""
        return {
            "past_diet": self.c_cov["past"][:, -3],
            "future_diet": self.c_cov["future"][:, -3],
            "past_medication": self.c_cov["past"][:, -2],
            "future_medication": self.c_cov["future"][:, -2],
            "past_exercise": self.c_cov["past"][:, -1],
            "future_exercise": self.c_cov["future"][:, -1],
        }

    def get_numpy_payload(self) -> Dict[str, np.ndarray]:
        """Handy shapes/arrays for quick debugging in notebooks."""
        return {
            "past_y": self.past_time.values.squeeze(-1),
            "future_h": np.array([len(self.future_time)], int),
            "past_cov": self.c_cov["past"],
            "future_cov": self.c_cov["future"],
        }

    def iter_all_instances(self, stride: int = 1):
        """
        Yields entries compatible with roi_crps(entry, forecast) for evaluation.

        Each entry corresponds to one pre-extracted window from the filtered data.

        Parameters:
        -----------
        stride : int
            Step size for iteration (default: 1 = all instances)

        Yields:
        -------
        dict: Entry with past_time, future_time, and metric metadata
        """
        import numpy as np, pandas as pd

        df = load_df_lts_data(self.get_task_config())
        C = int(self.cfg["context_length"])
        H = int(self.cfg["prediction_length"])

        for idx, item in df.iterrows():
            # Skip rows based on stride
            if idx % stride != 0:
                continue

            # Parse target sequence
            y = self._safe_seq(item["target_sequence"])
            if y is None or len(y) < C + H:
                continue

            n = len(y)

            # Get prediction_time_idx from the pkl file
            prediction_time_idx = item.get("prediction_time_idx")
            if prediction_time_idx is None:
                continue  # Skip rows without prediction_time_idx

            prediction_time_idx = int(prediction_time_idx)
            obs_point = prediction_time_idx + self.lag

            start_time = pd.to_datetime(item["start_time"])
            idx_range = pd.date_range(start=start_time, periods=n, freq="5min")

            # Create past and future windows
            past = pd.Series(y[obs_point - C:obs_point],
                           index=idx_range[obs_point - C:obs_point],
                           name="glucose_mg_dl").to_frame()
            fut = pd.Series(y[obs_point:obs_point + H],
                          index=idx_range[obs_point:obs_point + H],
                          name="glucose_mg_dl").to_frame()

            # Create entry dict for evaluation
            entry = {
                "name": "glucose_cgm_event_lag",
                "seed": int(hash((item.get("item_id", idx), prediction_time_idx, self.lag)) & 0x7FFFFFFF),
                "past_time": past.to_json(),
                "future_time": fut.to_json(),
                "region_of_interest": [],
                "metric_scaling": 1.0,
                "constraint_min": -np.inf,
                "constraint_max": np.inf,
                "constraint_variable_max_index": [],
                "constraint_variable_max_values": [],
                "weight": "1",
                "prediction_time_idx": prediction_time_idx,
                "lag": self.lag,
            }
            yield entry
