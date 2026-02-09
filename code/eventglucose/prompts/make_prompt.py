"""
Updated: 2025-10-13 New Version of Prompt Making Function compared to make_prompt_v1.py
Improvements: Works with all task definitions in python scripts to provide prompts of different levels of granularity.

This module provides a flexible MakePrompt class that generates prompts at four granularity levels:
0. No context level (noctx): Time series only, no patient or event information
1. Profile level: Includes patient biographic information
2. Medium event level: Includes event timing information
3. Detailed event level: Includes full event details and information

Usage:

Method 1 - String-based configuration (RECOMMENDED for task classes):
    In your task class's get_task_config():
        return {
            'prompt_level': 'medium_event',  # Just specify the level as a string!
            'prompt_max_digits': 1,           # Optional configuration
            # ... other config ...
        }

    The EventCGMTask_withEvent_withLag.random_instance() method will automatically
    create the prompt function and generate the prompt.

Method 2 - Direct instantiation (for notebooks/testing):
    prompt_maker = MakePrompt(time_format="time_value_pairs", max_history_points=50)
    fn_prompt = prompt_maker.get_prompt_fn(level="detailed_event")
    prompt = fn_prompt(task_instance)

Method 3 - Convenience function (legacy compatibility):
    from eventglucose.prompts import create_prompt_fn
    fn_prompt = create_prompt_fn(level="medium_event", max_digits=1)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Callable
from datetime import datetime


class MakePrompt:
    """
    Flexible prompt generator for glucose forecasting tasks with configurable granularity levels.

    This class provides four levels of prompt generation:
    - noctx: Time series only, no context (minimal baseline)
    - profile: Basic task with patient demographic information
    - medium_event: Adds event timing information (when events occur)
    - detailed_event: Adds complete event details (what, when, and attributes)

    Parameters
    ----------
    time_format : str, default="time_value_pairs"
        Format for time series representation:
        - "time_value_pairs": (timestamp, value) pairs
        - "value_sequence": Comma-separated values with context
    max_history_points : int, optional
        Maximum number of historical points to include. If None, includes all points.
    max_digits : int, default=1
        Number of decimal digits for glucose values
    include_units : bool, default=True
        Whether to include "mg/dL" units in the prompt
    """

    # Base prompt template (shared across all levels)
    BASE_TEMPLATE = """I have a time series forecasting task for you.

{context_section}

Here is a historical time series of glucose measurements in {data_format} format:
<history>
{history}
</history>

Please predict the glucose values at the following timestamps:
{pred_time}

Return your forecast in (timestamp, value) format between <forecast> and </forecast> tags.
Do not include any other information (e.g., comments, explanations) in the forecast.

Example format:
<forecast>
(2024-01-01 08:00:00, 120.5)
(2024-01-01 08:05:00, 122.3)
(2024-01-01 08:10:00, 125.1)
</forecast>
"""

    def __init__(
        self,
        time_format: str = "time_value_pairs",
        max_history_points: Optional[int] = None,
        max_digits: int = 1,
        include_units: bool = True
    ):
        self.time_format = time_format
        self.max_history_points = max_history_points
        self.max_digits = max_digits
        self.include_units = include_units

        # Validate time format
        valid_formats = ["time_value_pairs", "value_sequence"]
        if self.time_format not in valid_formats:
            raise ValueError(f"time_format must be one of {valid_formats}, got '{self.time_format}'")

    # =========================================================================
    # Data Extraction Methods
    # =========================================================================

    def _extract_data(self, source: Union[Any, Dict]) -> Dict[str, Any]:
        """
        Extract required data from task instance or dictionary.

        Parameters
        ----------
        source : task instance or dict
            Data source containing time series and metadata

        Returns
        -------
        dict
            Standardized data dictionary with keys:
            - hist_time: Historical timestamps (array)
            - hist_value: Historical glucose values (array)
            - pred_time: Prediction timestamps (array)
            - background: Background context (str)
            - scenario: Scenario description (str)
            - patient_metadata: Patient information (dict)
            - event_details: Event-specific details (dict)
        """
        data = {}

        # Handle dictionary input
        if isinstance(source, dict):
            data['hist_time'] = source.get('hist_time', [])
            data['hist_value'] = source.get('hist_value', [])
            data['pred_time'] = source.get('pred_time', [])
            data['background'] = source.get('background', '')
            data['scenario'] = source.get('scenario', '')
            data['patient_metadata'] = source.get('patient_metadata', {})
            data['event_details'] = source.get('event_details', {})

        # Handle task instance input
        else:
            # Extract time series data
            if hasattr(source, 'past_time') and source.past_time is not None:
                data['hist_time'] = source.past_time.index.strftime("%Y-%m-%d %H:%M:%S").values
                data['hist_value'] = source.past_time.values[:, -1]  # Last column is target
            else:
                data['hist_time'] = []
                data['hist_value'] = []

            if hasattr(source, 'future_time') and source.future_time is not None:
                data['pred_time'] = source.future_time.index.strftime("%Y-%m-%d %H:%M:%S").values
            else:
                data['pred_time'] = []

            # Extract context information
            data['background'] = getattr(source, 'background', '')
            data['scenario'] = getattr(source, 'scenario', '')
            data['patient_metadata'] = getattr(source, 'patient_metadata', {})

            # Extract event details
            if hasattr(source, 'selected_event'):
                data['event_details'] = source.selected_event
            else:
                data['event_details'] = {}

        return data

    def _format_history(self, hist_time: np.ndarray, hist_value: np.ndarray) -> tuple[str, str]:
        """
        Format historical time series according to specified format.

        Returns
        -------
        tuple[str, str]
            (formatted_history, format_description)
        """
        # Apply max_history_points limit if specified
        if self.max_history_points and len(hist_time) > self.max_history_points:
            hist_time = hist_time[-self.max_history_points:]
            hist_value = hist_value[-self.max_history_points:]

        if self.time_format == "time_value_pairs":
            history_lines = []
            for t, v in zip(hist_time, hist_value):
                value_str = f"{v:.{self.max_digits}f}"
                history_lines.append(f"({t}, {value_str})")

            history = "\n".join(history_lines)
            format_desc = "(timestamp, value) pairs"

        elif self.time_format == "value_sequence":
            start_time = hist_time[0]
            end_time = hist_time[-1]
            values_str = ", ".join(f"{v:.{self.max_digits}f}" for v in hist_value)

            history = (
                f"Start time: {start_time}\n"
                f"End time: {end_time}\n"
                f"Interval: 5 minutes\n"
                f"Values: {values_str}"
            )
            format_desc = "value sequence with timestamps"

        else:
            raise ValueError(f"Unknown time_format: {self.time_format}")

        return history, format_desc

    def _extract_patient_info(self, patient_metadata: Dict) -> str:
        """
        Extract and format patient biographical information.

        Returns
        -------
        str
            Formatted patient information string
        """
        if not patient_metadata:
            return "Patient information not available."

        info_parts = []

        # Extract age
        if 'year_of_birth' in patient_metadata:
            yob = patient_metadata['year_of_birth']
            if pd.notna(yob):
                try:
                    current_year = datetime.now().year
                    age = current_year - int(yob)
                    info_parts.append(f"Age: {age} years")
                except:
                    pass

        # Extract gender
        if 'gender' in patient_metadata:
            gender = patient_metadata['gender']
            gender_map = {1: "Male", 2: "Female", "1": "Male", "2": "Female"}
            gender_text = gender_map.get(gender, str(gender))
            info_parts.append(f"Gender: {gender_text}")

        # Extract disease type
        if 'disease_type' in patient_metadata:
            disease = patient_metadata['disease_type']
            disease_map = {
                1.0: "Type 1 Diabetes",
                2.0: "Type 2 Diabetes",
                "1.0": "Type 1 Diabetes",
                "2.0": "Type 2 Diabetes"
            }
            disease_text = disease_map.get(disease, f"Diabetes (type {disease})")
            info_parts.append(f"Condition: {disease_text}")

        # Extract timezone
        if 'timezone' in patient_metadata:
            tz = patient_metadata['timezone']
            if tz and tz != "Unknown":
                info_parts.append(f"Timezone: {tz}")

        # Extract patient ID if available
        if 'patient_id' in patient_metadata:
            pid = patient_metadata['patient_id']
            if pid and pid != "Unknown":
                info_parts.append(f"Patient ID: {pid}")

        if info_parts:
            return "Patient Profile: " + ", ".join(info_parts)
        else:
            return "Patient information not available."

    def _extract_event_timing(self, scenario: str, event_details: Dict) -> str:
        """
        Extract medium-level event information (timing only).

        Returns
        -------
        str
            Event timing information
        """
        if not scenario:
            return "No event information available."

        # Use scenario which already contains event timing info
        # Remove detailed descriptions and keep only timing
        lines = scenario.split('.')
        timing_lines = []

        for line in lines:
            line = line.strip()
            # Keep lines that mention event occurrence/timing
            if any(keyword in line.lower() for keyword in ['event occurs', 'event occurred', 'event will occur', 'scheduled']):
                timing_lines.append(line)

        if timing_lines:
            return "Event Timing: " + ". ".join(timing_lines) + "."
        else:
            # Fallback to event_details if available
            if event_details and 'time' in event_details:
                event_type = event_details.get('type', 'intervention')
                event_time = event_details.get('time', '')
                return f"Event Timing: A {event_type} event at {event_time}."
            return "No event timing information available."

    def _extract_event_details(self, scenario: str, event_details: Dict) -> str:
        """
        Extract detailed event information (timing + details).

        Returns
        -------
        str
            Complete event information
        """
        if not scenario:
            return "No event information available."

        # The scenario already contains detailed event information
        return "Event Information: " + scenario

    # =========================================================================
    # New Event Extraction Methods (for newmedium and newdetail levels)
    # =========================================================================

    def _round_value(self, value: float, max_decimals: int = 2) -> str:
        """Round a value to at most max_decimals, removing trailing zeros."""
        if value is None:
            return ""
        rounded = round(value, max_decimals)
        # Remove trailing zeros and unnecessary decimal point
        if rounded == int(rounded):
            return str(int(rounded))
        return f"{rounded:.{max_decimals}f}".rstrip('0').rstrip('.')

    def _extract_event_timing_new(self, scenario: str, event_details: Dict) -> str:
        """
        Extract new medium-level event information (timing + key metrics).

        For diet events: includes time, calories, and carbs.
        For other events: includes time and key attributes.

        Returns
        -------
        str
            Event timing and key metrics
        """
        parts = []

        # Extract timing from scenario
        if scenario:
            lines = scenario.split('.')
            for line in lines:
                line = line.strip()
                if any(keyword in line.lower() for keyword in ['event occurs', 'event occurred', 'event will occur', 'scheduled']):
                    parts.append(f"Timing: {line}.")
                    break

        # Add key metrics from event_info (calories and carbs for diet)
        if event_details and 'event_info' in event_details:
            event_info = event_details['event_info']
            if isinstance(event_info, dict):
                event_type = event_info.get('event_type', event_details.get('type', ''))

                if 'Diet' in event_type:
                    # Add meal name
                    food_name = event_info.get('FoodName', '')
                    if food_name:
                        parts.append(f"Meal: {food_name}")

                    # Add key metrics (calories and carbs)
                    metrics = []
                    if 'Calories' in event_info and event_info['Calories'] is not None:
                        metrics.append(f"Calories: {round(event_info['Calories'])}")
                    if 'Carbs' in event_info and event_info['Carbs'] is not None:
                        metrics.append(f"Carbs: {round(event_info['Carbs'], 1)}g")
                    if metrics:
                        parts.append(", ".join(metrics))

                elif 'Med' in event_type:
                    med_name = event_info.get('MedicationName', event_info.get('medication_name', ''))
                    if med_name:
                        parts.append(f"Medication: {med_name}")

                elif 'Exercise' in event_type:
                    import math
                    exercise_type = event_info.get('ExerciseType', event_info.get('exercise_type', ''))
                    duration = event_info.get('ExerciseDuration', event_info.get('Duration', event_info.get('duration', '')))
                    calories = event_info.get('CaloriesBurned', '')
                    if exercise_type:
                        parts.append(f"Exercise type: {exercise_type}")
                    if duration and not (isinstance(duration, float) and math.isnan(duration)):
                        parts.append(f"Duration: {round(duration)} min")
                    if calories and not (isinstance(calories, float) and math.isnan(calories)) and calories > 0:
                        parts.append(f"Calories burned: {round(calories)}")

        if not parts:
            # Fallback
            if event_details and 'time' in event_details:
                event_type = event_details.get('type', 'intervention')
                event_time = event_details.get('time', '')
                return f"Event: A {event_type} event at {event_time}."
            return "No event information available."

        return "Event Information: " + " ".join(parts)

    def _extract_event_details_new(self, scenario: str, event_details: Dict) -> str:
        """
        Extract new detailed event information (timing + full details).

        Returns
        -------
        str
            Complete event information including nutritional/medication/exercise details
        """
        parts = []

        # Add scenario (timing information)
        if scenario:
            parts.append(f"Timing: {scenario}")

        # Extract and format detailed event info
        if event_details and 'event_info' in event_details:
            event_info = event_details['event_info']
            if isinstance(event_info, dict):
                event_type = event_info.get('event_type', event_details.get('type', 'Unknown'))

                # Format based on event type
                if 'Diet' in event_type:
                    details = self._format_diet_event_new(event_info)
                elif 'Med' in event_type:
                    details = self._format_medication_event(event_info)
                elif 'Exercise' in event_type:
                    details = self._format_exercise_event(event_info)
                else:
                    details = self._format_generic_event(event_info)

                if details:
                    parts.append(details)

        if not parts:
            return "No event information available."

        return "Event Information:\n" + "\n".join(parts)

    def _format_diet_event_new(self, event_info: Dict) -> str:
        """Format diet/meal event details with values rounded to 2 decimals."""
        details = []

        # Food name
        food_name = event_info.get('FoodName', '')
        if food_name:
            details.append(f"Meal: {food_name}")

        # Activity type (e.g., BeforeDinner, Breakfast)
        activity = event_info.get('ActivityType', '')
        if activity:
            details.append(f"Meal type: {activity}")

        # Nutritional information (rounded to 2 decimal places max)
        nutrition = []
        if 'Carbs' in event_info and event_info['Carbs'] is not None:
            nutrition.append(f"Carbs: {self._round_value(event_info['Carbs'])}g")
        if 'Calories' in event_info and event_info['Calories'] is not None:
            nutrition.append(f"Calories: {self._round_value(event_info['Calories'])}")
        if 'Protein' in event_info and event_info['Protein'] is not None:
            nutrition.append(f"Protein: {self._round_value(event_info['Protein'])}g")
        if 'Fat' in event_info and event_info['Fat'] is not None:
            nutrition.append(f"Fat: {self._round_value(event_info['Fat'])}g")
        if 'Fiber' in event_info and event_info['Fiber'] is not None:
            nutrition.append(f"Fiber: {self._round_value(event_info['Fiber'])}g")
        if 'Sugar' in event_info and event_info['Sugar'] is not None:
            nutrition.append(f"Sugar: {self._round_value(event_info['Sugar'])}g")

        if nutrition:
            details.append("Nutrition: " + ", ".join(nutrition))

        return "\n".join(details) if details else ""

    def _format_medication_event(self, event_info: Dict) -> str:
        """Format medication event details."""
        details = []

        med_name = event_info.get('MedicationName', event_info.get('medication_name', ''))
        if med_name:
            details.append(f"Medication: {med_name}")

        dosage = event_info.get('Dosage', event_info.get('dosage', ''))
        if dosage:
            details.append(f"Dosage: {dosage}")

        med_type = event_info.get('MedicationType', event_info.get('medication_type', ''))
        if med_type:
            details.append(f"Type: {med_type}")

        return "\n".join(details) if details else ""

    def _format_exercise_event(self, event_info: Dict) -> str:
        """Format exercise event details."""
        import math
        details = []

        # Exercise type
        exercise_type = event_info.get('ExerciseType', event_info.get('exercise_type', ''))
        if exercise_type:
            details.append(f"Exercise type: {exercise_type}")

        # Duration - check both field name variants
        duration = event_info.get('ExerciseDuration', event_info.get('Duration', event_info.get('duration', '')))
        if duration and not (isinstance(duration, float) and math.isnan(duration)):
            details.append(f"Duration: {self._round_value(duration)} minutes")

        # Intensity - check both field name variants
        intensity = event_info.get('ExerciseIntensity', event_info.get('Intensity', event_info.get('intensity', '')))
        if intensity and not (isinstance(intensity, float) and math.isnan(intensity)):
            details.append(f"Intensity: {intensity}")

        # Calories burned
        calories = event_info.get('CaloriesBurned', event_info.get('calories_burned', ''))
        if calories and not (isinstance(calories, float) and math.isnan(calories)):
            if calories > 0:
                details.append(f"Calories burned: {self._round_value(calories)}")

        # Distance
        distance = event_info.get('DistanceInMeters', event_info.get('distance', ''))
        if distance and not (isinstance(distance, float) and math.isnan(distance)):
            if distance > 0:
                details.append(f"Distance: {self._round_value(distance)} meters")

        return "\n".join(details) if details else ""

    def _format_generic_event(self, event_info: Dict) -> str:
        """Format generic event details for unknown event types."""
        # Filter out internal/metadata fields
        excluded_keys = {'event_type', 'DT_r', 'DT_tz', 'time_to_last_entry',
                        'event_local_index', 'event_time'}
        details = []
        for key, value in event_info.items():
            if key not in excluded_keys and value is not None:
                if isinstance(value, float):
                    details.append(f"{key}: {value:.2f}")
                else:
                    details.append(f"{key}: {value}")
        return ", ".join(details[:10]) if details else ""  # Limit to 10 fields

    # =========================================================================
    # Prompt Generation Methods (Four Levels)
    # =========================================================================

    def make_prompt_noctx(self, source: Union[Any, Dict]) -> str:
        """
        Generate a prompt with NO context (time series only).

        This is the minimal level that includes:
        - Historical glucose time series only
        - Basic forecasting instruction
        - No patient information
        - No event information

        Parameters
        ----------
        source : task instance or dict
            Data source containing time series

        Returns
        -------
        str
            Generated prompt with time series only
        """
        data = self._extract_data(source)

        # Format history
        history, format_desc = self._format_history(data['hist_time'], data['hist_value'])

        # Minimal context section - no patient or event information
        context_section = "This is continuous glucose monitoring (CGM) data measuring blood glucose levels every 5 minutes."

        # Format prediction timestamps
        pred_time_str = ", ".join(data['pred_time'][:5])  # Show first 5
        if len(data['pred_time']) > 5:
            pred_time_str += f", ... ({len(data['pred_time'])} timestamps total)"

        # Generate prompt
        prompt = self.BASE_TEMPLATE.format(
            context_section=context_section,
            data_format=format_desc,
            history=history,
            pred_time=pred_time_str
        )

        return prompt

    def make_prompt_profile(self, source: Union[Any, Dict]) -> str:
        """
        Generate a prompt with patient profile information.

        This is the basic level that includes:
        - Historical glucose time series
        - Patient biographical information (age, gender, diabetes type)
        - Basic forecasting instruction

        Parameters
        ----------
        source : task instance or dict
            Data source containing time series and patient information

        Returns
        -------
        str
            Generated prompt with patient profile
        """
        data = self._extract_data(source)

        # Format history
        history, format_desc = self._format_history(data['hist_time'], data['hist_value'])

        # Build context section
        context_parts = []

        # Add patient information
        patient_info = self._extract_patient_info(data['patient_metadata'])
        context_parts.append(patient_info)

        # Add background if available
        if data['background']:
            context_parts.append(f"\nBackground: {data['background']}")

        context_section = "\n".join(context_parts)

        # Format prediction timestamps
        pred_time_str = ", ".join(data['pred_time'][:5])  # Show first 5
        if len(data['pred_time']) > 5:
            pred_time_str += f", ... ({len(data['pred_time'])} timestamps total)"

        # Generate prompt
        prompt = self.BASE_TEMPLATE.format(
            context_section=context_section,
            data_format=format_desc,
            history=history,
            pred_time=pred_time_str
        )

        return prompt

    def make_prompt_medium_event(self, source: Union[Any, Dict]) -> str:
        """
        Generate a prompt with medium-level event information.

        This level includes:
        - Historical glucose time series
        - Patient biographical information
        - Event timing (when events occur)
        - Basic event types

        Parameters
        ----------
        source : task instance or dict
            Data source containing time series and event information

        Returns
        -------
        str
            Generated prompt with event timing
        """
        data = self._extract_data(source)

        # Format history
        history, format_desc = self._format_history(data['hist_time'], data['hist_value'])

        # Build context section
        context_parts = []

        # Add patient information
        patient_info = self._extract_patient_info(data['patient_metadata'])
        context_parts.append(patient_info)

        # Add background if available
        if data['background']:
            context_parts.append(f"\nBackground: {data['background']}")

        # Add event timing information (medium level)
        event_timing = self._extract_event_timing(data['scenario'], data['event_details'])
        context_parts.append(f"\n{event_timing}")

        context_section = "\n".join(context_parts)

        # Format prediction timestamps
        pred_time_str = ", ".join(data['pred_time'][:5])  # Show first 5
        if len(data['pred_time']) > 5:
            pred_time_str += f", ... ({len(data['pred_time'])} timestamps total)"

        # Generate prompt
        prompt = self.BASE_TEMPLATE.format(
            context_section=context_section,
            data_format=format_desc,
            history=history,
            pred_time=pred_time_str
        )

        return prompt

    def make_prompt_detailed_event(self, source: Union[Any, Dict]) -> str:
        """
        Generate a prompt with detailed event information.

        This is the most comprehensive level that includes:
        - Historical glucose time series
        - Patient biographical information
        - Complete event details (timing, type, attributes)
        - Future intervention information
        - Scenario context

        Parameters
        ----------
        source : task instance or dict
            Data source containing complete task information

        Returns
        -------
        str
            Generated prompt with full event details
        """
        data = self._extract_data(source)

        # Format history
        history, format_desc = self._format_history(data['hist_time'], data['hist_value'])

        # Build context section
        context_parts = []

        # Add patient information
        patient_info = self._extract_patient_info(data['patient_metadata'])
        context_parts.append(patient_info)

        # Add background if available
        if data['background']:
            context_parts.append(f"\nBackground: {data['background']}")

        # Add detailed event information
        event_details_str = self._extract_event_details(data['scenario'], data['event_details'])
        #TODO: exlucde unneccesary information (like ids)
        #TODO: non urgent: change detail infor to categorical / summarized information
        context_parts.append(f"\n{event_details_str}")

        context_section = "\n".join(context_parts)

        # Format prediction timestamps
        pred_time_str = ", ".join(data['pred_time'][:5])  # Show first 5
        if len(data['pred_time']) > 5:
            pred_time_str += f", ... ({len(data['pred_time'])} timestamps total)"

        # Generate prompt
        prompt = self.BASE_TEMPLATE.format(
            context_section=context_section,
            data_format=format_desc,
            history=history,
            pred_time=pred_time_str
        )

        return prompt

    def make_prompt_newmedium(self, source: Union[Any, Dict]) -> str:
        """
        Generate a prompt with new medium-level event information.

        This level includes:
        - Historical glucose time series
        - Patient biographical information
        - Event timing + key metrics (calories and carbs for diet)

        Parameters
        ----------
        source : task instance or dict
            Data source containing time series and event information

        Returns
        -------
        str
            Generated prompt with event timing and key metrics
        """
        data = self._extract_data(source)

        # Format history
        history, format_desc = self._format_history(data['hist_time'], data['hist_value'])

        # Build context section
        context_parts = []

        # Add patient information
        patient_info = self._extract_patient_info(data['patient_metadata'])
        context_parts.append(patient_info)

        # Add background if available
        if data['background']:
            context_parts.append(f"\nBackground: {data['background']}")

        # Add new event timing information (with key metrics)
        event_timing = self._extract_event_timing_new(data['scenario'], data['event_details'])
        context_parts.append(f"\n{event_timing}")

        context_section = "\n".join(context_parts)

        # Format prediction timestamps
        pred_time_str = ", ".join(data['pred_time'][:5])  # Show first 5
        if len(data['pred_time']) > 5:
            pred_time_str += f", ... ({len(data['pred_time'])} timestamps total)"

        # Generate prompt
        prompt = self.BASE_TEMPLATE.format(
            context_section=context_section,
            data_format=format_desc,
            history=history,
            pred_time=pred_time_str
        )

        return prompt

    def make_prompt_newdetail(self, source: Union[Any, Dict]) -> str:
        """
        Generate a prompt with new detailed event information.

        This is the most comprehensive level that includes:
        - Historical glucose time series
        - Patient biographical information
        - Complete event details with rounded values (timing, type, nutrition)
        - Values rounded to 2 decimal places max

        Parameters
        ----------
        source : task instance or dict
            Data source containing complete task information

        Returns
        -------
        str
            Generated prompt with full event details (rounded values)
        """
        data = self._extract_data(source)

        # Format history
        history, format_desc = self._format_history(data['hist_time'], data['hist_value'])

        # Build context section
        context_parts = []

        # Add patient information
        patient_info = self._extract_patient_info(data['patient_metadata'])
        context_parts.append(patient_info)

        # Add background if available
        if data['background']:
            context_parts.append(f"\nBackground: {data['background']}")

        # Add new detailed event information (with rounded values)
        event_details_str = self._extract_event_details_new(data['scenario'], data['event_details'])
        context_parts.append(f"\n{event_details_str}")

        context_section = "\n".join(context_parts)

        # Format prediction timestamps
        pred_time_str = ", ".join(data['pred_time'][:5])  # Show first 5
        if len(data['pred_time']) > 5:
            pred_time_str += f", ... ({len(data['pred_time'])} timestamps total)"

        # Generate prompt
        prompt = self.BASE_TEMPLATE.format(
            context_section=context_section,
            data_format=format_desc,
            history=history,
            pred_time=pred_time_str
        )

        return prompt

    # =========================================================================
    # Factory Method
    # =========================================================================

    def get_prompt_fn(self, level: str) -> Callable:
        """
        Get the prompt generation function for the specified granularity level.

        This factory method returns the appropriate prompt generation function
        that can be used directly in task configurations as 'fn_prompt'.

        Parameters
        ----------
        level : str
            Granularity level, one of:
            - "noctx": Time series only (no context)
            - "profile": Patient profile information only
            - "medium_event": Profile + event timing (original)
            - "detailed_event": Profile + complete event details (original)
            - "newmedium": Profile + event timing + key metrics (calories/carbs)
            - "newdetail": Profile + full event details with rounded values

        Returns
        -------
        Callable
            Prompt generation function that takes a task instance/dict and returns a string

        Raises
        ------
        ValueError
            If level is not one of the valid options

        Examples
        --------
        >>> prompt_maker = MakePrompt()
        >>> fn_prompt = prompt_maker.get_prompt_fn(level="newdetail")
        >>> prompt = fn_prompt(task_instance)

        >>> # Use in task configuration
        >>> def get_task_config(self):
        ...     prompt_maker = MakePrompt(max_history_points=100)
        ...     return {
        ...         'fn_prompt': prompt_maker.get_prompt_fn(level="newmedium"),
        ...         # ... other config
        ...     }
        """
        level_map = {
            "noctx": self.make_prompt_noctx,
            "profile": self.make_prompt_profile,
            "medium_event": self.make_prompt_medium_event,
            "detailed_event": self.make_prompt_detailed_event,
            "newmedium": self.make_prompt_newmedium,
            "newdetail": self.make_prompt_newdetail,
        }

        if level not in level_map:
            valid_levels = list(level_map.keys())
            raise ValueError(
                f"Invalid level '{level}'. Must be one of {valid_levels}"
            )

        return level_map[level]

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def __call__(self, source: Union[Any, Dict], level: str = "detailed_event") -> str:
        """
        Make the class callable for convenient prompt generation.

        Parameters
        ----------
        source : task instance or dict
            Data source
        level : str, default="detailed_event"
            Granularity level

        Returns
        -------
        str
            Generated prompt

        Examples
        --------
        >>> prompt_maker = MakePrompt()
        >>> prompt = prompt_maker(task_instance, level="medium_event")
        """
        fn = self.get_prompt_fn(level)
        return fn(source)

    def available_levels(self) -> list[str]:
        """
        Get list of available prompt granularity levels.

        Returns
        -------
        list[str]
            Available level names
        """
        return ["noctx", "profile", "medium_event", "detailed_event", "newmedium", "newdetail"]

    def __repr__(self) -> str:
        return (
            f"MakePrompt(time_format='{self.time_format}', "
            f"max_history_points={self.max_history_points}, "
            f"max_digits={self.max_digits})"
        )


# =========================================================================
# Convenience Functions for Quick Usage
# =========================================================================

def create_prompt_fn(level: str = "detailed_event", **kwargs) -> Callable:
    """
    Create a prompt function with default settings.

    This is a convenience function for quickly creating prompt functions
    for use in task configurations.

    Parameters
    ----------
    level : str, default="detailed_event"
        Granularity level
    **kwargs
        Additional arguments passed to MakePrompt constructor

    Returns
    -------
    Callable
        Prompt generation function

    Examples
    --------
    >>> fn_prompt = create_prompt_fn(level="medium_event", max_history_points=50)
    >>>
    >>> def get_task_config(self):
    ...     return {
    ...         'fn_prompt': create_prompt_fn(level="profile"),
    ...         # ... other config
    ...     }
    """
    prompt_maker = MakePrompt(**kwargs)
    return prompt_maker.get_prompt_fn(level)
