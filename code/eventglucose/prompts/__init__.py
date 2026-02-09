# Each python covers a version of prompt making function
"""
1. How to express time series data for CGM
i) (T,V) for all possible time value pair -> issue: redundant time information
ii) sequence of V, (start and end time, 5 min interval as context,)

2. How to express event (Scenario) data
i) Level 1: Event Type (Diet, Exercise, Medication)
ii) Level 2: Event Granularity Level(Diet, Exercise, Medication)
iii) Level 3: Event Details (the whole dictionary)

3. Background information
i) With / Without Patient Biographical information
ii) How detail are the biographical information

4. How to structure the prompt
i). Plain Text

"""

# Import the new MakePrompt class (v2)
from .make_prompt import MakePrompt, create_prompt_fn

# Exports
__all__ = [
    'MakePrompt',
    'create_prompt_fn',
]

prompt_config = {}

"""
    def make_prompt(self, max_digits=6, use_context=True, use_patient_info=False):
        
        Generate the prompt for the model

        Notes:
        - Assumes a uni-variate time series

        
        task_instance = self
        # logger.info("Building prompt for model.")

        # Extract time series data
        hist_time = task_instance.past_time.index.strftime("%Y-%m-%d %H:%M:%S").values
        hist_value = task_instance.past_time.values[:, -1]
        pred_time = task_instance.future_time.index.strftime("%Y-%m-%d %H:%M:%S").values
        # "g" Print up to max_digits digits, although it switch to scientific notation when y >= 1e6,
        # so switch to "f" without any digits after the dot if y is too large.
        history = "\n".join(
            f"({x}, {y:.{max_digits}g})" if y < 10**max_digits else f"({x}, {y:.0f})"
            for x, y in zip(hist_time, hist_value)
        )

        # Extract context
        context = ""
        if use_context:
            if task_instance.background:
                context += f"Background: {task_instance.background}\n"
            if task_instance.constraints:
                context += f"Constraints: {task_instance.constraints}\n"
            if task_instance.scenario:
                context += f"Scenario: {task_instance.scenario}\n"

        if use_patient_info:
            ... # context += f"Patient Information: {task_instance.patient_metadata}\n"

        prompt = BASE_PROMPT.format(
            context=context, 
            history=history, 
            pred_time=pred_time
        )
        return prompt
"""