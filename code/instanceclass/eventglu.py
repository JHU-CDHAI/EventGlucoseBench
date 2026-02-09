f'''

For this one, I am thinking about an instance.

An instance means now I want to do a prediction invokation for this instance. 

So what is the instance looks like? 

I think now you need to make the dataclass. 

Or we call it the prediction instance class. 

instance_dataclass will then go to a model. 
For this model, you will have its own preprocessfn (instance_dataclass -> model_input_dataclass)
And then model_input_dataclass will go to a model. you will have model_output_dataclass. 
Then you will have a postprocessfn (model_output_dataclass -> instance_dataclass)

THe instance_dataclas therefore make the connection between the instance and the model. 


So now, let take a look at the instance_dataclass. 

Most essential.

target_history
target_future
event_history: we will also have an event dataclass, 
    - event_local_index
    - event_time_index
    - event_info: this should be a dictionary of k:v. 

event_future: 
    - event_local_index
    - event_time_index
    - event_info: this should be a dictionary of k:v. 

static_information: a dictionary of k:v. 
prediction_time: the datetime. 
prediction_time_step: the length: 5 minutes, etc. 


predicted_target_future: you can have a list of predicted target future. 
predicted_target_future_confidence: you can have a list of predicted target future confidence. 
'''



# afte this, you can either make them to be TextFormat (To LLM-API) or TensorFormat (To Torch-Model).


from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd


class Event(BaseModel):
    """Represents an intervention event (diet, medication, exercise)"""
    event_local_index: int = Field(..., description="Position in the sequence")
    event_time_index: int = Field(..., description="Timestamp index relative to prediction time")
    event_info: Dict[str, Any] = Field(default_factory=dict, description="Event attributes (type, magnitude, duration, etc.)")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('event_info')
    @classmethod
    def validate_event_info(cls, v):
        if not isinstance(v, dict):
            return {}
        return v


class DataPointInstance(BaseModel):
    """Core dataclass for glucose prediction with event context"""
    # Historical data
    target_history: np.ndarray = Field(..., description="Historical glucose values")
    event_history: List[Event] = Field(default_factory=list, description="Past intervention events")

    # Future data (ground truth for training/evaluation)
    target_future: Optional[np.ndarray] = Field(None, description="Future glucose values")
    event_future: List[Event] = Field(default_factory=list, description="Upcoming events")

    # Metadata
    static_information: Dict[str, Any] = Field(default_factory=dict, description="Patient-specific attributes")
    prediction_time: Optional[datetime] = Field(None, description="Datetime of prediction")
    prediction_time_step: timedelta = Field(default=timedelta(minutes=5), description="Interval length")

    # Predictions (populated after model inference)
    predicted_target_future: Optional[np.ndarray] = Field(None, description="Model predictions")
    predicted_target_future_confidence: Optional[np.ndarray] = Field(None, description="Uncertainty estimates")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('target_history', 'target_future', 'predicted_target_future', 'predicted_target_future_confidence')
    @classmethod
    def validate_numpy_array(cls, v):
        if v is not None and not isinstance(v, np.ndarray):
            return np.asarray(v)
        return v

    @field_validator('static_information')
    @classmethod
    def validate_static_info(cls, v):
        if not isinstance(v, dict):
            return {}
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'target_history': self.target_history.tolist() if self.target_history is not None else None,
            'target_future': self.target_future.tolist() if self.target_future is not None else None,
            'event_history': [e.model_dump() for e in self.event_history],
            'event_future': [e.model_dump() for e in self.event_future],
            'static_information': self.static_information,
            'prediction_time': self.prediction_time.isoformat() if self.prediction_time else None,
            'prediction_time_step_minutes': self.prediction_time_step.total_seconds() / 60,
            'predicted_target_future': self.predicted_target_future.tolist() if self.predicted_target_future is not None else None,
            'predicted_target_future_confidence': self.predicted_target_future_confidence.tolist() if self.predicted_target_future_confidence is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataPointInstance':
        """Create instance from dictionary"""
        import pandas as pd

        return cls(
            target_history=np.array(data['target_history']) if data.get('target_history') else np.array([]),
            target_future=np.array(data['target_future']) if data.get('target_future') else None,
            event_history=[Event(**e) for e in data.get('event_history', [])],
            event_future=[Event(**e) for e in data.get('event_future', [])],
            static_information=data.get('static_information', {}),
            prediction_time=pd.to_datetime(data['prediction_time']) if data.get('prediction_time') else None,
            prediction_time_step=timedelta(minutes=data.get('prediction_time_step_minutes', 5)),
            predicted_target_future=np.array(data['predicted_target_future']) if data.get('predicted_target_future') else None,
            predicted_target_future_confidence=np.array(data['predicted_target_future_confidence']) if data.get('predicted_target_future_confidence') else None,
        )


class ModelInput(BaseModel):
    """Preprocessed format ready for model consumption"""
    features: np.ndarray = Field(..., description="Model input features")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModelOutput(BaseModel):
    """Raw model predictions"""
    predictions: np.ndarray = Field(..., description="Model predictions")
    confidence: Optional[np.ndarray] = Field(None, description="Prediction confidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Output metadata")

    model_config = ConfigDict(arbitrary_types_allowed=True)
