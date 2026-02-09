import logging

# Set up logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



class CGMForecasterWrapper:
    """Wrapper to standardize model interfaces for CGM tasks."""
    
    def __init__(self, name: str, model, version: str = "1.0"):
        self.name = name
        self.model = model
        self.__version__ = version
    
    def __call__(self, task_instance, n_samples: int):
        """Call the wrapped model and handle different return formats."""
        if self.name == "random":
            return random_baseline(task_instance, n_samples)
        elif self.name == "oracle":
            return self._oracle_baseline(task_instance, n_samples)
        elif self.name == "lag_llama":
            # lag_llama is a function, not a class
            result = lag_llama(task_instance, n_samples)
            # lag_llama returns (samples, extra_info) tuple
            if isinstance(result, tuple):
                return result[0]  # Return just the samples
            return result
        elif self.name.startswith("timellm") or self.name.startswith("unitime"):
            # TimeLLM and UniTime models need special handling
            try:
                result = self.model(task_instance, n_samples)
                # Handle tuple returns
                if isinstance(result, tuple):
                    return result[0]
                return result
            except Exception as e:
                logger.error(f"Error calling {self.name}: {e}")
                raise
        elif hasattr(self.model, '__call__'):
            result = self.model(task_instance, n_samples)
            # Handle tuple returns from some models (e.g., ExponentialSmoothingForecaster)
            if isinstance(result, tuple):
                return result[0]  # Return just the samples
            return result
        else:
            raise ValueError(f"Unknown model type: {self.name}")
    
    def _oracle_baseline(self, task_instance, n_samples: int):
        """Oracle baseline implementation."""
        target = task_instance.future_time.to_numpy()
        
        # Ensure target has the right shape [time, 1] for univariate
        if target.ndim == 1:
            target = target.reshape(-1, 1)
        elif target.ndim == 2 and target.shape[1] != 1:
            target = target[:, 0].reshape(-1, 1)
        
        # Create samples: [n_samples, time, variables]
        samples = np.tile(target[None, :, :], (n_samples, 1, 1))
        
        # Add tiny jitter for each sample
        jitter = np.random.rand(n_samples, target.shape[0], target.shape[1]) * 1e-6
        
        return samples + jitter