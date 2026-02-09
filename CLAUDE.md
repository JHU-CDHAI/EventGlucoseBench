# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EventGlucose (also known as GlucoCIK - Glucose Context is Key) is a specialized time series forecasting benchmark focused on continuous glucose monitoring (CGM) data with contextual information. The project combines:

1. **EventGlucose Core Framework**: A glucose-specific forecasting benchmark with event-aware sampling
2. **Time Series Foundation Models**: Integration with state-of-the-art forecasting models
3. **Contextual Information**: Diet, medication, and exercise events as covariates
4. **Comprehensive Evaluation**: CRPS-based metrics with caching and parallel processing

The framework extends traditional time series forecasting by incorporating intervention events (meals, medication, exercise) and patient demographics to improve glucose prediction accuracy.

## Data Preparation

The project expects symbolic links to data and model directories:
```bash
# Create symbolic links (adjust paths as needed)
ln -s /path/to/your/data/_Data ~/EventGlucose/_Data
ln -s /path/to/your/models/_Model ~/EventGlucose/_Model
```

The CGM data should be stored in pickle format at `_Data/8-Data_LTS/` with required columns: `item_id`, `target`, `start`.

## Development Commands

### Environment Setup
```bash
# Create and activate environment
python -m venv .venv && source .venv/bin/activate

# Install core package with development dependencies
uv pip install -e ".[dev]"

# Install R and R packages for statistical baselines (optional)
sudo apt-get update && sudo apt-get install -y r-base r-base-dev libcurl4-openssl-dev
sudo R -e "install.packages(c('unix', 'forecast'), repos='https://cloud.r-project.org/')"
uv pip install rpy2

# Install time series foundation models (optional, install individually as needed)
uv pip install -r requirements-timeseries.txt
```

### Running Experiments
```bash
# After installation, use the CLI commands (recommended):
eglu-run --exp-spec experiments/glucose_comprehensive.json

# Run specific experiment types
eglu-run --exp-spec experiments/glucose_naive.json --n-samples 25

# Skip missing cache entries (useful for incomplete experiments)
eglu-run --exp-spec experiments/glucose_comprehensive.json --skip-cache-miss

# List available experiment methods and their parameters
eglu-run --list-exps

# Run individual task/method combinations
eglu-run-individual --task glucose_cgm_5min --method chronos --n-samples 50

# Alternative: Run scripts directly (if not installed)
python code/scripts/run_baselines.py --exp-spec experiments/glucose_comprehensive.json
```

### Data Synchronization
```bash
# Sync data with remote storage (Google Drive/S3)
# Requires: source env.sh first to load storage configuration

# Push local data to remote
eglu-remote-sync --push --path Data/1-SourceStore

# Pull remote data to local
eglu-remote-sync --pull --path Data/4-AIDataStore

# Dry run (preview changes without syncing)
eglu-remote-sync --push --path Data --dry-run

# Sync with dataset mode (specific datasets within stores)
eglu-remote-sync --push --source --name WellDoc2025CVS

# Legacy alias (same functionality)
hai-remote-sync --push --path Data
```

### Testing and Validation
```bash
# Test core package imports
python -c "import eventglucose, scripts; print('Core packages imported successfully!')"

# Test specific baseline models
python -c "from eventglucose.baselines.unitime import UniTimeForecaster; print('UniTime ready!')"
python -c "from eventglucose.baselines.chronos import ChronosForecaster; print('Chronos ready!')"
```

### Code Quality
```bash
# Format code with Black
black code/ --line-length 88

# Sort imports with isort
isort code/ --profile black

# Run type checking with mypy
mypy code/eventglucose/ --python-version 3.12
```

## Architecture Overview

### Core Framework Structure

**EventGlucose Package** (`code/eventglucose/`)
- `base.py`: Abstract task classes (`UnivariateCRPSTask`) with evaluation framework
- `config.py`: Global configuration, paths, API keys, and environment variables
- `evaluation.py`: Parallel evaluation system with caching and task management
- `tasks/`: Task implementations for different glucose forecasting scenarios
  - `glucose_cgm_task.py`: Main CGM tasks with event-aware windowing
- `baselines/`: Forecasting model implementations
- `metrics/`: CRPS computation and ROI-based evaluation metrics
- `utils/`: Caching system, plotting utilities, and helper functions

**Task System Architecture**
The framework uses a hierarchical task system:
- `UnivariateCRPSTask`: Base class with evaluation metrics and caching
- `GlucoseCGMTask`: Random window sampling with intervention covariates
- `GlucoseCGMTask_withEvent_withLag`: Event-focused sampling with lag adjustment

### Key Design Patterns

**Event-Aware Sampling**
- **Random Sampling**: `GlucoseCGMTask` uses random windows across patient timelines
- **Event-Focused Sampling**: `GlucoseCGMTask_withEvent_withLag` centers windows around intervention events
- **Lag Adjustment**: Control temporal relationship between events and observation points
- **Context Integration**: Combine historical glucose with intervention flags and calendar features

**Data Structure**
- **5-minute intervals**: CGM data at 288 timesteps per day (24*60/5)
- **Default windows**: 289 timesteps context (~24h), 24 timesteps prediction (~2h)
- **Covariates**: Calendar features (day-of-week, hour-of-day) + intervention flags (Diet5Min, Med5Min, Exercise5Min)
- **Patient metadata**: Demographics, disease type, timezone for contextual information

**Baseline Model Integration**
- **Foundation Models**: Chronos, Lag-Llama, Moirai, UniTime, TimeLLM (require separate Git installation)
- **LLM-based**: DirectPrompt (GPT/Claude), LLM Processes with text context
- **Statistical**: Statsmodels, R-based (ETS, ARIMA) for traditional baselines
- **Evaluation Pipeline**: Unified interface for probabilistic forecasting with CRPS metrics

**Caching and Evaluation**
- **HDF5 disk cache**: Expensive model predictions cached by task ID, method name, and parameters
- **Parallel processing**: Configurable via `max_parallel` parameter for batch evaluation
- **Result aggregation**: Automatic compilation of results across tasks, seeds, and methods
- **Cache invalidation**: Automatic based on model version strings and parameter changes

### Experiment Configuration System

**JSON-based Experiments** (`experiments/`)
The framework uses JSON files to specify experiment configurations:
- Each experiment has a `label`, `method`, and method-specific parameters
- Methods correspond to functions prefixed with `experiment_` in `run_baselines.py`
- Sequential execution with automatic result compilation and caching

**Available Experiment Methods**
- `experiment_naive`: Random and oracle baselines
- `experiment_chronos`: Chronos foundation models (small, base, large)
- `experiment_lag_llama`: Lag-Llama foundation model
- `experiment_moirai`: Moirai forecasting models
- `experiment_directprompt`: GPT/Claude direct prompting with context
- `experiment_timellm`: TimeLLM multimodal approach
- `experiment_unitime`: UniTime foundation model
- `experiment_statsmodels`: Traditional statistical methods
- `experiment_timegen1`: Nixtla TimeGEN-1 API

### Configuration and Environment

**Storage Paths** (configurable via environment variables)
- `CIK_DATA_STORE`: Dataset storage (default: `./_Data`)
- `CIK_MODEL_STORE`: Model weights storage (default: `./_Model`)
- `CIK_RESULT_CACHE`: Prediction cache (default: `./Data/_Inference_Cache`)

**API Configuration**
- **OpenAI**: `CIK_OPENAI_API_KEY`, `CIK_OPENAI_USE_AZURE` for GPT models
- **Nixtla**: `CIK_NIXTLA_BASE_URL`, `CIK_NIXTLA_API_KEY` for TimeGEN
- **Llama**: `CIK_LLAMA31_405B_URL`, `CIK_LLAMA31_405B_API_KEY` for large models

**⚠️ SECURITY WARNING**: The `code/eventglucose/config.py` file currently contains hardcoded API keys. These should be:
1. Removed from the codebase immediately
2. Rotated/invalidated at the provider
3. Set via environment variables only
4. Added to `.gitignore` if stored in local config files

### Foundation Model Dependencies

**Git-based Installation** (separate from main package)
Foundation models are installed individually due to their size and research nature:
```bash
# Time series foundation models
pip install git+https://github.com/ServiceNow/TACTiS.git@tactis-2
pip install git+https://github.com/time-series-foundation-models/lag-llama.git@main
pip install git+https://github.com/amazon-science/chronos-forecasting.git@main
pip install git+https://github.com/AndrewRWilliams/UniTimeBaseline.git
pip install git+https://github.com/AndrewRWilliams/TimeLLMBaseline.git
```

**Version Compatibility**
- PyTorch versions constrained to `>=2.0.0,<2.6.0` for foundation model compatibility
- Models may have specific dependency requirements that override base package versions

## Critical Implementation Details

1. **Task Inheritance**: Child classes should call `super().__init__()` first, then add specific attributes
2. **Random State Management**: Tasks use `self.random` for reproducible sampling across methods
3. **Covariate Stacking**: Calendar features (31 dims) + interventions (3 dims) = 34-dimensional covariates
4. **Event Parsing**: Sparse intervention dictionaries converted to dense 0/1 arrays for model consumption
5. **Result Format**: Models return `(n_samples, prediction_length, 1)` arrays for CRPS evaluation
6. **Context Integration**: Background (patient demographics) + scenario (event descriptions) for LLM methods

## Instance Data Model

The `code/instanceclass/eventglu.py` module defines Pydantic models for prediction instances:

- **`Event`**: Represents intervention events (diet, medication, exercise) with local/time indices and attributes
- **`DataPointInstance`**: Core dataclass connecting instances to models with:
  - Historical data: `target_history`, `event_history`
  - Future data: `target_future`, `event_future` (ground truth)
  - Metadata: `static_information`, `prediction_time`, `prediction_time_step`
  - Predictions: `predicted_target_future`, `predicted_target_future_confidence`
- **`ModelInput`/`ModelOutput`**: Preprocessed formats for model consumption

This abstraction layer enables conversion between text formats (for LLM APIs) and tensor formats (for PyTorch models).

## Package Structure

- **`code/eventglucose/`**: Core benchmark framework
- **`code/instanceclass/`**: Pydantic-based data models for prediction instances
- **`code/scripts/`**: Experiment runners and utility scripts
- **`experiments/`**: JSON experiment specifications and documentation
- **`notebooks/`**: Analysis and demonstration notebooks
- **`requirements-timeseries.txt`**: Foundation model dependencies
- **`_Data/`**: Dataset storage (symbolic link)
- **`_Model/`**: Model weights storage (symbolic link)
- **`_Data/_Inference_Cache/`**: Prediction cache (created automatically)