# EventGlucose: Event-Aware Glucose Forecasting Benchmark

A comprehensive benchmark for continuous glucose monitoring (CGM) forecasting with contextual event information. EventGlucose combines state-of-the-art time series foundation models with intervention-aware sampling to advance personalized diabetes care.

## 🎯 Overview

EventGlucose (GlucoCIK - Glucose Context is Key) provides:
- **Event-aware sampling**: Predictions centered around intervention events (meals, medications, exercise)
- **Foundation model evaluation**: Integration with 15+ SOTA forecasting models
- **Contextual information**: Patient demographics, intervention timing, and calendar features
- **Probabilistic forecasting**: CRPS-based evaluation with uncertainty quantification

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/EventGlucose/EventGlucoseBench.git
cd EventGlucoseBench

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core package
pip install -e ".[dev]"

# Install time series foundation models (optional, install individually)
pip install -r requirements-timeseries.txt
```

### 2. Configure Credentials

```bash
# Copy environment template
cp env.sh.template env.sh

# Edit and add your API keys (see SETUP_CREDENTIALS.md for details)
nano env.sh

# Source the configuration
source env.sh
```

See [SETUP_CREDENTIALS.md](SETUP_CREDENTIALS.md) for detailed credential setup instructions.

### 3. Prepare Data

```bash
# Create symbolic links to your data and model directories
ln -s /path/to/your/data _Data
ln -s /path/to/your/models _Model

# Or use the default workspace structure
mkdir -p _WorkSpace/{Data,Model,Result}
```

### 4. Run Experiments

```bash
# Run a simple baseline experiment
eglu-run --exp-spec experiments/statistical-models/statsmodels_c40.json

# Run foundation model baselines
eglu-run --exp-spec experiments/foundation-models/chronos_small_g1.json

# List available experiment methods
eglu-run --list-exps
```

## 🏗️ Architecture

### Core Components

```
code/
├── eventglucose/          # Main benchmark framework
│   ├── base.py           # Task base classes
│   ├── config.py         # Configuration and environment
│   ├── evaluation.py     # Parallel evaluation system
│   ├── tasks/            # Task implementations
│   ├── baselines/        # Model implementations
│   ├── metrics/          # CRPS and evaluation metrics
│   └── utils/            # Utilities and caching
├── instanceclass/        # Pydantic data models
└── scripts/              # CLI entry points
```

### Task Types

1. **GlucoseCGMTask**: Random window sampling across patient timelines
2. **GlucoseCGMTask_withEvent_withLag**: Event-centered sampling with lag control

### Supported Models

**Foundation Models:**
- Chronos (Tiny, Small, Base, Large)
- Lag-Llama
- Moirai (Small, Base, Large)
- UniTime
- TimeLLM

**LLM-based:**
- DirectPrompt (GPT-4, Claude, Gemini)
- LLM Processes (Llama, Qwen, Mixtral)

**Statistical:**
- Statsmodels (ARIMA, ETS)
- R Forecast
- Nixtla TimeGEN

## 📊 Experiments

Pre-configured experiment specifications in `experiments/`:

```bash
experiments/
├── foundation-models/     # Chronos, Lag-Llama, Moirai
├── llmp-models/          # LLM Processes with various models
├── multimodal-models/    # UniTime, TimeLLM
├── direct-prompt-models/ # GPT/Claude direct prompting
└── statistical-models/   # Classical forecasting methods
```

## 🔧 CLI Tools

### Run Experiments
```bash
# Run baseline experiments from JSON specs
eglu-run --exp-spec experiments/glucose_comprehensive.json

# Run specific task/method combination
eglu-run-individual --task glucose_cgm_5min --method chronos --n-samples 50

# Skip missing cache entries
eglu-run --exp-spec experiments/glucose_comprehensive.json --skip-cache-miss
```



## 📈 Evaluation

The framework uses:
- **CRPS (Continuous Ranked Probability Score)**: Primary probabilistic metric
- **Caching system**: HDF5-based prediction cache to avoid recomputation
- **Parallel processing**: Configurable workers for batch evaluation
- **Result compilation**: Automatic aggregation across tasks and seeds

## 📦 Dependencies

### Core Requirements
- Python ≥ 3.12
- PyTorch ≥ 2.0.0, < 2.6.0
- Transformers, GluonTS, Statsmodels
- NumPy < 2.0 (for PyTorch compatibility)

### Optional
- R and R packages (for statistical baselines)
- rclone (for data synchronization)
- CUDA (for GPU acceleration)

## 📄 License

Apache-2.0 License

## 📧 Contact

- **Issues:** https://github.com/EventGlucose/EventGlucoseBench/issues
- **Discussions:** https://github.com/EventGlucose/EventGlucoseBench/discussions



## 📝 Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{eventglucose2026,
  title={EventGlucose: Context-Aware Glucose Forecasting with Event-Based Sampling},
  author={EventGlucose Team},
  journal={arXiv preprint},
  year={2026}
}
```

---

**Note:** This is a research benchmark. Predictions should not be used for medical decision-making without proper clinical validation.
