# EventGlucose: Event-Aware Glucose Forecasting Benchmark

A comprehensive benchmark for continuous glucose monitoring (CGM) forecasting with contextual event information. EventGlucose combines state-of-the-art time series foundation models with intervention-aware sampling to advance personalized diabetes care.

## 🎯 Overview

EventGlucose (GlucoCIK - Glucose Context is Key) provides:
- **Event-aware sampling**: Predictions centered around intervention events (meals, medications, exercise)
- **Foundation model evaluation**: Integration with 18 SOTA forecasting models
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

The EventGlucose benchmark dataset is available on the Hugging Face Hub at [`CDHAI/EventGlucoseBench`](https://huggingface.co/datasets/CDHAI/EventGlucoseBench).

```bash
# Create symbolic links to your data and model directories
ln -s /path/to/your/data _Data
ln -s /path/to/your/models _Model

# Or use the default workspace structure
mkdir -p _WorkSpace/{Data,Model,Result}
```

### 4. Run Experiments
The published dataset contains anonymized subgroups that match the exact data used in the paper for evaluation. Use `--pre-sampled-dir` to run on it instead of the private `@task` data.
```bash
# List available tasks and models
python code/scripts/run_individual.py --list-tasks
python code/scripts/run_individual.py --list-models
python code/scripts/run_individual.py --list-model-types

# Single task
python code/scripts/run_individual.py \
    --task EventCGMTask_D1_Age18_Diet_Ontime_NoCtx \
    --model random \
    --n-instances 10 --n-samples 25 \
    --pre-sampled-dir _WorkSpace/Data/EventGlucose/publish-data

# All context levels
python code/scripts/run_individual.py \
    --task EventCGMTask_D1_Age18_Diet_Ontime_allcontext \
    --model gpt-4o-context \
    --n-instances 10 --n-samples 25 \
    --pre-sampled-dir _WorkSpace/Data/EventGlucose/publish-data

# Full benchmark sweep
python code/scripts/run_individual.py \
    --all-tasks \
    --model foundation-all \
    --n-instances 10 --n-samples 50 \
    --pre-sampled-dir _WorkSpace/Data/EventGlucose/publish-data
```

**Key properties of `--pre-sampled-dir`:**
- Rows are selected **deterministically** by seed: seed 1 → row 0, seed 2 → row 1, …
- `--n-instances` should not exceed the number of rows in each PKL (exactly 10 in `publish-data`)
- Results are saved locally to `_WorkSpace/Result/` — the public dataset is never modified
- Can be combined with `--skip-done` to resume interrupted runs

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

**Direct Prompt (LLM APIs):**
- Claude 4.5 (Haiku, Sonnet, Opus) — via Anthropic API or Claude Code SDK
- GPT-4o, GPT-4o-mini, GPT-5-mini — via OpenAI API
- Gemini-2.5-Flash, Qwen-3-235B, Llama-3, Mixtral — via OpenRouter

**LLM Processes (HuggingFace):**
- Llama-3 (8B, 70B, Instruct variants)
- Mixtral-8x7B (Instruct variants)
- Qwen-2.5 (0.5B, 7B Instruct)

**Multimodal Foundation Models:**
- UniTime (ETTh1 backbone)
- TimeLLM (ETTh1 backbone)

**Time Series Foundation Models:**
- Chronos (tiny/mini/small/base/large)
- Moirai (small/base/large)
- Lag-Llama

**Transformer Models (trained on-the-fly):**
- iTransformer, Autoformer, Causal Transformer
- All available with (`-ctx`) and without context covariates

**LTSF-Linear Models:**
- DLinear, NLinear

**Statistical Baselines:**
- ARIMA, ETS (via R forecast)
- Exponential Smoothing (via statsmodels)

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

<!-- ## 📄 License

Apache-2.0 License -->

## 📧 Contact

- **Issues:** https://github.com/EventGlucose/EventGlucoseBench/issues
- **Discussions:** https://github.com/EventGlucose/EventGlucoseBench/discussions



<!-- ## 📝 Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{eventglucose2026,
  title={EventGlucose: Context-Aware Glucose Forecasting with Event-Based Sampling},
  author={EventGlucose Team},
  journal={arXiv preprint},
  year={2026}
}
``` -->

---

**Note:** This is a research benchmark. Predictions should not be used for medical decision-making without proper clinical validation.
