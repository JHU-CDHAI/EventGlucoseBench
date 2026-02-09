import pandas as pd
import numpy as np
from .base import Baseline
from ..base import BaseTask
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import set_seed
import os
import sys
from typing import Optional, Union, Dict, Callable, Iterable

import warnings
import logging
import json
import sys

from huggingface_hub import hf_hub_download

script_dir = os.path.dirname(os.path.abspath(__file__))

time_llm_models_path = os.path.join(script_dir, "Time-LLM", "models")

sys.path.append(time_llm_models_path)

from timellm.models.TimeLLM import Model as TimeLLMModel

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs available")

    if num_gpus >= 2:
        primary_device = torch.device("cuda:1")
        available_gpus = list(range(num_gpus))
        print(f"Multi-GPU setup: Primary device cuda:1, Available GPUs: {available_gpus}")
    else:
        primary_device = torch.device("cuda:0")
        available_gpus = [0]
        print("Single GPU setup: Using cuda:0")
else:
    primary_device = torch.device("cpu")
    available_gpus = []
    print("No GPU available, using CPU")

torch_device = primary_device

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("Cleared GPU cache on all available devices")

def truncate_mse_loss(future_time, future_pred):
    min_length = min(future_time.shape[-1], future_pred.shape[-1])
    return F.mse_loss(future_time[..., :min_length], future_pred[..., :min_length])

def truncate_mae_loss(future_time, future_pred):
    min_length = min(future_time.shape[-1], future_pred.shape[-1])
    return F.l1_loss(future_time[..., :min_length], future_pred[..., :min_length])

class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def find_pred_len_from_path(path: str) -> int:
    if "pl_96" or "pl96" in path:
        pred_len = 96
    elif "pl_192" or "pl192" in path:
        pred_len = 192
    elif "pl_336" or "pl336" in path:
        pred_len = 336
    elif "pl720" or "pl720" in path:
        pred_lent = 720
    else:
        raise ValueError(
            f"Could not determine prediction length of model from path {path}. Expected path to contain a substring of the form 'pl_{{pred_len}}' or 'pl{{pred_len}}'."
        )

    return pred_len

def find_model_name_from_path(path: str) -> str:
    path = path.lower()
    if "time-llm" in path or "timellm" in path:
        model_name = "time-llm"
    elif "unitime" in path:
        model_name = "unitime"
    else:
        raise ValueError(
            f"Could not determine model name from path {path}. Expected path to contain either 'time-llm', 'timellm', or 'unitime'."
        )

    return model_name

TIME_LLM_CONFIGS = DotDict(
    {
        "task_name": "long_term_forecast",
        "seq_len": 512,
        "enc_in": 7,
        "d_model": 32,
        "d_ff": 128,
        "llm_layers": 32,
        "llm_dim": 4096,
        "patch_len": 16,
        "stride": 8,
        "llm_model": "LLAMA",
        "llm_layers": 32,
        "prompt_domain": 1,
        "content": None,
        "dropout": 0.1,
        "d_model": 32,
        "n_heads": 8,
        "enc_in": 7,
    }
)

class TimeLLMWrapper(nn.Module):

    def __init__(self, time_llm_model):
        super().__init__()

        assert isinstance(
            time_llm_model, TimeLLMModel
        ), f"TimeLLMWrapper can only wrap a model of class TimeLLM.Model but got {type(time_llm_model)}"
        self.base_model = time_llm_model

    def forward(self, past_time, context):
        self.base_model.description = context
        return self.base_model(
            x_enc=past_time.unsqueeze(-1), x_mark_enc=None, x_dec=None, x_mark_dec=None
        ).squeeze(-1)

class WrappedBaseline(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.base_model = model
        if isinstance(self.base_model, TimeLLMModel):
            self.wrapped_model = TimeLLMWrapper(self.base_model)
        else:
            raise ValueError(
                f"WrappedBaseline can only wrap a model of class TimeLLM.Model but got {type(model)}"
            )

    def forward(self, past_time, context):
        return self.wrapped_model(past_time, context)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.base_model.load_state_dict(state_dict, strict, assign)

class EvaluationPipeline:

    def __init__(
        self,
        model: TimeLLMModel,
        metrics: Optional[Union[Callable, Dict[str, Callable]]] = None,
        device: Optional[torch.device] = None,
    ):
        self.metrics = (
            metrics if metrics is not None else {"mse_loss": truncate_mse_loss}
        )

        if device is not None:
            self.device = device
            print(f"EvaluationPipeline using provided device: {device}")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                warnings.warn(
                    "Warning: No CUDA device detected, proceeding with EvaluationPipeline on CPU ....."
                )

        try:
            self.model = WrappedBaseline(model).to(self.device)
            print(f"✓ WrappedBaseline successfully moved to {self.device}")
        except torch.cuda.OutOfMemoryError as e:
            print(f"✗ OOM moving WrappedBaseline to {self.device}, falling back to CPU")
            self.device = "cpu"
            self.model = WrappedBaseline(model).to(self.device)
            print(f"✓ WrappedBaseline moved to CPU")

    def get_evaluation_loader(self) -> Iterable:
        samples = []
        for sample in self.dataset.values():
            past_time = (
                torch.from_numpy(sample["past_time"].to_numpy().T)
                .float()
                .to(self.device)
            )
            future_time = (
                torch.from_numpy(sample["future_time"].to_numpy().T)
                .float()
                .to(self.device)
            )
            context = sample["context"]

            samples.append([past_time, future_time, context])

        return samples

    def compute_loss(self, future_time, future_pred):
        return {
            m_name: m(future_time, future_pred) for m_name, m in self.metrics.items()
        }

    def evaluation_step(self, past_time, future_time, context):
        with torch.no_grad():
            future_pred = self.model(past_time, context)
            loss = self.compute_loss(future_time, future_pred)
        return loss, future_pred

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        infer_dataloader = self.get_evaluation_loader()
        losses, predictions = {m_name: [] for m_name in self.metrics.keys()}, []
        for past_time, future_time, context in infer_dataloader:
            loss_dict, preds = self.evaluation_step(past_time, future_time, context)

            for m_name, loss in loss_dict.items():
                losses[m_name].append(loss)
            predictions.append(preds)

        self.model.train()
        return losses, predictions

class TimeLLMForecaster(Baseline):

    __version__ = "0.0.8"
    def __init__(
        self,
        use_context,
        dataset="ETTh2",
        pred_len=96,
        seed: int = 42,
        hf_repo="transfer-starcaster/time-llm-starcaster",
    ):

        self.use_context = use_context
        self.dataset = dataset
        self.pred_len = pred_len
        self.seed = seed

        set_seed(self.seed)
        ckpt_filename = f"TimeLLM-{dataset}-pl_{pred_len}-ckpt.pth"

        script_dir = os.path.dirname(os.path.abspath(__file__))

        ckpt_dir = os.path.join(script_dir, "Time-LLM", "checkpoints")

        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = os.path.join(ckpt_dir, ckpt_filename)

        if not os.path.exists(ckpt_path):
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename=ckpt_filename)

        args = DotDict(dict())

        args.pred_len = 96
        args.model_name = "time-llm"
        args.seed = seed
        self.model_name = args.model_name

        if args.model_name == "time-llm":
            args.update(TIME_LLM_CONFIGS)

        print(f"Initializing model from config:\n{args} .....")

        if args.model_name == "time-llm":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current_device = torch_device

            model_loaded = False

            if len(available_gpus) == 1:
                print(f"🚀 SINGLE GPU: Attempting to load on GPU {available_gpus[0]}")
                torch.cuda.empty_cache()

                try:
                    self.model = TimeLLMModel(args)
                    self.model = self.model.to(f"cuda:{available_gpus[0]}")
                    current_device = torch.device(f"cuda:{available_gpus[0]}")
                    model_loaded = True
                    print(f"✅ SUCCESS: Model loaded on single GPU: cuda:{available_gpus[0]}")
                except Exception as e:
                    print(f"❌ Single GPU failed: {e}")

            elif len(available_gpus) >= 2:
                print(f"🚀 MULTI-GPU: Attempting DataParallel across GPUs: {available_gpus}")
                torch.cuda.empty_cache()

                try:
                    print("Trying regular DataParallel")
                    self.model = TimeLLMModel(args)

                    try:
                        self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)
                        self.model = self.model.to(f"cuda:{available_gpus[0]}")
                    except NotImplementedError as meta_e:
                        if "Cannot copy out of meta tensor; no data!" in str(meta_e):
                            print("Meta tensors detected, using to_empty()")
                            self.model = torch.nn.DataParallel(self.model, device_ids=available_gpus)
                            self.model = self.model.to_empty(device=f"cuda:{available_gpus[0]}")
                        else:
                            raise meta_e

                    current_device = torch.device(f"cuda:{available_gpus[0]}")
                    model_loaded = True
                    print(f"✅ SUCCESS: DataParallel loaded on GPUs: {available_gpus}")
                except Exception as e:
                    print(f"❌ DataParallel failed: {e}")

            if not model_loaded:
                print("⚠️ GPU loading failed, attempting CPU fallback...")
                try:
                    self.model = TimeLLMModel(args)
                    self.model = self.model.to("cpu")
                    current_device = torch.device("cpu")
                    model_loaded = True
                    print("✅ SUCCESS: Model loaded on CPU (slower performance)")
                except Exception as e:
                    raise RuntimeError(f"All loading strategies failed. Last error: {e}")

            self.device = current_device
            self.uses_multi_gpu = isinstance(self.model, torch.nn.DataParallel)
            if self.uses_multi_gpu:
                print(f"Model using DataParallel across {len(available_gpus)} GPUs")
            else:
                print(f"Model using single device: {self.device}")
            self.backbone = TIME_LLM_CONFIGS.llm_model

        if ckpt_path is not None:
            try:
                if self.device.type == 'cpu':
                    ckpt = torch.load(ckpt_path, map_location='cpu')
                    print(f"Loaded checkpoint to CPU")
                else:
                    ckpt = torch.load(ckpt_path, map_location=self.device)
                    print(f"Loaded checkpoint to {self.device}")

                self.model.load_state_dict(ckpt["module"])
                print("✓ Checkpoint loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load checkpoint: {e}")
                print("Continuing without checkpoint")

        super().__init__()

    def __call__(
        self,
        task_instance: BaseTask,
        n_samples: Optional[int] = 1,
    ):
        set_seed(self.seed)
        self.model.pred_len = task_instance.future_time.shape[0]
        pipeline = EvaluationPipeline(
            self.model,
            metrics={"mse_loss": truncate_mse_loss, "mae_loss": truncate_mae_loss},
            device=self.device,
        )

        if self.use_context:
            context = self._make_prompt(task_instance)
        else:
            context = ""

        raw_data = task_instance.past_time[[task_instance.past_time.columns[-1]]].to_numpy().transpose()
        original_seq_len = raw_data.shape[-1]

        target_seq_len = 512
        if original_seq_len < target_seq_len:
            pad_size = target_seq_len - original_seq_len
            padding = np.repeat(raw_data[:, :1], pad_size, axis=1)
            raw_data = np.concatenate([padding, raw_data], axis=1)
            print(f"[TimeLLM] Padded input from {original_seq_len} to {raw_data.shape[-1]} for cuFFT compatibility")

        past_time = (
            torch.tensor(raw_data, dtype=torch.float32)
            .expand(n_samples, -1)
            .to(self.device)
        )

        print(f"[TimeLLM] past_time shape: {past_time.shape}, device: {past_time.device}")

        future_time = (
            torch.tensor(
                task_instance.future_time[[task_instance.future_time.columns[-1]]]
                .to_numpy()
                .transpose(),
                dtype=torch.float32,
            )
            .expand(n_samples, -1)
            .to(self.device)
        )

        batch_size = n_samples
        subgroup_size = 1
        predictions = []

        for i in range(0, batch_size, subgroup_size):
            past_time_subgroup = past_time[i : i + subgroup_size]
            future_time_subgroup = future_time[i : i + subgroup_size]
            context_subgroup = context

            if past_time_subgroup.shape[0] == 0:
                continue

            _, preds_subgroup = pipeline.evaluation_step(
                past_time_subgroup,
                future_time_subgroup,
                context_subgroup,
            )
            predictions.append(preds_subgroup)

            torch.cuda.empty_cache()

        prediction_tensor = torch.cat(predictions, dim=0)
        if prediction_tensor.shape[-1] < future_time.shape[-1]:
            last_value = prediction_tensor[:, -1].unsqueeze(-1)
            repeat_count = future_time.shape[-1] - prediction_tensor.shape[-1]
            prediction_tensor = torch.cat(
                [prediction_tensor, last_value.repeat(1, repeat_count)], dim=-1
            )

        prediction_tensor = prediction_tensor.unsqueeze(-1)

        return prediction_tensor.cpu().numpy()

    def _make_prompt(self, task_instance):
        Formats the prompt and adds it to the LLMP arguments

        Forecast the future values of this time series, while considering the following
        background knowledge, scenario, and constraints.

        Background knowledge:
        {task_instance.background}

        Scenario:
        {task_instance.scenario}

        Constraints:
        {task_instance.constraints}
