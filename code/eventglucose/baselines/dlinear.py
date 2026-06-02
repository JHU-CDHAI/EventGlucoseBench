"""
DLinear and NLinear baselines (LTSF-Linear family).

Both models are purely linear layers — no pretrained weights, no downloads.
They are fit on-the-fly using the context window of each task instance.

References:
    Zeng et al. (2023) "Are Transformers Effective for Time Series Forecasting?"
    https://arxiv.org/abs/2205.13504
"""

import time
import numpy as np
import torch
import torch.nn as nn

from .base import Baseline
from ..base import BaseTask


# ---------------------------------------------------------------------------
# Internal model definitions
# ---------------------------------------------------------------------------

class _MovingAvg(nn.Module):
    """Trend extraction via centered average pooling."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end   = x[:, -1:, :].repeat(1, pad, 1)
        x_padded = torch.cat([front, x, end], dim=1)         # (B, T+pad*2, 1)
        trend = self.avg(x_padded.permute(0, 2, 1))           # (B, 1, T)
        return trend.permute(0, 2, 1)                          # (B, T, 1)


class _DLinear(nn.Module):
    """Decomposition-Linear: separate linear layers for trend and seasonality."""

    def __init__(self, seq_len: int, pred_len: int, kernel_size: int = 25):
        super().__init__()
        self.decomp  = _MovingAvg(kernel_size)
        self.linear_trend  = nn.Linear(seq_len, pred_len)
        self.linear_season = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        trend  = self.decomp(x)
        season = x - trend
        out = (self.linear_trend(trend.squeeze(-1)) +
               self.linear_season(season.squeeze(-1)))         # (B, pred_len)
        return out.unsqueeze(-1)                               # (B, pred_len, 1)


class _NLinear(nn.Module):
    """Normalized-Linear: subtract last value then apply a single linear layer."""

    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 1)
        last = x[:, -1:, :]                                    # (B, 1, 1)
        normed = (x - last).squeeze(-1)                        # (B, T)
        out = self.linear(normed).unsqueeze(-1)                # (B, pred_len, 1)
        return out + last                                       # add back last value


# ---------------------------------------------------------------------------
# Forecaster wrapper (fits on context window, then predicts)
# ---------------------------------------------------------------------------

class DLinearForecaster(Baseline):
    """
    Zero-shot DLinear / NLinear forecaster.

    Strategy: fit the chosen linear model on the context window using a
    rolling self-supervised objective (predict the last `pred_len` steps
    from the preceding `seq_len - pred_len` steps), then make a single
    forward pass to obtain a point forecast. Uncertainty is captured by
    adding Gaussian noise scaled to the residual std of the history.

    Parameters
    ----------
    model_type : "dlinear" | "nlinear"
    n_epochs   : gradient steps for the on-context fit
    lr         : Adam learning rate
    kernel_size: moving-average kernel for DLinear trend extraction
    seed       : random seed for reproducibility
    """

    __version__ = "0.1.1"  # bump to invalidate cache

    def __init__(
        self,
        model_type: str = "dlinear",
        n_epochs: int = 100,
        lr: float = 1e-3,
        kernel_size: int = 25,
        seed: int = 42,
    ):
        assert model_type in ("dlinear", "nlinear"), \
            f"model_type must be 'dlinear' or 'nlinear', got {model_type!r}"
        self.model_type  = model_type
        self.n_epochs    = n_epochs
        self.lr          = lr
        self.kernel_size = kernel_size
        self.seed        = seed
        super().__init__()

    # ------------------------------------------------------------------
    def __call__(self, task_instance: BaseTask, n_samples: int) -> tuple:
        t0 = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        hist_col = task_instance.past_time.columns[-1]
        hist = (
            task_instance.past_time[hist_col]
            .values.astype(np.float32)
        )
        pred_len = len(task_instance.future_time)
        seq_len  = len(hist)
        device   = "cuda" if torch.cuda.is_available() else "cpu"

        # Split history: use first (seq_len - pred_len) steps as input,
        # last pred_len steps as self-supervised training target.
        # At inference the same x_in is used to produce the actual forecast.
        fit_len = max(seq_len - pred_len, 1)
        x_full  = torch.tensor(hist, dtype=torch.float32, device=device).view(1, seq_len, 1)
        x_in    = x_full[:, :fit_len, :]   # (1, fit_len, 1)
        y_in    = x_full[:, fit_len:, :]   # (1, pred_len, 1)

        # ---- Build + fit single model ------------------------------------
        model     = self._make_model(fit_len, pred_len).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        t_fit_start = time.time()
        model.train()
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            loss = criterion(model(x_in), y_in)
            loss.backward()
            optimizer.step()
        fit_time = time.time() - t_fit_start

        # ---- Inference ---------------------------------------------------
        t_inf = time.time()
        model.eval()
        with torch.no_grad():
            point_pred = model(x_in).squeeze(0).cpu().numpy()  # (pred_len, 1)
        inference_time = time.time() - t_inf

        # ---- Sample around point estimate --------------------------------
        noise_std = float(np.std(np.diff(hist))) if len(hist) > 1 else 1.0
        samples = (
            point_pred[None]  # (1, pred_len, 1)
            + np.random.randn(n_samples, pred_len, 1) * noise_std * 0.1
        )  # (n_samples, pred_len, 1)

        extra_info = {
            "total_time": time.time() - t0,
            "fit_time": fit_time,
            "inference_time": inference_time,
        }
        return samples, extra_info

    # ------------------------------------------------------------------
    def _make_model(self, seq_len: int, pred_len: int) -> nn.Module:
        if self.model_type == "dlinear":
            ks = min(self.kernel_size, seq_len - 1) if seq_len > 1 else 1
            # kernel must be odd and ≥ 1
            ks = max(ks | 1, 1)
            return _DLinear(seq_len, pred_len, kernel_size=ks)
        else:
            return _NLinear(seq_len, pred_len)

    @property
    def cache_name(self) -> str:
        return (
            f"{self.__class__.__name__}_{self.model_type}"
            f"_ep{self.n_epochs}_lr{self.lr}"
        )
