"""
Transformer-based baselines: iTransformer, Autoformer, CausalTransformer.

All three are trained from scratch on the context window of each task instance
(no pre-trained weights), following the same interface as DLinearForecaster.

Context-awareness (use_context=True):
    When the task instance has a `c_cov` attribute (event tasks), the past
    covariates (calendar features + intervention flags) are stacked with glucose
    as additional input variates.  When use_context=False (or the task has no
    c_cov), only the glucose series is used.

Models
------
iTransformer  – Inverted attention over the variate dimension; each variate
    (glucose + covariates) is its own token.  Attention captures cross-variate
    dependencies.  Ref: Liu et al. (2024), https://arxiv.org/abs/2310.06625

Autoformer    – Seasonal-trend decomposition + FFT-based auto-correlation
    attention.  Trend is projected linearly; seasonal component is encoded by
    the auto-correlation encoder.
    Ref: Wu et al. (2021), https://arxiv.org/abs/2106.13008

CausalTransformer – Decoder-only transformer with causal (upper-triangular)
    attention mask.  Processes the input sequence step-by-step; uses the last
    token's representation for direct multi-step prediction.  "CausalFormer"
    is not an established single paper — this is the most natural interpretation
    (autoregressive masked attention for sequential forecasting).
"""

import time
import numpy as np
import torch
import torch.nn as nn

from .base import Baseline
from ..base import BaseTask


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

class _MovingAvg(nn.Module):
    """Centered moving-average pooling for trend extraction."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_p = torch.cat([front, x, end], dim=1)           # (B, T+2*pad, C)
        trend = self.avg(x_p.permute(0, 2, 1))            # (B, C, T)
        return trend.permute(0, 2, 1)                      # (B, T, C)


class _SeriesDecomp(nn.Module):
    """Decompose into (seasonal, trend) via centered moving average."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.ma = _MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.ma(x)
        return x - trend, trend                            # seasonal, trend


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))       # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1), :])


# ---------------------------------------------------------------------------
# 1. iTransformer
# ---------------------------------------------------------------------------

class _iTransformerModel(nn.Module):
    """
    Inverted Transformer (Liu et al., 2024).

    Input  : (B, T, n_vars)
    Step 1 : Permute → (B, n_vars, T)
    Step 2 : Linear projection per variate: T → d_model  →  (B, n_vars, d_model)
    Step 3 : TransformerEncoder with attention OVER variates (not time)
    Step 4 : Output linear on the glucose variate's token (index 0): d_model → pred_len
    Output : (B, pred_len, 1)
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_vars)
        x = x.permute(0, 2, 1)         # (B, n_vars, T)
        x = self.input_proj(x)          # (B, n_vars, d_model)
        x = self.encoder(x)             # (B, n_vars, d_model)  — attention over variates
        out = self.output_proj(x[:, 0, :])  # glucose variate (idx 0)  → (B, pred_len)
        return out.unsqueeze(-1)        # (B, pred_len, 1)


# ---------------------------------------------------------------------------
# 2. Autoformer
# ---------------------------------------------------------------------------

class _AutoCorrelation(nn.Module):
    """
    Simplified Auto-Correlation block (Wu et al., 2021).

    Computes time-lag correlations via FFT and aggregates values at the
    top-k most correlated delays.  The per-sample delay is approximated
    by the mean delay across the batch and heads (sufficient for on-the-fly
    single-window training).
    """

    def __init__(self, d_model: int, n_heads: int, top_k: int = 3, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.top_k = top_k

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)  — self-attention
        B, T, _ = x.shape

        def split(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        Q = split(self.q(x))   # (B, H, T, d_head)
        K = split(self.k(x))
        V = split(self.v(x))

        # FFT cross-correlation
        q_f = torch.fft.rfft(Q, dim=2)
        k_f = torch.fft.rfft(K, dim=2)
        corr = torch.fft.irfft(q_f * k_f.conj(), n=T, dim=2)  # (B, H, T, d_head)

        top_k = min(self.top_k, T)
        scores = corr.mean(dim=-1)                               # (B, H, T)
        topk_vals, topk_idx = torch.topk(scores, top_k, dim=-1) # (B, H, top_k)
        weights = torch.softmax(topk_vals, dim=-1)               # (B, H, top_k)

        # Time-delay aggregation with mean-lag approximation
        agg = torch.zeros_like(V)                                # (B, H, T, d_head)
        for j in range(top_k):
            lag = int(topk_idx[:, :, j].float().mean().round().item()) % T
            rolled = torch.roll(V, shifts=lag, dims=2)
            w = weights[:, :, j].unsqueeze(-1).unsqueeze(-1)     # (B, H, 1, 1)
            agg = agg + w * rolled

        out = self.drop(agg.transpose(1, 2).reshape(B, T, self.n_heads * self.d_head))
        return self.out(out)


class _AutoformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        top_k: int = 3,
        kernel_size: int = 25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.autocorr = _AutoCorrelation(d_model, n_heads, top_k=top_k, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.decomp = _SeriesDecomp(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Auto-correlation sub-layer + decomp residual
        residual, _ = self.decomp(self.norm1(x + self.autocorr(x)))
        # Feed-forward sub-layer + decomp residual
        residual, _ = self.decomp(self.norm2(residual + self.ff(residual)))
        return residual


class _AutoformerModel(nn.Module):
    """
    Simplified Autoformer (Wu et al., 2021).

    Prediction = seasonal forecast (auto-correlation encoder) + trend forecast (linear).
    Covariates are embedded as additional input channels.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        top_k: int = 3,
        kernel_size: int = 25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.decomp = _SeriesDecomp(kernel_size)
        self.embed = nn.Linear(n_vars, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        self.encoder = nn.ModuleList(
            [
                _AutoformerEncoderLayer(
                    d_model, n_heads, d_ff, top_k=top_k,
                    kernel_size=kernel_size, dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

        # Seasonal: last encoder token → pred_len
        self.seasonal_proj = nn.Linear(d_model, pred_len)
        # Trend: glucose trend across time → pred_len
        self.trend_proj = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_vars)
        seasonal, trend = self.decomp(x)

        # Trend prediction from glucose channel (variate 0)
        trend_pred = self.trend_proj(trend[:, :, 0])       # (B, pred_len)

        # Seasonal encoding
        h = self.pos_enc(self.embed(seasonal))             # (B, T, d_model)
        for layer in self.encoder:
            h = layer(h)
        h = self.norm(h)
        seasonal_pred = self.seasonal_proj(h[:, -1, :])    # (B, pred_len)

        return (seasonal_pred + trend_pred).unsqueeze(-1)   # (B, pred_len, 1)


# ---------------------------------------------------------------------------
# 3. CausalTransformer
# ---------------------------------------------------------------------------

class _CausalTransformerModel(nn.Module):
    """
    Decoder-only transformer with causal (masked) self-attention.

    Each timestep can only attend to itself and earlier positions (upper-
    triangular mask).  Covariates are concatenated to glucose at each step
    before the input embedding.  The last token's representation is used for
    direct multi-step prediction.

    Note: "CausalFormer" is not a single established paper; this implements
    the most natural interpretation — an autoregressive-style masked transformer
    for sequential time-series forecasting.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_vars: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(n_vars, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_vars)
        T = x.size(1)
        # Causal mask: True = masked (ignored), upper-triangular excluding diagonal
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.pos_enc(self.embed(x))        # (B, T, d_model)
        h = self.decoder(h, mask=mask)          # (B, T, d_model)
        out = self.output_proj(h[:, -1, :])     # last token → (B, pred_len)
        return out.unsqueeze(-1)                # (B, pred_len, 1)


# ---------------------------------------------------------------------------
# Unified forecaster wrapper
# ---------------------------------------------------------------------------

class TransformerForecaster(Baseline):
    """
    On-the-fly transformer forecaster.

    Fits one of three architectures (iTransformer / Autoformer / CausalTransformer)
    from scratch on the context window of each task instance, then makes a point
    prediction.  Uncertainty is represented by adding Gaussian noise scaled to
    the residual std of the history (same approach as DLinearForecaster).

    Parameters
    ----------
    model_type   : "itransformer" | "autoformer" | "causal"
    use_context  : if True, concatenate past covariates (c_cov["past"]) to the
                   glucose series as additional input variates.
    d_model      : transformer hidden dimension
    n_heads      : attention heads (must divide d_model)
    n_layers     : number of encoder/decoder layers
    d_ff         : feed-forward intermediate size
    n_epochs     : gradient steps for on-context fit
    lr           : Adam learning rate
    kernel_size  : moving-average window for Autoformer decomposition
    top_k        : number of top correlation lags for Autoformer
    dropout      : dropout rate
    seed         : random seed
    """

    __version__ = "0.1.0"

    def __init__(
        self,
        model_type: str = "itransformer",
        use_context: bool = False,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        n_epochs: int = 200,
        lr: float = 1e-3,
        kernel_size: int = 25,
        top_k: int = 3,
        dropout: float = 0.1,
        seed: int = 42,
    ):
        assert model_type in ("itransformer", "autoformer", "causal"), (
            f"model_type must be 'itransformer', 'autoformer', or 'causal', got {model_type!r}"
        )
        self.model_type = model_type
        self.use_context = use_context
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.n_epochs = n_epochs
        self.lr = lr
        self.kernel_size = kernel_size
        self.top_k = top_k
        self.dropout = dropout
        self.seed = seed
        super().__init__()

    # ------------------------------------------------------------------
    def _get_inputs(self, task_instance: BaseTask):
        """
        Extract model inputs from the task instance.

        Returns
        -------
        x    : np.ndarray (seq_len, n_vars) — input features
        hist : np.ndarray (seq_len,)        — glucose history (for noise estimation)
        """
        hist_col = task_instance.past_time.columns[-1]
        hist = task_instance.past_time[hist_col].values.astype(np.float32)   # (C,)

        has_cov = (
            self.use_context
            and hasattr(task_instance, "c_cov")
            and task_instance.c_cov is not None
        )

        if has_cov:
            cov = task_instance.c_cov["past"].astype(np.float32)             # (C, K)
            x = np.concatenate([hist[:, None], cov], axis=1)                 # (C, 1+K)
        else:
            x = hist[:, None]                                                 # (C, 1)

        return x, hist

    # ------------------------------------------------------------------
    def __call__(self, task_instance: BaseTask, n_samples: int) -> tuple:
        t0 = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        x, hist = self._get_inputs(task_instance)
        pred_len = len(task_instance.future_time)
        seq_len, n_vars = x.shape
        device = "cuda" if torch.cuda.is_available() else "cpu"

        fit_len = max(seq_len - pred_len, 1)

        # Training tensors
        x_in = torch.tensor(
            x[:fit_len], dtype=torch.float32, device=device
        ).unsqueeze(0)                                     # (1, fit_len, n_vars)
        y_in = torch.tensor(
            hist[fit_len:], dtype=torch.float32, device=device
        ).view(1, pred_len, 1)                             # (1, pred_len, 1)

        # Build & train
        model = self._make_model(fit_len, pred_len, n_vars).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        t_fit = time.time()
        model.train()
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            loss = criterion(model(x_in), y_in)
            loss.backward()
            optimizer.step()
        fit_time = time.time() - t_fit

        # Inference
        t_inf = time.time()
        model.eval()
        with torch.no_grad():
            point_pred = model(x_in).squeeze(0).cpu().numpy()  # (pred_len, 1)
        inf_time = time.time() - t_inf

        # Sample uncertainty via residual noise
        noise_std = float(np.std(np.diff(hist))) if len(hist) > 1 else 1.0
        samples = (
            point_pred[None]
            + np.random.randn(n_samples, pred_len, 1) * noise_std * 0.1
        )                                                  # (n_samples, pred_len, 1)

        extra_info = {
            "total_time": time.time() - t0,
            "fit_time": fit_time,
            "inf_time": inf_time,
            "n_vars": n_vars,
        }
        return samples, extra_info

    # ------------------------------------------------------------------
    def _make_model(self, seq_len: int, pred_len: int, n_vars: int) -> nn.Module:
        common = dict(
            seq_len=seq_len,
            pred_len=pred_len,
            n_vars=n_vars,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
        )
        if self.model_type == "itransformer":
            return _iTransformerModel(**common)
        elif self.model_type == "autoformer":
            ks = min(self.kernel_size, seq_len - 1) if seq_len > 1 else 1
            ks = max(ks | 1, 1)   # ensure odd and ≥ 1
            return _AutoformerModel(**common, top_k=self.top_k, kernel_size=ks)
        else:  # causal
            return _CausalTransformerModel(**common)

    # ------------------------------------------------------------------
    @property
    def cache_name(self) -> str:
        ctx = "ctx" if self.use_context else "noctx"
        return (
            f"{self.__class__.__name__}_{self.model_type}_{ctx}"
            f"_ep{self.n_epochs}_d{self.d_model}"
        )
