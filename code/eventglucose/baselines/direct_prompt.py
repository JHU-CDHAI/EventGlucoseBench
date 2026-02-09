"""
Direct prompt method

"""
import inspect
import logging
import numpy as np
import os
import hashlib
import torch
import requests
from functools import partial
import time
import re

from transformers import pipeline
from types import SimpleNamespace
from lmformatenforcer import JsonSchemaParser, RegexParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

from .base import Baseline
from ..config import (
    LLAMA31_405B_URL,
    LLAMA31_405B_API_KEY,
    OPENAI_API_KEY,
    OPENAI_API_VERSION,
    OPENAI_AZURE_ENDPOINT,
    OPENAI_USE_AZURE,
    ANTHROPIC_API_KEY,
)
from .utils import extract_html_tags

from .hf_utils.dp_hf_api import LLM_MAP, get_model_and_tokenizer, hf_generate

# Claude Code SDK imports (for subscription-based inference)
# Note: Imports moved to ccsdk_client function to avoid circular imports

# For OpenRouter
from openai import OpenAI
from os import getenv

logger = logging.getLogger("DirectPrompt")

# Default system prompt used for chat-based models (OpenAI/OpenRouter/local chat templates).
DEFAULT_SYSTEM_PROMPT = "You are a useful forecasting assistant."

# System prompt used for any model whose name contains both "llama" and "70b"
LLAMA70_SYSTEM_PROMPT = "You can predict future values."








# As of 28 Sep 2024
OPENROUTER_COSTS = {
    "openrouter-llama-3-8b-instruct-DeepInfra": {"input": 0.000055, "output": 0.000055},
    "openrouter-llama-3-8b-instruct-NovitaAI": {"input": 0.000063, "output": 0.000063},
    "openrouter-llama-3-8b-instruct-Together": {"input": 0.00007, "output": 0.00007},
    "openrouter-llama-3-8b-instruct-Lepton": {"input": 0.000162, "output": 0.000162},
    "openrouter-llama-3-8b-instruct-Mancer": {"input": 0.0001875, "output": 0.001125},
    "openrouter-llama-3-8b-instruct-Fireworks": {"input": 0.0002, "output": 0.0002},
    "openrouter-llama-3-8b-instruct-Mancer (private)": {"input": 0.00025,  "output": 0.0015},
    "openrouter-llama-3-70b-instruct-DeepInfra": {"input": 0.00035, "output": 0.0004},
    "openrouter-llama-3-70b-instruct-NovitaAI": {"input": 0.00051, "output": 0.00074},
    "openrouter-llama-3-70b-instruct-Together": {"input": 0.000792, "output": 0.000792},
    "openrouter-llama-3-70b-instruct-Lepton": {"input": 0.0008, "output": 0.0008},
    "openrouter-llama-3-70b-instruct-Fireworks": {"input": 0.0009, "output": 0.0009},
    "openrouter-mixtral-8x7b-instruct-DeepInfra": {"input": 0.00024, "output": 0.00024},
    "openrouter-mixtral-8x7b-instruct-Fireworks": {"input": 0.0005, "output": 0.0005},
    "openrouter-mixtral-8x7b-instruct-Lepton": {"input": 0.0005, "output": 0.0005},
    "openrouter-mixtral-8x7b-instruct-Together": {"input": 0.00054, "output": 0.00054},
}


def dict_to_obj(data):
    if isinstance(data, dict):
        # Recursively convert dictionary values
        return SimpleNamespace(
            **{key: dict_to_obj(value) for key, value in data.items()}
        )
    elif isinstance(data, list):
        # Recursively convert each item in the list
        return [dict_to_obj(item) for item in data]
    else:
        # Return the data if it's neither a dict nor a list
        return data



def openrouter_client(model, messages, n=1, max_tokens=1500, temperature=1.0):
    """
    Client for OpenRouter chat models
    """
    # gets API Key from environment variable OPENAI_API_KEY
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=getenv("OPENROUTER_API_KEY"),
    )

    # Extract model name without "openrouter-" prefix
    model_suffix = model[11:]

    # Model migration: llama-3-70b-instruct is broken on OpenRouter
    # (Hyperbolic returns empty, Novita stops early). Use llama-3.1 instead.
    if model_suffix == "llama-3-70b-instruct":
        model_suffix = "llama-3.1-70b-instruct"
        logger.info("OpenRouter: llama-3-70b-instruct broken, using llama-3.1-70b-instruct")

    # Model ID mapping: internal name -> OpenRouter model ID
    if model_suffix == "qwen3-235b-a22b-instruct":
        model_suffix = "qwen3-235b-a22b-2507"
        logger.info("OpenRouter: mapping qwen3-235b-a22b-instruct to qwen3-235b-a22b-2507")

    if model_suffix.startswith("llama"):
        model_from = "meta-llama"
    elif (
        model_suffix.startswith("mist")
        or model_suffix.startswith("mixt")
        or model_suffix.startswith("Mist")
    ):
        model_from = "mistralai"
    elif model_suffix.startswith("qwen"):
        model_from = "qwen"
    elif model_suffix.startswith("gemini"):
        model_from = "google"
    elif model_suffix.startswith("claude"):
        model_from = "anthropic"

    full_model_name = f"{model_from}/{model_suffix}"

    completion = client.chat.completions.create(
        model=full_model_name,
        messages=messages,
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return completion


def anthropic_client(model, messages, n=1, max_tokens=1000, temperature=1.0):
    """
    Client for Anthropic Claude models.

    Returns an OpenAI-compatible response object for consistency with DirectPrompt.
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package is required. Install with: pip install anthropic")

    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY (CIK_ANTHROPIC_API_KEY) is not set. Please set it in env.sh.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Map internal model names to Anthropic model IDs
    model_map = {
        "claude-4.5-opus": "claude-opus-4-5-20251101",
        "claude-4.5-sonnet": "claude-sonnet-4-5-20250514",
    }
    anthropic_model = model_map.get(model, model)

    # Extract system message if present
    system_content = None
    user_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system_content = msg.get("content", "")
        else:
            user_messages.append(msg)

    # Claude API doesn't support n>1, so we make multiple requests
    choices = []
    total_input_tokens = 0
    total_output_tokens = 0

    for _ in range(n):
        kwargs = {
            "model": anthropic_model,
            "max_tokens": max_tokens,
            "messages": user_messages,
        }
        if system_content:
            kwargs["system"] = system_content
        if temperature != 1.0:
            kwargs["temperature"] = temperature

        response = client.messages.create(**kwargs)

        # Extract text content from response
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        choices.append(SimpleNamespace(message=SimpleNamespace(content=content)))
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

    # Build OpenAI-compatible response
    usage = SimpleNamespace(
        prompt_tokens=total_input_tokens,
        completion_tokens=total_output_tokens,
    )
    return SimpleNamespace(choices=choices, usage=usage)


def ccsdk_client(model, messages, n=1, max_tokens=1000, temperature=1.0):
    """
    Client for Claude Code SDK (subscription-based).

    Uses ~/.claude subscription when available, falls back to API key.
    Returns OpenAI-compatible response object for consistency with DirectPrompt.

    Args:
        model: Model name (e.g., "claude-sdk-sonnet-4.5-nocontext")
        messages: List of message dicts with "role" and "content"
        n: Number of samples to generate
        max_tokens: Max tokens per response
        temperature: Sampling temperature

    Returns:
        SimpleNamespace with OpenAI-compatible structure:
            - choices: List of responses with .message.content
            - usage: Token usage info
    """
    # Extract SDK model name and map to specific snapshot version
    # Input: "claude-sdk-haiku-4.5-nocontext" -> Output: "claude-haiku-4-5-20251001"
    model_base = model.replace("claude-sdk-", "")  # "haiku-4.5-nocontext"

    # Strip context suffix
    model_base = model_base.replace("-nocontext", "").replace("-context", "")  # "haiku-4.5"

    # Map to specific snapshot versions (Option B: Production-ready with fixed versions)
    model_map = {
        "haiku-4.5": "claude-haiku-4-5-20251001",
        "sonnet-4.5": "claude-sonnet-4-5-20250929",
        "opus-4.5": "claude-opus-4-5-20251101",
    }

    sdk_model = model_map.get(model_base, model_base)  # Fallback to model_base if not in map

    # Build prompt from messages
    prompt_parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt_parts.append(f"System: {content}")
        else:
            prompt_parts.append(content)

    prompt_text = "\n\n".join(prompt_parts)

    # Create CCSDK provider with explicit config (no YAML required)
    from ..tools.tools_llm_provider import LLMProvider, LLMProviderConfig

    # System prompt: instruct model to give direct output without tool use
    system_prompt = """You are a helpful assistant that provides direct, concise responses.

IMPORTANT INSTRUCTIONS:
- Provide your answer directly in the response
- Do NOT use any tools or functions
- Do NOT perform extended thinking or reasoning steps
- Simply analyze the prompt and give the requested output immediately
- Follow the exact format specified in the user's request"""

    config = LLMProviderConfig(
        llm_backend="ccsdk",
        model=sdk_model,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_fallback=False,
        resolved_from="direct_prompt",
        ccsdk_allowed_tools=[],  # Disable all tools - just want text completion
        ccsdk_max_turns=1,  # Single turn response (no agent loops)
        ccsdk_system_prompt=system_prompt,  # Instruct: direct output only
    )
    provider = LLMProvider(config)

    # Generate n samples (loop similar to anthropic_client)
    choices = []
    total_cost = 0.0

    for _ in range(n):
        response = provider.invoke(prompt_text, temperature=temperature)
        choices.append(SimpleNamespace(message=SimpleNamespace(content=response)))

    # Build OpenAI-compatible response
    usage = SimpleNamespace(
        prompt_tokens=0,  # Not tracked by CCSDK
        completion_tokens=0,
    )
    return SimpleNamespace(choices=choices, usage=usage)


class DirectPrompt(Baseline):
    """
    A simple baseline that uses any instruction-tuned LLM to produce forecastss

    Parameters:
    -----------
    model: str
        The name of the model to use for forecasting
    use_context: bool, default=True
        If True, use context in the prompt, otherwise ignore it
    fail_on_invalid: bool, default=True
        If True, raise an exception if an invalid sample is encountered
        in the forecast. Otherwise, print a warning and skip the sample.
    n_retries: int, default=3
        The number of retries to use in rejection sampling
    max_batch_size: int, default=None
        If not None, the maximum batch size on the attemps (before the retries)
    batch_size_on_retry: int, default=5
        The batch size to use on retries
    constrained_decoding: bool, default=True
        If True, use constrained decoding to ensure the model returns the forecast in the expected format.
        Note: this is only supported for HuggingFace models.
    token_cost: dict, default=None
            The cost of tokens used in the API call. If provided, the cost of the API call will be estimated.
            Expected keys are "input" and "output" for the price of input and output tokens, respectively.

    """

    __version__ = "0.0.5"  # Modification will trigger re-caching

    def __init__(
        self,
        model,
        use_context=True,
        fail_on_invalid=True,
        n_retries=3,
        batch_size_on_retry=5,
        batch_size=None,
        constrained_decoding=True,
        token_cost: dict = None,
        temperature: float = 1.0,
        dry_run: bool = False,
        sleep_between_requests: float = 0.0) -> None:

        self.model = model
        self.use_context = use_context
        self.fail_on_invalid = fail_on_invalid
        if model == "llama-3.1-405b-instruct" or model == "llama-3.1-405b":
            self.n_retries = 10
        elif model.startswith("openrouter-"):
            self.n_retries = 50  # atleast 25 required since batch size is 1
        else:
            self.n_retries = n_retries
        self.batch_size = batch_size
        # OpenRouter doesn't support batch requests (n>1), so use batch_size=1
        # Exception: Mixtral may work with batch_size > 1
        if model.startswith("openrouter-") and "mixtral" not in model.lower():
            self.batch_size_on_retry = 1
        else:
            self.batch_size_on_retry = batch_size_on_retry
        self.constrained_decoding = constrained_decoding
        self.token_cost = token_cost
        self.total_input_cost = 0  # For OpenRouter
        self.total_output_cost = 0  # For OpenRouter
        self.total_cost = 0  # Accumulator for monetary value of queries
        self.temperature = temperature
        self.dry_run = dry_run
        self.sleep_between_requests = sleep_between_requests  # Sleep time in seconds between API requests

        # ------------------------------------------------------------------
        # Model-specific system prompt override (Option A)
        #
        # By default all models get DEFAULT_SYSTEM_PROMPT.
        # For any model with "llama" and "70b" in its name, use LLAMA70_SYSTEM_PROMPT.
        #
        # NOTE: We incorporate a short hash into the cache key (when overridden)
        # so cached results remain consistent when you edit the prompt.
        # ------------------------------------------------------------------
        _model_lc = str(self.model).lower()
        if ("llama" in _model_lc) and ("70b" in _model_lc):
            self.system_prompt = LLAMA70_SYSTEM_PROMPT
        else:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self._system_prompt_hash = hashlib.sha1(self.system_prompt.encode("utf-8")).hexdigest()[:8]

        if not dry_run and self.model in LLM_MAP.keys():
            self.llm, self.tokenizer = get_model_and_tokenizer(
                llm_path=None, llm_type=self.model
            )
        else:
            self.llm, self.tokenizer = None, None
        self.client = self.get_client()

    def get_client(self):
        """
        Setup the OpenAI client based on configuration preferences
        """
        if self.model.startswith("gpt"):
            logger.info("Using standard OpenAI client.")
            from openai import OpenAI
            # print(OPENAI_API_KEY)
            client = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create

        # elif self.model == "llama-3.1-405b-instruct":
        #     return partial(llama_3_1_405b_instruct_client, temperature=self.temperature)

        elif self.model.startswith("openrouter-"):
            return partial(
                openrouter_client,
                temperature=self.temperature,
            )

        # Claude Code SDK models (subscription-based) - check BEFORE generic "claude-"
        elif self.model.startswith("claude-sdk-"):
            return partial(
                ccsdk_client,
                temperature=self.temperature,
            )

        # Anthropic Claude models (API-based)
        elif self.model.startswith("claude-"):
            return partial(
                anthropic_client,
                temperature=self.temperature,
            )

        # Local HuggingFace models (e.g., qwen2.5-0.5B-Instruct, llama-3-8B, etc.)
        elif self.model in LLM_MAP:
            return partial(self._hf_client, temperature=self.temperature)

        else:
            raise NotImplementedError(f"Model {self.model} not supported.")

        return client

    def _is_api_based(self):
        """
        Check if the model uses an API (OpenAI, OpenRouter, Claude, etc.) rather than local inference.

        Returns:
            bool: True if the model is API-based, False otherwise
        """
        return (
            self.model.startswith("gpt") or
            self.model.startswith("openrouter-") or
            self.model.startswith("claude-") or
            (self.model == "llama-3.1-405b-instruct" or self.model == "llama-3.1-405b")
        )

    def _hf_client(
        self,
        model: str,
        messages,
        n: int = 1,
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: float = 0.95,
        future_timestamps=None,
    ):
        """
        Minimal local HF "chat.completions.create"-like wrapper used by DirectPrompt.__call__().

        Returns an object with:
        - `.choices[i].message.content`
        - `.usage.prompt_tokens` / `.usage.completion_tokens`

        Notes:
        - We ignore `future_timestamps` (constrained decoding is only supported by some API backends).
        - This path is intended for local smoke tests; performance depends on your hardware.
        """
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError(
                f"Local HF client requested for {model}, but model/tokenizer are not loaded."
            )

        # Build a single prompt string from the message list.
        # Prefer the tokenizer's chat template when available (best for instruct-tuned models).
        prompt_text = None
        try:
            if hasattr(self.tokenizer, "apply_chat_template"):
                prompt_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        except Exception:
            prompt_text = None

        if not prompt_text:
            prompt_text = "\n".join(
                [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
            )

        # Replicate prompt n times (DirectPrompt uses n=batch_size for sampling).
        prompts = [prompt_text] * int(n)

        # Tokenize and generate on the best available device.
        import torch
        import os

        # Determine a reasonable device target (works for single-device models).
        try:
            device = next(self.llm.parameters()).device
        except Exception:
            device = torch.device("cpu")

        # IMPORTANT: cap prompt length for local inference (especially on macOS MPS/CPU).
        # Without truncation, long history prompts can create huge attention buffers.
        max_prompt_tokens = int(os.environ.get("EVENTGLUCOSE_HF_MAX_PROMPT_TOKENS", "2048"))
        batch = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        prompt_tokens = int(batch["input_ids"].numel())

        with torch.inference_mode():
            generate_ids = self.llm.generate(
                **batch,
                do_sample=True,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                renormalize_logits=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Strip prompt tokens and decode only the newly generated text
        num_input_ids = batch["input_ids"].shape[1]
        gen_strs = self.tokenizer.batch_decode(
            generate_ids[:, num_input_ids:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        completion_tokens = int(generate_ids[:, num_input_ids:].numel())

        # Build OpenAI-like response shape
        from types import SimpleNamespace

        choices = [
            SimpleNamespace(message=SimpleNamespace(content=s)) for s in gen_strs
        ]
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return SimpleNamespace(choices=choices, usage=usage)

    


    def __call__(self, task_instance, n_samples):
        """
        Infer forecasts from the model

        Parameters:
        -----------
        task_instance: TimeSeriesTask
            The task instance to forecast
        n_samples: int
            The number of samples to generate
        n_retries: int
            The number of rejection sampling steps
        batch_size_on_retry: int
            The batch size to use on retries. This is useful to avoid asking for way too many samples
            from the API.

        Returns:
        --------
        samples: np.ndarray, shape [n_samples, time dimension, number of variables]
            The forecast samples. Note: only univariate is supported at the moment (number of variables = 1)
        extra_info: dict
            A dictionary containing informations pertaining to the cost of running this model
        """

        default_batch_size = n_samples if not self.batch_size else self.batch_size
        if self.batch_size:
            assert (
                self.batch_size * self.n_retries >= n_samples
            ), f"Not enough iterations to cover {n_samples} samples"
        assert (
            self.batch_size_on_retry <= default_batch_size
        ), f"Batch size on retry should be equal to or less than {default_batch_size}"

        starting_time = time.time()
        total_client_time = 0.0

        # prompt = self.make_prompt(task_instance)
        use_context = self.use_context
        try:
            use_patient_info = self.use_patient_info
        except Exception:
            use_patient_info = False

        # Tasks may or may not precompute `task_instance.prompt`.
        # If absent, fall back to building a prompt using the shared prompt generator.
        prompt = getattr(task_instance, "prompt", None)
        if prompt is None:
            try:
                from ..prompts.make_prompt import MakePrompt
                prompt_maker = MakePrompt()
                prompt = prompt_maker(task_instance, level="noctx")
            except Exception as e:
                raise ValueError(
                    "Prompt is None and automatic prompt construction failed. "
                    "Provide task_config['prompt_level'] or ensure task_instance.prompt is set."
                ) from e
        
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": prompt},
        ]

        # Get forecast samples via rejection sampling until we have the desired number of samples
        # or until we run out of retries

        total_tokens = {"input": 0, "output": 0}
        valid_forecasts = []

        max_batch_size = task_instance.max_directprompt_batch_size

        # OpenRouter doesn't support batch requests (n>1) - always returns 1 choice
        # and fails silently with 0 completion tokens for n>=10
        if self.model.startswith("openrouter-"):
            max_batch_size = 1
            logger.info("OpenRouter detected: forcing batch_size=1 (batch requests not supported)")

        if max_batch_size is not None:
            batch_size = min(default_batch_size, max_batch_size)
            n_retries = self.n_retries + default_batch_size // batch_size
        else:
            batch_size = default_batch_size
            n_retries = self.n_retries

        llm_outputs = []

        while len(valid_forecasts) < n_samples and n_retries > 0:
            logger.info(f"Requesting forecast of {batch_size} samples from the model.")
            client_start_time = time.time()

            # Pass future timestamps as kwarg in case the client supports constrained decoding
            if "future_timestamps" in inspect.signature(self.client).parameters:
                chat_completion = self.client(
                    model=self.model,
                    n=batch_size,
                    messages=messages,
                    # Pass future timestamps as kwarg in case the client supports constrained decoding
                    future_timestamps=task_instance.future_time.index.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ).values,
                )
            else:
                chat_completion = self.client(
                    model=self.model, n=batch_size, messages=messages
                )

            total_client_time += time.time() - client_start_time
            total_tokens["input"] += chat_completion.usage.prompt_tokens
            total_tokens["output"] += chat_completion.usage.completion_tokens

            # Add sleep between API requests to avoid rate limiting
            if self._is_api_based() and self.sleep_between_requests > 0:
                logger.info(f"Sleeping {self.sleep_between_requests}s between API requests to avoid rate limiting...")
                time.sleep(self.sleep_between_requests)

            logger.info("Parsing forecasts from completion.")
            for choice in chat_completion.choices:
                llm_outputs.append(choice.message.content)
                try:
                    # Extract forecast from completion
                    forecast = extract_html_tags(choice.message.content, ["forecast"])[
                        "forecast"
                    ][0]
                    forecast = forecast.replace("(", "").replace(")", "")
                    forecast = forecast.split("\n")
                    forecast = {
                        x.split(",")[0]
                        .replace("'", "")
                        .replace('"', ""): float(x.split(",")[1])
                        for x in forecast
                    }

                    # Get forecasted values at expected timestamps (will fail if model hallucinated timestamps, which is ok)
                    forecast = [
                        forecast[t]
                        for t in task_instance.future_time.index.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    ]

                    # Append forecast to list of valid forecasts
                    valid_forecasts.append(forecast)

                    # If OpenRouter, compute costs here as costs differ per call
                    if self.model.startswith("openrouter-"):
                        # Get provider
                        provider = chat_completion.provider
                        # Make string
                        model_name = self.model + "-" + provider
                        # Compute costs with Openrouter cost dict
                        if model_name in OPENROUTER_COSTS:
                            input_cost = (
                                total_tokens["input"]
                                / 1000
                                * OPENROUTER_COSTS[model_name]["input"]
                            )
                            output_cost = (
                                total_tokens["output"]
                                / 1000
                                * OPENROUTER_COSTS[model_name]["input"]
                            )
                            current_cost = round(input_cost + output_cost, 2)
                            logger.info(f"Forecast cost: {current_cost}$")
                        else:
                            input_cost = output_cost = current_cost = 0
                            logger.info(f"Cost not recorded")

                        self.total_input_cost += input_cost
                        self.total_output_cost += output_cost
                        self.total_cost += current_cost
                except Exception as e:
                    logger.info("Sample rejected due to invalid format.")
                    logger.debug(f"Rejection details: {e}")
                    logger.debug(f"Choice: {choice.message.content}")

            n_retries -= 1
            if max_batch_size is not None:
                # Do not go down to self.batch_size_on_retry until we are almost done
                remaining_samples = n_samples - len(valid_forecasts)
                batch_size = max(remaining_samples, self.batch_size_on_retry)
                batch_size = min(batch_size, max_batch_size)
            else:
                batch_size = self.batch_size_on_retry

            valid_forecasts = valid_forecasts[:n_samples]
            logger.info(f"Got {len(valid_forecasts)}/{n_samples} valid forecasts.")
            if len(valid_forecasts) < n_samples:
                logger.info(f"Remaining retries: {n_retries}.")

        # If we couldn't collect enough forecasts, handle based on configuration
        if self.fail_on_invalid and len(valid_forecasts) < n_samples:
            raise RuntimeError(
                f"Failed to get {n_samples} valid forecasts. Got {len(valid_forecasts)} instead."
            )
        elif len(valid_forecasts) < n_samples:
            # Partial samples available - will save data but skip evaluation
            logger.warning(
                f"Got {len(valid_forecasts)}/{n_samples} valid forecasts. "
                f"Will save partial results but skip evaluation and visualization."
            )

        extra_info = {
            "total_input_tokens": total_tokens["input"],
            "total_output_tokens": total_tokens["output"],
            "llm_outputs": llm_outputs,
        }
        
        # Mark if we have partial samples (for evaluation.py to skip evaluation)
        if len(valid_forecasts) < n_samples:
            extra_info["partial_samples"] = True
            extra_info["n_samples_requested"] = n_samples
            extra_info["n_samples_obtained"] = len(valid_forecasts)

        # Estimate cost of API calls
        logger.info(f"Total tokens used: {total_tokens}")
        if self.model.startswith("openrouter-"):
            extra_info["input_token_cost"] = self.total_input_cost
            extra_info["output_token_cost"] = self.total_output_cost
            extra_info["total_token_cost"] = self.total_cost

        elif self.token_cost is not None:
            input_cost = total_tokens["input"] / 1000 * self.token_cost["input"]
            output_cost = total_tokens["output"] / 1000 * self.token_cost["output"]
            current_cost = round(input_cost + output_cost, 2)
            logger.info(f"Forecast cost: {current_cost}$")
            self.total_cost += current_cost

            extra_info["input_token_cost"] = self.token_cost["input"]
            extra_info["output_token_cost"] = self.token_cost["output"]
            extra_info["total_token_cost"] = current_cost

        # Convert the list of valid forecasts to a numpy array
        samples = np.array(valid_forecasts)[:, :, None]

        extra_info["total_time"] = time.time() - starting_time
        extra_info["total_client_time"] = total_client_time

        return samples, extra_info

    @property
    def cache_name(self):
        args_to_include = [
            "model",
            "use_context",
            "fail_on_invalid",
            "n_retries",
        ]
        if not self.model.startswith("gpt"):
            args_to_include.append("temperature")
        # Include sleep_between_requests in cache name if non-zero to ensure different cache entries
        if self.sleep_between_requests > 0:
            args_to_include.append("sleep_between_requests")

        # Ensure system-prompt tweaks for this specific model don't collide in cache.
        if self.system_prompt != DEFAULT_SYSTEM_PROMPT:
            args_to_include.append("_system_prompt_hash")

        return f"{self.__class__.__name__}_" + "_".join(
            [f"{k}={getattr(self, k)}" for k in args_to_include]
        )

    # CLAUDE CODE MODIFICATION: Added custom pickle support for multiprocessing compatibility
    # This allows DirectPrompt to be used with max_parallel > 1 in evaluate_all_tasks()
    def __getstate__(self):
        """
        Custom pickle support - exclude non-picklable objects for multiprocessing.

        The DirectPrompt class contains several objects that cannot be pickled:
        - OpenAI/API clients (contain connection objects and threading locks)
        - Transformers models and tokenizers (large objects with internal state)
        - Partial functions used for client methods

        This method removes these objects during serialization so the class can be
        passed to worker processes in multiprocessing.Pool.
        """
        state = self.__dict__.copy()
        # Remove non-picklable objects that will be recreated in the worker process
        state['client'] = None      # API clients (OpenAI, OpenRouter, etc.)
        state['llm'] = None         # HuggingFace transformer models
        state['tokenizer'] = None   # HuggingFace tokenizers
        return state

    def __setstate__(self, state):
        """
        Custom unpickle support - restore non-picklable objects in worker process.

        When a DirectPrompt object is unpickled in a worker process, this method
        recreates the API clients and models that were excluded during pickling.
        Each worker process gets its own independent client/model instances,
        avoiding shared state and threading issues.
        """
        self.__dict__.update(state)

        # Recreate HuggingFace models if needed (for local inference)
        if not self.dry_run and self.model in LLM_MAP.keys():
            self.llm, self.tokenizer = get_model_and_tokenizer(
                llm_path=None, llm_type=self.model
            )
        else:
            self.llm, self.tokenizer = None, None

        # Recreate the API client (OpenAI, OpenRouter, Llama, etc.)
        self.client = self.get_client()
