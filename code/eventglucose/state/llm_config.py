"""
LLM Configuration Loader
========================

Module Structure
----------------

┌─────────────────────────────────────────────────────────────────────┐
│                      llm_config.py                                  │
│              YAML-based LLM Provider Configuration                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CONFIG SOURCE: config/dikw-agent.yaml (SINGLE SOURCE OF TRUTH)     │
│                                                                     │
│  RESOLUTION PRIORITY (highest to lowest)                            │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  1. planner.nodes.generate_plan.llm_backend (per-node)     │    │
│  │  2. planner.llm_backend (per-agent)                        │    │
│  │  3. default.llm_backend (global) - REQUIRED                │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  LLMNodeConfig DATACLASS                                            │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  llm_backend ────> "direct" | "ccsdk"                      │    │
│  │  model ──────────> e.g., "gpt-4o", "claude-3-5-sonnet"     │    │
│  │  temperature ────> float (0.0-1.0)                         │    │
│  │  max_tokens ─────> int                                     │    │
│  │  enable_fallback ─> bool                                   │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  MAIN FUNCTIONS                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  get_llm_config(agent, node) ─> LLMNodeConfig              │    │
│  │  reload_config() ─────────────> Refresh from YAML          │    │
│  │  get_config_path() ───────────> Path to YAML file          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Usage
-----

::

    from haiagent.dikwgraph.state.llm_config import get_llm_config

    config = get_llm_config(agent="planner_agent", node="generate_plan")
    print(config.llm_backend)  # "direct" or "ccsdk"
    print(config.model)        # "gpt-4o"
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Literal, List
from pathlib import Path
from dataclasses import dataclass
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

BackendType = Literal["direct", "ccsdk"]
AgentType = Literal[
    "dikw_agent", "planner_agent",
    "data_agent", "information_agent", "knowledge_agent", "wisdom_agent",
    "task_execution"
]


# =============================================================================
# CONFIG DATA CLASS
# =============================================================================

@dataclass
class LLMNodeConfig:
    """Configuration for a specific LLM invocation."""

    llm_backend: BackendType
    model: str
    temperature: float
    max_tokens: int
    enable_fallback: bool

    # Source tracking (for debugging)
    resolved_from: str = "default"  # e.g., "planner.nodes.generate_plan", "planner", "default"

    # Alias for backward compatibility
    @property
    def backend(self) -> BackendType:
        """Alias for llm_backend (backward compatibility)."""
        return self.llm_backend

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "llm_backend": self.llm_backend,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enable_fallback": self.enable_fallback,
            "resolved_from": self.resolved_from,
        }


# =============================================================================
# CONFIG LOADER
# =============================================================================

# Default config path
CONFIG_DIR = Path(__file__).parent.parent / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "dikw-agent.yaml"

# Cached config
_config_cache: Optional[Dict[str, Any]] = None
_config_path: Optional[Path] = None


def _load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load and cache YAML configuration.

    The YAML file is REQUIRED - raises error if not found.
    """
    global _config_cache, _config_path

    # Use provided path or default
    path = config_path or DEFAULT_CONFIG_PATH

    # Return cached config if same path
    if _config_cache is not None and _config_path == path:
        return _config_cache

    # Load config - REQUIRED (no fallback)
    if not path.exists():
        raise FileNotFoundError(
            f"LLM config file not found: {path}\n"
            f"This file is REQUIRED. Please create it with the default settings.\n"
            f"See config/dikw-agent.yaml for the expected format."
        )

    try:
        with open(path, 'r', encoding='utf-8') as f:
            _config_cache = yaml.safe_load(f) or {}
        _config_path = path
        logger.info(f"Loaded LLM config from: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load LLM config from {path}: {e}")

    # Validate that 'default' section exists
    if "default" not in _config_cache:
        raise ValueError(
            f"LLM config file {path} must have a 'default' section.\n"
            f"This section defines the base configuration for all agents."
        )

    return _config_cache


def reload_config(config_path: Optional[Path] = None) -> None:
    """Force reload of configuration."""
    global _config_cache, _config_path
    _config_cache = None
    _config_path = None
    _load_config(config_path)


def _get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Get nested config value."""
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def _resolve_value(
    config: Dict[str, Any],
    agent: Optional[str],
    node: Optional[str],
    key: str,
) -> tuple[Any, str]:
    """
    Resolve a config value with priority order (YAML only).

    Priority:
    1. Per-node setting (agent.nodes.node.key)
    2. Per-agent setting (agent.key)
    3. Global default (default.key)

    Returns:
        Tuple of (value, source) where source indicates where value came from.

    Raises:
        ValueError if key not found in any level
    """
    # Priority 1: Per-node setting
    if agent and node:
        value = _get_nested(config, agent, "nodes", node, key)
        if value is not None:
            return value, f"{agent}.nodes.{node}"

    # Priority 2: Per-agent setting
    if agent:
        value = _get_nested(config, agent, key)
        if value is not None:
            return value, agent

    # Priority 3: Global default from config (REQUIRED)
    value = _get_nested(config, "default", key)
    if value is not None:
        return value, "default"

    # No fallback - error if not found
    raise ValueError(
        f"LLM config key '{key}' not found.\n"
        f"Checked: {agent}.nodes.{node}.{key}, {agent}.{key}, default.{key}\n"
        f"Please add '{key}' to the 'default' section of config/dikw-agent.yaml"
    )


# =============================================================================
# MAIN API
# =============================================================================

def get_llm_config(
    agent: Optional[AgentType] = None,
    node: Optional[str] = None,
    config_path: Optional[Path] = None,
) -> LLMNodeConfig:
    """
    Get LLM configuration for a specific agent/node.

    All configuration comes from the YAML file - no hardcoded defaults.

    Args:
        agent: Agent type ("dikw_agent", "decision_agent", "planner_agent",
               "data_agent", "information_agent", "knowledge_agent", "wisdom_agent")
        node: Node/function name (e.g., "generate_plan", "reasoning_synthesis")
        config_path: Optional custom config file path

    Returns:
        LLMNodeConfig with resolved settings

    Raises:
        FileNotFoundError: If YAML config file not found
        ValueError: If required config keys are missing

    Examples:
        # Get config for planner's generate_plan node
        config = get_llm_config(agent="planner_agent", node="generate_plan")

        # Get config for knowledge agent's reasoning_synthesis
        config = get_llm_config(agent="knowledge_agent", node="reasoning_synthesis")

        # Get global defaults
        config = get_llm_config()
    """
    config = _load_config(config_path)

    # Resolve each setting from YAML (no hardcoded defaults)
    llm_backend, backend_src = _resolve_value(config, agent, node, "llm_backend")
    model, model_src = _resolve_value(config, agent, node, "model")
    temperature, temp_src = _resolve_value(config, agent, node, "temperature")
    max_tokens, tokens_src = _resolve_value(config, agent, node, "max_tokens")
    enable_fallback, fallback_src = _resolve_value(config, agent, node, "enable_fallback")

    # Determine primary source (use llm_backend source as representative)
    resolved_from = backend_src

    # Handle type conversions
    if isinstance(temperature, str):
        temperature = float(temperature)
    if isinstance(max_tokens, str):
        max_tokens = int(max_tokens)
    if isinstance(enable_fallback, str):
        enable_fallback = enable_fallback.lower() in ("true", "1", "yes")

    return LLMNodeConfig(
        llm_backend=llm_backend,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_fallback=enable_fallback,
        resolved_from=resolved_from,
    )


def get_all_agent_configs() -> Dict[str, Dict[str, LLMNodeConfig]]:
    """
    Get all agent configurations (useful for debugging/display).

    Returns:
        Dict mapping agent -> node -> config
    """
    config = _load_config()
    agents = [
        "dikw_agent", "planner_agent",
        "data_agent", "information_agent", "knowledge_agent", "wisdom_agent",
        "task_execution"
    ]

    result = {}
    for agent in agents:
        agent_config = config.get(agent, {})
        nodes = agent_config.get("nodes", {})

        result[agent] = {
            "_default": get_llm_config(agent=agent),  # Agent-level default
        }

        for node_name in nodes.keys():
            result[agent][node_name] = get_llm_config(agent=agent, node=node_name)

    return result


def print_config_summary() -> None:
    """Print a summary of all LLM configurations."""
    configs = get_all_agent_configs()

    print("\n" + "=" * 70)
    print("LLM Configuration Summary (from config/dikw-agent.yaml)")
    print("=" * 70)

    for agent, nodes in configs.items():
        print(f"\n{agent.upper()}")
        print("-" * 40)

        for node, cfg in nodes.items():
            if node == "_default":
                print(f"  (default): {cfg.llm_backend:8} | {cfg.model} | from={cfg.resolved_from}")
            else:
                print(f"  {node:25} {cfg.llm_backend:8} | temp={cfg.temperature} | from={cfg.resolved_from}")

    print("\n" + "=" * 70)


def get_config_file_path() -> Path:
    """Get the path to the config file."""
    return DEFAULT_CONFIG_PATH


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "BackendType",
    "AgentType",
    "LLMNodeConfig",
    "get_llm_config",
    "get_all_agent_configs",
    "reload_config",
    "print_config_summary",
    "get_config_file_path",
]
