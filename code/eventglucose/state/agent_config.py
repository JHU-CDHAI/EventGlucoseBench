"""
Agent Configuration Models
==========================

Module Structure
----------------

┌─────────────────────────────────────────────────────────────────────┐
│                     agent_config.py                                 │
│              Centralized DIKW Agent Configuration                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  TYPE DEFINITIONS                                                   │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  LevelType ────> "D" | "I" | "K" | "W"                     │    │
│  │  AgentType ────> "P" | "D" | "I" | "K" | "W"               │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  DISPLAY INFO                                                       │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  LEVEL_AGENT_INFO = {                                      │    │
│  │    "P": {"name": "Planner Agent", "emoji": "📋"},          │    │
│  │    "D": {"name": "Data Agent", "emoji": "📊"},             │    │
│  │    "I": {"name": "Information Agent", "emoji": "📈"},      │    │
│  │    "K": {"name": "Knowledge Agent", "emoji": "🧠"},        │    │
│  │    "W": {"name": "Wisdom Agent", "emoji": "💡"}            │    │
│  │  }                                                         │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  LevelConfig CLASS                                                  │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  level ──────────> "D" | "I" | "K" | "W"                   │    │
│  │  execution_mode ─> "sdk_full" | "sdk_context" | "llm_only" │    │
│  │  prompt_files ───> List of YAML prompt templates           │    │
│  │  context_sources ─> For sdk_context mode                   │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  PRE-CONFIGURED INSTANCES                                           │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  DATA_CONFIG ────────> execution_mode="sdk_full"           │    │
│  │  INFORMATION_CONFIG ─> execution_mode="sdk_full"           │    │
│  │  KNOWLEDGE_CONFIG ───> execution_mode="llm_only"           │    │
│  │  WISDOM_CONFIG ──────> execution_mode="llm_only"           │    │
│  │  PLANNER_CONFIG ─────> execution_mode="llm_only"           │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Usage
-----

::

    from haiagent.dikwgraph.state.agent_config import (
        DATA_CONFIG, INFORMATION_CONFIG, KNOWLEDGE_CONFIG, WISDOM_CONFIG
    )
"""

from typing import List, Literal
from pydantic import BaseModel, Field


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

LevelType = Literal["D", "I", "K", "W"]
AgentType = Literal["P", "D", "I", "K", "W"]  # Includes Planner


# =============================================================================
# LEVEL AGENT INFO (display names and emojis)
# =============================================================================

LEVEL_AGENT_INFO = {
    "P": {"name": "Planner Agent", "emoji": "📋"},
    "D": {"name": "Data Agent", "emoji": "📊"},
    "I": {"name": "Information Agent", "emoji": "📈"},
    "K": {"name": "Knowledge Agent", "emoji": "🧠"},
    "W": {"name": "Wisdom Agent", "emoji": "💡"}
}


# =============================================================================
# CONFIGURATION MODEL
# =============================================================================

class LevelConfig(BaseModel):
    """
    Configuration for a DIKW level agent.

    This defines how each agent (D/I/K/W) operates:
    - Which prompt files to use
    - Which state fields to read/write
    - How to execute tasks (sdk_full, sdk_context, llm_only)
    - What context sources to read (for sdk_context mode)
    """

    # Basic identity
    level: LevelType = Field(..., description="DIKW level: D, I, K, or W")
    level_name: str = Field(..., description="Full name: data, information, knowledge, wisdom")

    # Prompt files (relative to prompts/level_agent/)
    prompt_file: str = Field(..., description="Task generation prompt file")
    aggregation_prompt_file: str = Field(..., description="Results aggregation prompt file")

    # Task template (relative to prompts/level_task/)
    yaml_file: str = Field(..., description="Initial tasks YAML file")

    # State field names
    current_tasks_field: str = Field(..., description="State field for current tasks")
    historical_tasks_field: str = Field(..., description="State field for completed tasks")
    result_field: str = Field(..., description="State field for level results")

    # Execution mode
    agent_execution_mode: Literal["sdk_full", "sdk_context", "llm_only"] = Field(
        default="sdk_full",
        description="""
        How this agent executes tasks:
        - sdk_full: SDK handles everything (D/I default - code execution)
        - sdk_context: SDK gathers context, then LLM reasons (K/W with context)
        - llm_only: Pure LLM reasoning, no SDK (K/W default)
        """
    )

    # Context sources for sdk_context mode
    context_sources: List[str] = Field(
        default_factory=list,
        description="""
        Folders to read for context (relative to workspace).
        Used when agent_execution_mode='sdk_context'.
        Examples: ['source/raw/', 'reports/d/', 'reports/i/']
        """
    )

    # SDK summarization settings (for sdk_context mode)
    use_sdk_summarization: bool = Field(
        default=False,
        description="""
        Use Claude SDK for intelligent context summarization.
        When True, gathered context is summarized before passing to LLM.
        Reduces token count while preserving key information.
        """
    )
    max_summary_tokens: int = Field(
        default=8000,
        description="Maximum tokens for SDK summarization output"
    )


# =============================================================================
# AGENT CONFIGURATIONS
# =============================================================================

DATA_CONFIG = LevelConfig(
    level="D",
    level_name="data",
    prompt_file="data_prompt.txt",
    aggregation_prompt_file="data_aggregation_prompt.txt",
    yaml_file="init_data_tasks.yaml",
    current_tasks_field="current_data_tasks",
    historical_tasks_field="data_tasks",
    result_field="d_results",
    # D-level uses SDK for code execution (default)
    agent_execution_mode="sdk_full",
    context_sources=[]
)

INFORMATION_CONFIG = LevelConfig(
    level="I",
    level_name="information",
    prompt_file="information_prompt.txt",
    aggregation_prompt_file="information_aggregation_prompt.txt",
    yaml_file="init_information_tasks.yaml",
    current_tasks_field="current_information_tasks",
    historical_tasks_field="information_tasks",
    result_field="i_results",
    # I-level uses SDK for code execution (default)
    agent_execution_mode="sdk_full",
    context_sources=[]
)

KNOWLEDGE_CONFIG = LevelConfig(
    level="K",
    level_name="knowledge",
    prompt_file="knowledge_prompt.txt",
    aggregation_prompt_file="knowledge_aggregation_prompt.txt",
    yaml_file="init_knowledge_tasks.yaml",
    current_tasks_field="current_knowledge_tasks",
    historical_tasks_field="knowledge_tasks",
    result_field="k_results",
    # K-level uses SDK to gather context from D/I reports, then LLM reasoning
    agent_execution_mode="sdk_context",
    context_sources=["report/data/", "report/information/"],
    # Enable intelligent summarization of gathered context
    use_sdk_summarization=True,
    max_summary_tokens=8000
)

WISDOM_CONFIG = LevelConfig(
    level="W",
    level_name="wisdom",
    prompt_file="wisdom_prompt.txt",
    aggregation_prompt_file="wisdom_aggregation_prompt.txt",
    yaml_file="init_wisdom_tasks.yaml",
    current_tasks_field="current_wisdom_tasks",
    historical_tasks_field="wisdom_tasks",
    result_field="w_results",
    # W-level uses SDK to gather context from all previous levels, then LLM reasoning
    agent_execution_mode="sdk_context",
    context_sources=["report/data/", "report/information/", "report/knowledge/"],
    # Enable intelligent summarization of gathered context
    use_sdk_summarization=True,
    max_summary_tokens=10000  # W has more context, allow larger summary
)


# =============================================================================
# PLANNER AGENT CONFIGURATION
# =============================================================================

class PlannerConfig(BaseModel):
    """
    Configuration for the Planner Agent (P-level).

    The Planner has a different structure than D/I/K/W agents:
    - No task generation/execution loop
    - Generates/revises DIKW plans
    - Uses context from all levels when revising plans
    """

    # Basic identity
    level: Literal["P"] = Field(default="P", description="Planner level")
    level_name: str = Field(default="planner", description="Full name: planner")

    # Prompt files (relative to prompts/dikw_planner/)
    plan_generation_prompt: str = Field(
        default="plan_generation_prompt.txt",
        description="Prompt file for initial plan generation"
    )
    plan_revision_prompt: str = Field(
        default="plan_revision_prompt.txt",
        description="Prompt file for plan revision"
    )

    # Execution mode for plan revision
    agent_execution_mode: Literal["sdk_full", "sdk_context", "llm_only"] = Field(
        default="sdk_context",
        description="""
        How the planner gathers context when revising plans:
        - sdk_context: SDK gathers context from reports, then LLM revises (default)
        - llm_only: Pure LLM revision without context gathering
        """
    )

    # Context sources for plan revision (sdk_context mode)
    revision_context_sources: List[str] = Field(
        default_factory=lambda: [
            "source/raw/",        # Raw data to understand what's available
            "report/data/",       # D-level reports
            "report/information/", # I-level reports
            "report/knowledge/",  # K-level reports
            "report/wisdom/",     # W-level reports (if any)
        ],
        description="""
        Folders to read when revising plans (relative to workspace).
        Used when agent_execution_mode='sdk_context'.
        Includes all existing reports to understand what's been done.
        """
    )

    # SDK summarization settings
    use_sdk_summarization: bool = Field(
        default=True,
        description="Use Claude SDK for intelligent context summarization"
    )
    max_context_tokens: int = Field(
        default=50000,
        description="Maximum tokens to include in revision context"
    )


PLANNER_CONFIG = PlannerConfig()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "LevelType",
    "AgentType",
    "LevelConfig",
    "PlannerConfig",
    "LEVEL_AGENT_INFO",
    "DATA_CONFIG",
    "INFORMATION_CONFIG",
    "KNOWLEDGE_CONFIG",
    "WISDOM_CONFIG",
    "PLANNER_CONFIG",
]
