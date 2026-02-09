"""
State: Single State + Graph Routing Architecture
================================================

Unified AgentState for the DIKW agent system. State holds DATA, Graph holds
ROUTING. Single state class used by all nodes.

Module Structure
----------------

┌─────────────────────────────────────────────────────────────────────┐
│                     state/__init__.py                               │
│              Unified AgentState for DIKW System                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  DESIGN PRINCIPLES                                                  │
│  • State holds DATA, Graph holds ROUTING                            │
│  • Single state class used by all nodes                             │
│  • No transform functions needed                                    │
│  • DecisionAgent is the hub - all agents report to it               │
│                                                                     │
│  STATE MODELS (from unified_state.py)                               │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  AgentState ───────────> Main state (alias DIKWUnifiedState)│    │
│  │  DIKWPlan ─────────────> Plan with D/I/K/W instructions    │    │
│  │  DIKWStep ─────────────> Step in plan                      │    │
│  │  DIKWStepResult ───────> Result of executing step          │    │
│  │  DIKWMetadata ─────────> Project paths and configuration   │    │
│  │  Task ─────────────────> Unified task class                │    │
│  │  TaskResult ───────────> Result of task execution          │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ROUTING MODELS                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  AgentRecommendation, DecisionAnalysisResult, RoutingDecision   │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  AGENT CONFIG (from agent_config.py)                                │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  LevelConfig, PlannerConfig, DATA_CONFIG, etc.             │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Usage
-----

::

    from haiagent.dikwgraph.state import AgentState

    state = AgentState(
        high_level_questions=["What patterns exist?"],
        review_mode="auto",
    )
    result = graph.invoke(state.model_dump())
"""

import os

from haiagent.dikwgraph.state.unified_state import (
    DIKWUnifiedState,
    DIKWPlan,
    DIKWStep,
    DIKWStepResult,
    DIKWMetadata,
    Task,
    TaskResult,
    # Routing models (new)
    AgentRecommendation,
    DecisionAnalysisResult,
    RoutingDecision,
    # Decision point models
    ResultAnalysis,
    DecisionRecommendation,
    HumanResponse,
    DecisionPoint,
)

from haiagent.dikwgraph.state.agent_config import (
    LevelType,
    AgentType,
    LevelConfig,
    PlannerConfig,
    LEVEL_AGENT_INFO,
    DATA_CONFIG,
    INFORMATION_CONFIG,
    KNOWLEDGE_CONFIG,
    WISDOM_CONFIG,
    PLANNER_CONFIG,
)

def get_model_name(override: str = None, agent: str = None, node: str = None) -> str:
    """
    Get LLM model name from YAML config.

    No fallbacks - raises error if YAML config is not available.

    Args:
        override: Direct override value
        agent: Agent name for YAML lookup (e.g., "decision", "planner")
        node: Node name for YAML lookup (e.g., "routing_analysis")

    Returns:
        Model name string

    Raises:
        FileNotFoundError: If YAML config file not found
        ValueError: If required config keys missing
    """
    if override:
        return override

    from haiagent.dikwgraph.state.llm_config import get_llm_config
    config = get_llm_config(agent=agent, node=node)
    return config.model


class AgentState(DIKWUnifiedState):
    """Compatibility layer that aliases AgentState to the unified state model.

    The new single-state architecture is still converging, so we inherit the
    existing DIKWUnifiedState schema to keep LangGraph nodes in sync while the
    planner/agent refactor progresses.
    """
    pass

__all__ = [
    "AgentState",
    "get_model_name",
    # State models
    "DIKWUnifiedState",
    "DIKWPlan",
    "DIKWStep",
    "DIKWStepResult",
    "DIKWMetadata",
    "Task",
    "TaskResult",
    # Routing models
    "AgentRecommendation",
    "DecisionAnalysisResult",
    "RoutingDecision",
    # Decision point models
    "ResultAnalysis",
    "DecisionRecommendation",
    "HumanResponse",
    "DecisionPoint",
    # Agent config models
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
