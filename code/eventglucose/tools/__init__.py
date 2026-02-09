"""
DIKW Tools Layer
================

This module provides tools for the DIKW Agent system.

Architecture Overview
---------------------

┌─────────────────────────────────────────────────────────────────────┐
│                         DIKW TOOLS LAYER                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  EXECUTION TOOLS (D/I Levels - code generation + execution)        │
│  ├── claude_sdk_tool  - Claude Agent SDK (subscription-based)      │
│  ├── codex_sdk_tool   - Codex SDK (OpenAI Codex CLI)               │
│  └── openhand_sdk_tool - OpenHands SDK (docker-based)              │
│                                                                     │
│  LLM ABSTRACTION (All Levels - pure LLM calls)                      │
│  └── LLMProvider - Unified interface with "direct"/"ccsdk" backends│
│                                                                     │
│  WORKSPACE TOOLS (Decision Agent - read-only exploration)          │
│  └── WORKSPACE_TOOLS - List of langchain tools for file ops        │
│                                                                     │
│  UTILITIES                                                          │
│  └── run_workspace_scripts - Batch script execution                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Usage Examples
--------------

Execution (D/I levels)::

    from haiagent.dikwgraph.tools import claude_sdk_tool, ClaudeSDKArgs

    args = ClaudeSDKArgs(
        prompt="Analyze the data...",
        topic_type="data",
        topic_idx=1,
        topic_name="exploration",
        workspace_dir="projspace/my_project"
    )
    result = claude_sdk_tool(args)

LLM Abstraction (All levels)::

    from haiagent.dikwgraph.tools import get_llm_provider

    provider = get_llm_provider(agent="planner_agent", node="generate_plan")
    response = provider.invoke("Generate a plan for...")

Workspace Tools (Decision Agent)::

    from haiagent.dikwgraph.tools import WORKSPACE_TOOLS, set_workspace_path

    set_workspace_path("/path/to/workspace")
    # WORKSPACE_TOOLS contains: list_workspace_files, read_workspace_file, etc.
"""

# =============================================================================
# EXECUTION TOOLS (D/I Levels)
# =============================================================================

from .tools_ccsdk import (
    ClaudeSDKArgs,
    ClaudeSDKResults,
    claude_sdk_tool,
)

# Codex SDK - optional dependency
try:
    from .tools_codexsdk import (
        CodexSDKArgs,
        CodexSDKResults,
        codex_sdk_tool,
    )
except ImportError:
    # codex-client not installed - skip these imports
    CodexSDKArgs = None
    CodexSDKResults = None
    codex_sdk_tool = None

# OpenHands SDK - optional dependency
try:
    from .tools_ohsdk import (
        OpenHandSDKArgs,
        OpenHandSDKResults,
        openhand_sdk_tool,
    )
except ImportError:
    # OpenHands not installed - skip these imports
    OpenHandSDKArgs = None
    OpenHandSDKResults = None
    openhand_sdk_tool = None

# =============================================================================
# LLM ABSTRACTION (All Levels)
# =============================================================================

from .tools_llm_provider import (
    BackendType,
    LLMResult,
    AgenticTaskResult,
    LLMProviderConfig,
    LLMProvider,
    get_llm_provider,
    invoke_llm,
)

# =============================================================================
# WORKSPACE TOOLS (Decision Agent)
# =============================================================================

# Workspace tools - optional dependency (requires langchain)
try:
    from .tools_workspace import (
        WORKSPACE_TOOLS,
        set_workspace_path,
        get_workspace_path,
        list_workspace_files,
        read_workspace_file,
        get_workspace_tree,
        check_dikw_status,
        get_plan_summary,
        read_report,
    )
except ImportError:
    # Langchain not installed - skip workspace tools
    WORKSPACE_TOOLS = None
    set_workspace_path = None
    get_workspace_path = None
    list_workspace_files = None
    read_workspace_file = None
    get_workspace_tree = None
    check_dikw_status = None
    get_plan_summary = None
    read_report = None

# =============================================================================
# UTILITIES
# =============================================================================

from .tools_runcode import (
    run_workspace_scripts,
    run_workspace_docker,
)

# =============================================================================
# INTERNAL UTILITIES (for use by tool modules)
# =============================================================================

from ._auth import (
    AuthMethod,
    SubscriptionAuthContext,
    EnsureClaudeSDKAuth,
    configure_claude_auth,
    restore_api_key,
    get_api_key_for_model,
    # New API key management
    get_anthropic_api_key,
    has_anthropic_api_key,
    has_subscription_auth,
)

from ._sdk_interface import (
    SDKArgs,
    SDKResult,
    ExecutionSDK,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # --- Execution Tools (D/I Levels) ---
    "ClaudeSDKArgs",
    "ClaudeSDKResults",
    "claude_sdk_tool",
    "CodexSDKArgs",
    "CodexSDKResults",
    "codex_sdk_tool",
    "OpenHandSDKArgs",
    "OpenHandSDKResults",
    "openhand_sdk_tool",

    # --- LLM Abstraction (All Levels) ---
    "BackendType",
    "LLMResult",
    "AgenticTaskResult",
    "LLMProviderConfig",
    "LLMProvider",
    "get_llm_provider",
    "invoke_llm",

    # --- Workspace Tools (Decision Agent) ---
    "WORKSPACE_TOOLS",
    "set_workspace_path",
    "get_workspace_path",
    "list_workspace_files",
    "read_workspace_file",
    "get_workspace_tree",
    "check_dikw_status",
    "get_plan_summary",
    "read_report",

    # --- Utilities ---
    "run_workspace_scripts",
    "run_workspace_docker",

    # --- Internal (for tool modules) ---
    "AuthMethod",
    "SubscriptionAuthContext",
    "EnsureClaudeSDKAuth",
    "configure_claude_auth",
    "restore_api_key",
    "get_api_key_for_model",
    "get_anthropic_api_key",
    "has_anthropic_api_key",
    "has_subscription_auth",
    "SDKArgs",
    "SDKResult",
    "ExecutionSDK",
]
