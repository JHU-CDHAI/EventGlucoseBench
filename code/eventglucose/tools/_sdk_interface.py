"""
SDK Interface - Abstract Base for Execution SDKs
=================================================

This module defines the abstract interface that all execution SDKs must implement.
Currently supports:
- Claude Agent SDK (tools_ccsdk.py)
- OpenHands SDK (tools_ohsdk.py)

Future SDKs can easily be added by implementing the ExecutionSDK interface.

Architecture
------------

::

    ┌─────────────────────────────────────────────────────────────────┐
    │                    ExecutionSDK (Abstract)                      │
    │                                                                 │
    │  Properties:                                                    │
    │    - name: str           # SDK identifier                       │
    │    - supports_code_exec  # Can run code?                        │
    │    - supports_file_ops   # Can read/write files?                │
    │                                                                 │
    │  Methods:                                                       │
    │    - execute(args) -> SDKResult                                 │
    │    - validate_workspace(path) -> bool                           │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
    ┌─────────┴─────────┐         ┌──────────┴──────────┐
    │ ClaudeAgentSDK    │         │ OpenHandsSDK        │
    │ (tools_ccsdk.py)  │         │ (tools_ohsdk.py)    │
    │                   │         │                     │
    │ - Subscription    │         │ - Docker sandbox    │
    │ - Async execution │         │ - LLM agnostic      │
    │ - Claude models   │         │ - Multi-model       │
    └───────────────────┘         └─────────────────────┘

Usage
-----

::

    from eventglucose.tools._sdk_interface import SDKArgs, SDKResult

    # Common args for any SDK
    args = SDKArgs(
        prompt="Analyze the customer data",
        topic_type="data",
        topic_idx=1,
        topic_name="exploration",
        workspace_dir="projspace/my_project"
    )

    # Execute with any SDK
    result = some_sdk.execute(args)
    if result.success:
        print(result.analysis_text)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SDKArgs:
    """
    Base arguments for SDK execution.

    This is the common interface for all execution SDKs.
    SDK-specific args classes (ClaudeSDKArgs, OpenHandSDKArgs) extend this.

    Attributes:
        prompt: Task description / instructions for the agent
        topic_type: Category of task ("data", "information", "knowledge", "wisdom")
        topic_idx: Task index within the category
        topic_name: Human-readable task name
        workspace_dir: Root directory for execution
        max_iterations: Maximum agent turns/iterations (default: 30)
        llm_model: Model to use (default: from YAML config)
        llm_api_key: API key override (default: from env)
    """
    # Core inputs
    prompt: str
    topic_type: str
    topic_idx: int
    topic_name: str
    workspace_dir: str

    # Runtime options
    max_iterations: int = 30
    llm_model: Optional[str] = None
    llm_api_key: Optional[str] = None

    # Workspace subpaths (auto-derived if not provided)
    workspace_code_dir: Optional[str] = None
    workspace_data_dir: Optional[str] = None
    workspace_reports_dir: Optional[str] = None
    workspace_runs_dir: Optional[str] = None

    # Run folder (optional - use existing instead of creating new)
    run_folder_path: Optional[str] = None

    # Output probing
    output_probe_files: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Derive missing subpaths from workspace_dir."""
        root = Path(self.workspace_dir).absolute()
        self.workspace_dir = str(root)

        if not self.workspace_code_dir:
            self.workspace_code_dir = str(root / "code")
        if not self.workspace_data_dir:
            self.workspace_data_dir = str(root / "data")
        if not self.workspace_reports_dir:
            self.workspace_reports_dir = str(root / "report")
        if not self.workspace_runs_dir:
            self.workspace_runs_dir = str(root / "runs")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "topic_type": self.topic_type,
            "topic_idx": self.topic_idx,
            "topic_name": self.topic_name,
            "workspace_dir": self.workspace_dir,
            "max_iterations": self.max_iterations,
            "llm_model": self.llm_model,
            "workspace_code_dir": self.workspace_code_dir,
            "workspace_data_dir": self.workspace_data_dir,
            "workspace_reports_dir": self.workspace_reports_dir,
            "workspace_runs_dir": self.workspace_runs_dir,
            "run_folder_path": self.run_folder_path,
        }


@dataclass
class SDKResult:
    """
    Base result from SDK execution.

    This is the common interface for all execution SDK results.
    SDK-specific result classes extend this with additional fields.

    Attributes:
        success: Whether execution completed successfully
        message: Human-readable status message
        run_dir: Directory where execution artifacts are stored
        log_file: Path to execution log file
        analysis_text: Generated report/analysis content (if any)
        analysis_file: Path to generated report file (if any)
        total_cost: LLM cost in USD (if available)
        execution_status: Final status string (e.g., "FINISHED", "ERROR")
        error_message: Error details if success=False
    """
    # Core result
    success: bool
    message: str

    # Execution artifacts
    run_dir: str
    log_file: str

    # Generated outputs
    analysis_text: Optional[str] = None
    analysis_file: Optional[str] = None

    # Metrics
    total_cost: Optional[float] = None
    num_turns: Optional[int] = None

    # Status
    execution_status: Optional[str] = None
    error_message: Optional[str] = None

    # Additional metadata
    config_echo: Optional[Dict[str, Any]] = None
    log_tail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "run_dir": self.run_dir,
            "log_file": self.log_file,
            "analysis_text": self.analysis_text,
            "analysis_file": self.analysis_file,
            "total_cost": self.total_cost,
            "num_turns": self.num_turns,
            "execution_status": self.execution_status,
            "error_message": self.error_message,
        }


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class ExecutionSDK(ABC):
    """
    Abstract interface for code execution SDKs.

    All execution SDKs (Claude Agent SDK, OpenHands, etc.) should implement
    this interface to ensure consistent behavior.

    Subclasses must implement:
        - execute(args) -> SDKResult
        - name property

    Example implementation::

        class MyCustomSDK(ExecutionSDK):
            @property
            def name(self) -> str:
                return "my_custom_sdk"

            @property
            def supports_code_execution(self) -> bool:
                return True

            def execute(self, args: SDKArgs) -> SDKResult:
                # Implementation here
                pass
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        SDK identifier for logging and selection.

        Returns:
            SDK name string (e.g., "claude_sdk", "openhands", "aider")
        """
        pass

    @property
    def supports_code_execution(self) -> bool:
        """
        Whether this SDK can execute code.

        Returns:
            True if SDK can run Python/bash code
        """
        return True

    @property
    def supports_file_operations(self) -> bool:
        """
        Whether this SDK can read/write files.

        Returns:
            True if SDK can create/modify files in workspace
        """
        return True

    @abstractmethod
    def execute(self, args: SDKArgs) -> SDKResult:
        """
        Execute a task with this SDK.

        Args:
            args: SDKArgs with task configuration

        Returns:
            SDKResult with execution outcome

        Raises:
            Exception: If execution fails catastrophically
        """
        pass

    def validate_workspace(self, workspace_dir: str) -> bool:
        """
        Validate that workspace is properly configured.

        Args:
            workspace_dir: Path to workspace directory

        Returns:
            True if workspace is valid, False otherwise
        """
        workspace = Path(workspace_dir)
        return workspace.exists() and workspace.is_dir()

    def ensure_workspace_structure(self, workspace_dir: str) -> None:
        """
        Create standard workspace directory structure.

        Creates: code/, data/, report/, runs/

        Args:
            workspace_dir: Path to workspace root
        """
        root = Path(workspace_dir)
        for subdir in ["code", "data", "report", "runs"]:
            (root / subdir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# SDK REGISTRY
# =============================================================================

# Registry of available SDKs (populated by SDK modules)
_SDK_REGISTRY: Dict[str, type] = {}


def register_sdk(name: str, sdk_class: type) -> None:
    """
    Register an SDK class in the global registry.

    Args:
        name: SDK identifier (e.g., "claude_sdk", "openhands")
        sdk_class: SDK class that implements ExecutionSDK
    """
    _SDK_REGISTRY[name] = sdk_class


def get_sdk(name: str) -> Optional[type]:
    """
    Get an SDK class from the registry.

    Args:
        name: SDK identifier

    Returns:
        SDK class or None if not found
    """
    return _SDK_REGISTRY.get(name)


def list_sdks() -> List[str]:
    """
    List all registered SDK names.

    Returns:
        List of SDK identifier strings
    """
    return list(_SDK_REGISTRY.keys())


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "SDKArgs",
    "SDKResult",

    # Abstract base
    "ExecutionSDK",

    # Registry
    "register_sdk",
    "get_sdk",
    "list_sdks",
]
