"""
Workspace Tools for DIKWAgent
=============================

Module Structure
----------------

┌─────────────────────────────────────────────────────────────────────┐
│                    tools_workspace.py                               │
│              Workspace Utility Tools (Lightweight)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PURPOSE: Tools for Decision Agent conversation interface           │
│           (NO code execution - just file exploration)               │
│                                                                     │
│  AVAILABLE TOOLS (WORKSPACE_TOOLS list)                             │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                                                            │    │
│  │  list_workspace_files(subfolder?, pattern?)                │    │
│  │    └─> List files in workspace folder                      │    │
│  │        Default: source/raw/, Pattern: "*.csv"              │    │
│  │                                                            │    │
│  │  read_workspace_file(file_path, max_lines?)                │    │
│  │    └─> Read file content (max 1000 lines, truncated)       │    │
│  │                                                            │    │
│  │  get_workspace_tree(max_depth?)                            │    │
│  │    └─> Get ASCII directory tree structure                  │    │
│  │                                                            │    │
│  │  check_dikw_status()                                       │    │
│  │    └─> Check D/I/K/W level completion status               │    │
│  │                                                            │    │
│  │  get_plan_summary()                                        │    │
│  │    └─> Get current DIKW plan (D/I/K/W instructions)        │    │
│  │                                                            │    │
│  │  read_report(level, task_name?)                            │    │
│  │    └─> Read generated report for a level                   │    │
│  │                                                            │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  WORKSPACE PATH MANAGEMENT                                          │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  set_workspace_path(path) ─> Set global path               │    │
│  │  get_workspace_path() ────> Get current path               │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Tool Usage in Conversation
--------------------------

::

    User: "What files are in my data folder?"
      │
      ▼
    process_response_node detects tool request
      │
      ▼
    state.pending_tool_call = {"tool_name": "list_workspace_files", ...}
      │
      ▼
    tool_call_node executes list_workspace_files("source/raw")
      │
      ▼
    Returns formatted file list to user

Note: For code execution, use D/I agents with Claude SDK.
      For deep analysis, use K/W agents with LLM reasoning.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Global workspace path - set by DIKWAgent before tool use
_WORKSPACE_PATH: Optional[Path] = None


def set_workspace_path(path: Path) -> None:
    """Set the workspace path for tools to use."""
    global _WORKSPACE_PATH
    _WORKSPACE_PATH = Path(path)
    logger.info(f"Workspace tools: path set to {_WORKSPACE_PATH}")


def get_workspace_path() -> Optional[Path]:
    """Get the current workspace path."""
    return _WORKSPACE_PATH


def _resolve_path(relative_path: str) -> Optional[Path]:
    """Resolve a relative path against the workspace root."""
    if _WORKSPACE_PATH is None:
        return None

    # Handle empty path
    if not relative_path or relative_path in [".", ""]:
        return _WORKSPACE_PATH

    # Resolve and ensure within workspace (security)
    resolved = (_WORKSPACE_PATH / relative_path).resolve()
    if not str(resolved).startswith(str(_WORKSPACE_PATH.resolve())):
        logger.warning(f"Path escape attempt blocked: {relative_path}")
        return None

    return resolved


# =============================================================================
# WORKSPACE TOOLS
# =============================================================================

@tool
def list_workspace_files(
    subfolder: str = "",
    pattern: str = "*",
    max_files: int = 50
) -> str:
    """
    List files in the DIKW workspace.

    Args:
        subfolder: Subfolder to list (e.g., "source/raw", "reports/d", "code/d").
                   Empty string or "." lists the root workspace.
        pattern: Glob pattern to filter files (default: "*" for all files).
                 Examples: "*.csv", "*.py", "*.md"
        max_files: Maximum number of files to return (default: 50)

    Returns:
        List of files with their sizes, or error message if path invalid.

    Examples:
        - list_workspace_files() -> lists root workspace
        - list_workspace_files("source/raw") -> lists raw data files
        - list_workspace_files("reports", "*.md") -> lists markdown reports
    """
    if _WORKSPACE_PATH is None:
        return "Error: Workspace path not set. Run agent first."

    target_path = _resolve_path(subfolder)
    if target_path is None:
        return f"Error: Invalid path '{subfolder}'"

    if not target_path.exists():
        return f"Folder does not exist: {subfolder or 'workspace root'}"

    if not target_path.is_dir():
        return f"Not a directory: {subfolder}"

    # List files
    try:
        files = list(target_path.glob(pattern))[:max_files]

        if not files:
            return f"No files matching '{pattern}' in {subfolder or 'workspace root'}"

        # Format output
        lines = [f"Files in {subfolder or 'workspace root'} (pattern: {pattern}):"]
        lines.append("-" * 50)

        for f in sorted(files):
            if f.is_file():
                size = f.stat().st_size
                if size < 1024:
                    size_str = f"{size} B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f} MB"

                rel_path = f.relative_to(target_path)
                lines.append(f"  {rel_path} ({size_str})")
            elif f.is_dir():
                lines.append(f"  {f.name}/ (folder)")

        if len(files) >= max_files:
            lines.append(f"  ... (showing first {max_files} files)")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing files: {e}"


@tool
def read_workspace_file(
    file_path: str,
    max_lines: int = 100,
    start_line: int = 0
) -> str:
    """
    Read a file from the DIKW workspace.

    Args:
        file_path: Relative path from workspace root (e.g., "source/raw/data.csv")
        max_lines: Maximum lines to return (default: 100)
        start_line: Line number to start from (default: 0, first line)

    Returns:
        File contents (truncated if exceeds max_lines), or error message.

    Examples:
        - read_workspace_file("source/raw/data.csv") -> first 100 lines
        - read_workspace_file("reports/d/exploration.md", 50) -> first 50 lines
        - read_workspace_file("code/d/analysis.py", 100, 50) -> lines 50-150
    """
    if _WORKSPACE_PATH is None:
        return "Error: Workspace path not set. Run agent first."

    target_path = _resolve_path(file_path)
    if target_path is None:
        return f"Error: Invalid path '{file_path}'"

    if not target_path.exists():
        return f"File does not exist: {file_path}"

    if not target_path.is_file():
        return f"Not a file: {file_path}"

    try:
        # Check file size
        size = target_path.stat().st_size
        if size > 10 * 1024 * 1024:  # 10MB limit
            return f"File too large ({size / (1024*1024):.1f} MB). Max 10MB."

        # Read file
        with open(target_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()

        total_lines = len(lines)

        # Slice lines
        end_line = start_line + max_lines
        selected_lines = lines[start_line:end_line]

        # Format output
        header = f"File: {file_path} (lines {start_line+1}-{min(end_line, total_lines)} of {total_lines})"
        separator = "-" * 50
        content = "".join(selected_lines)

        result = f"{header}\n{separator}\n{content}"

        if end_line < total_lines:
            result += f"\n{separator}\n... truncated ({total_lines - end_line} more lines)"

        return result

    except UnicodeDecodeError:
        return f"Binary file (cannot display): {file_path}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def get_workspace_tree(max_depth: int = 3) -> str:
    """
    Get the folder structure of the DIKW workspace.

    Args:
        max_depth: Maximum depth to traverse (default: 3)

    Returns:
        Tree-style representation of the workspace structure.

    Example output:
        workspace/
        ├── source/
        │   └── raw/
        │       ├── data.csv
        │       └── metadata.json
        ├── code/
        │   ├── d/
        │   └── i/
        └── reports/
            ├── d/
            └── i/
    """
    if _WORKSPACE_PATH is None:
        return "Error: Workspace path not set. Run agent first."

    def _tree(path: Path, prefix: str = "", depth: int = 0) -> List[str]:
        if depth >= max_depth:
            return [f"{prefix}..."]

        lines = []
        try:
            items = sorted(path.iterdir())
            dirs = [i for i in items if i.is_dir() and not i.name.startswith('.')]
            files = [i for i in items if i.is_file() and not i.name.startswith('.')]

            # Show directories first
            all_items = dirs + files[:10]  # Limit files per folder

            for i, item in enumerate(all_items):
                is_last = (i == len(all_items) - 1)
                connector = "└── " if is_last else "├── "

                if item.is_dir():
                    lines.append(f"{prefix}{connector}{item.name}/")
                    extension = "    " if is_last else "│   "
                    lines.extend(_tree(item, prefix + extension, depth + 1))
                else:
                    lines.append(f"{prefix}{connector}{item.name}")

            if len(files) > 10:
                lines.append(f"{prefix}    ... ({len(files) - 10} more files)")

        except PermissionError:
            lines.append(f"{prefix}(permission denied)")

        return lines

    try:
        tree_lines = [f"{_WORKSPACE_PATH.name}/"]
        tree_lines.extend(_tree(_WORKSPACE_PATH))
        return "\n".join(tree_lines)
    except Exception as e:
        return f"Error generating tree: {e}"


@tool
def check_dikw_status() -> str:
    """
    Check the current status of DIKW analysis.

    Returns:
        Summary of:
        - Which levels (D/I/K/W) have been completed
        - What reports have been generated
        - Current plan status

    Use this to understand what analysis has been done and what remains.
    """
    if _WORKSPACE_PATH is None:
        return "Error: Workspace path not set. Run agent first."

    status_lines = ["DIKW Analysis Status", "=" * 50]

    # Check reports for each level
    levels = ["d", "i", "k", "w"]
    level_names = {"d": "Data", "i": "Information", "k": "Knowledge", "w": "Wisdom"}

    for level in levels:
        reports_dir = _WORKSPACE_PATH / "reports" / level
        code_dir = _WORKSPACE_PATH / "code" / level

        level_status = []

        # Check reports
        if reports_dir.exists():
            reports = list(reports_dir.glob("*.md"))
            if reports:
                level_status.append(f"Reports: {len(reports)} files")

        # Check code
        if code_dir.exists():
            code_files = list(code_dir.glob("*.py"))
            if code_files:
                level_status.append(f"Code: {len(code_files)} files")

        if level_status:
            status_lines.append(f"✅ {level_names[level]} ({level.upper()}-level): {', '.join(level_status)}")
        else:
            status_lines.append(f"⬜ {level_names[level]} ({level.upper()}-level): Not started")

    # Check for plan
    plan_file = _WORKSPACE_PATH / "plan.json"
    if plan_file.exists():
        status_lines.append("")
        status_lines.append("📋 Plan: Generated")

    # Check source data
    source_dir = _WORKSPACE_PATH / "source" / "raw"
    if source_dir.exists():
        source_files = list(source_dir.iterdir())
        status_lines.append("")
        status_lines.append(f"📁 Source data: {len(source_files)} files")

    return "\n".join(status_lines)


@tool
def get_plan_summary() -> str:
    """
    Get a summary of the current DIKW analysis plan.

    Returns:
        Summary of the plan including:
        - High-level questions being answered
        - Tasks for each DIKW level
        - Current progress

    Use this to understand what the analysis is trying to accomplish.
    """
    if _WORKSPACE_PATH is None:
        return "Error: Workspace path not set. Run agent first."

    # Try to find plan file
    plan_file = _WORKSPACE_PATH / "plan.json"

    if not plan_file.exists():
        # Check langgraph folder
        lg_plan = _WORKSPACE_PATH / "langgraph" / "plan.json"
        if lg_plan.exists():
            plan_file = lg_plan
        else:
            return "No plan found. The analysis hasn't started yet, or the plan hasn't been generated."

    try:
        import json
        with open(plan_file, 'r') as f:
            plan_data = json.load(f)

        lines = ["DIKW Analysis Plan", "=" * 50]

        # Extract plan details
        if "high_level_questions" in plan_data:
            lines.append("")
            lines.append("Questions to Answer:")
            for i, q in enumerate(plan_data["high_level_questions"], 1):
                lines.append(f"  {i}. {q}")

        # Show plan for each level
        for level in ["D", "I", "K", "W"]:
            if level in plan_data:
                lines.append("")
                lines.append(f"{level}-Level Plan:")
                plan_text = plan_data[level]
                # Truncate if too long
                if len(plan_text) > 200:
                    plan_text = plan_text[:200] + "..."
                lines.append(f"  {plan_text}")

        return "\n".join(lines)

    except json.JSONDecodeError:
        return "Plan file exists but is not valid JSON."
    except Exception as e:
        return f"Error reading plan: {e}"


@tool
def read_report(level: str, report_name: str = "") -> str:
    """
    Read a generated DIKW report.

    Args:
        level: DIKW level ("d", "i", "k", or "w")
        report_name: Specific report filename (optional).
                     If empty, reads the main report for that level.

    Returns:
        Report contents, or list of available reports if multiple exist.

    Examples:
        - read_report("d") -> read D-level report
        - read_report("k", "synthesis.md") -> read specific K-level report
    """
    if _WORKSPACE_PATH is None:
        return "Error: Workspace path not set. Run agent first."

    level = level.lower()
    if level not in ["d", "i", "k", "w"]:
        return f"Invalid level '{level}'. Must be d, i, k, or w."

    reports_dir = _WORKSPACE_PATH / "reports" / level

    if not reports_dir.exists():
        return f"No reports found for {level.upper()}-level. Analysis may not have reached this level yet."

    # Find reports
    reports = list(reports_dir.glob("*.md"))
    if not reports:
        return f"No markdown reports in {level.upper()}-level reports folder."

    # If specific report requested
    if report_name:
        target = reports_dir / report_name
        if not target.exists():
            available = [r.name for r in reports]
            return f"Report '{report_name}' not found. Available: {', '.join(available)}"
        report_path = target
    else:
        # Use first/main report
        report_path = sorted(reports)[0]

    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Truncate if too long
        max_chars = 5000
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n... (truncated, {len(content) - max_chars} more characters)"

        header = f"Report: {report_path.name} ({level.upper()}-level)"
        return f"{header}\n{'=' * 50}\n\n{content}"

    except Exception as e:
        return f"Error reading report: {e}"


# =============================================================================
# TOOL LIST FOR EXPORT
# =============================================================================

WORKSPACE_TOOLS = [
    list_workspace_files,
    read_workspace_file,
    get_workspace_tree,
    check_dikw_status,
    get_plan_summary,
    read_report,
]

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "set_workspace_path",
    "get_workspace_path",
    "list_workspace_files",
    "read_workspace_file",
    "get_workspace_tree",
    "check_dikw_status",
    "get_plan_summary",
    "read_report",
    "WORKSPACE_TOOLS",
]
