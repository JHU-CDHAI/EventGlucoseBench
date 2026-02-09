"""
Claude Agent SDK Tool - Official Anthropic SDK Implementation
=============================================================

Module Structure
----------------

┌─────────────────────────────────────────────────────────────────────┐
│                       tools_ccsdk.py                                │
│               Claude Agent SDK Integration                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MAIN FUNCTION                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  run_with_claude_agent_sdk(task, metadata, verbose)        │    │
│  │    │                                                       │    │
│  │    ▼                                                       │    │
│  │  Flow:                                                     │    │
│  │  ┌─────────────────────────────────────────────────────┐  │    │
│  │  │ 1. Set up workspace folder                          │  │    │
│  │  │    └─> {project}/run_workspace/{task_name}/         │  │    │
│  │  │                                                     │  │    │
│  │  │ 2. Create task prompt from template                 │  │    │
│  │  │    └─> Include: data paths, instructions, output    │  │    │
│  │  │                                                     │  │    │
│  │  │ 3. Execute via Claude Agent SDK                     │  │    │
│  │  │    └─> claude_sdk.run(prompt, workspace_dir)        │  │    │
│  │  │                                                     │  │    │
│  │  │ 4. Parse outputs                                    │  │    │
│  │  │    └─> Extract code files (*.py)                    │  │    │
│  │  │    └─> Extract report files (*.md)                  │  │    │
│  │  │                                                     │  │    │
│  │  │ 5. Copy to project folders                          │  │    │
│  │  │    └─> Code: {project}/code/{level}/{task}/         │  │    │
│  │  │    └─> Reports: {project}/reports/{level}/{task}/   │  │    │
│  │  │                                                     │  │    │
│  │  │ 6. Return TaskResult                                │  │    │
│  │  │    └─> success, paths, summary, timing              │  │    │
│  │  └─────────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  HELPER FUNCTIONS                                                   │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  _build_prompt() ──────> Construct prompt with context     │    │
│  │  _parse_sdk_output() ──> Extract files from SDK result     │    │
│  │  _copy_outputs() ──────> Copy to code/ and reports/        │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  EXECUTION MODES                                                    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  sdk_full ────> Full code execution (D/I levels)           │    │
│  │  sdk_context ─> SDK gathers context, LLM reasons           │    │
│  │  llm_only ────> Pure LLM reasoning (K/W levels)            │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Authentication Priority
-----------------------

1. ~/.claude subscription (HIGHEST PRIORITY - cost effective!)
2. ANTHROPIC_API_KEY env var (fallback - pay-per-token)
3. llm_api_key parameter (lowest priority)

Usage
-----

::

    from eventglucose.tools.tools_ccsdk import ClaudeSDKArgs, claude_sdk_tool

    args = ClaudeSDKArgs(
        prompt="Your task description",
        topic_type="data",
        topic_idx=1,
        topic_name="analysis_task",
        workspace_dir="projspace/proj_analysis"
    )
    result = claude_sdk_tool(args)
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import asyncio

# Claude Agent SDK imports
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
from claude_agent_sdk.types import (
    AssistantMessage,
    UserMessage,
    SystemMessage,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
)

# Configure logging
logger = logging.getLogger(__name__)


# Log content templates
LOG_CONTENT_TEMPLATE = """Claude Agent SDK Execution Log
============================

Run: {run_name}
Agent: {agent_name}
Model: {model}
Max Turns: {max_turns}
Elapsed Time: {elapsed_time:.2f} seconds

Status: {execution_status}
Success: {success}

Final Response:
{final_response}

Metrics:
- Cost: ${total_cost}
- Turns: {num_turns}

"""

ERROR_LOG_TEMPLATE = """Claude Agent SDK Execution Error
==============================

Run: {run_name}
Agent: {agent_name}
Elapsed Time: {elapsed_time:.2f} seconds

Error: {error_message}

Full traceback:
{traceback}
"""


def _slugify(text: str) -> str:
    """Convert text to safe folder name"""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    return text


@dataclass
class ClaudeSDKArgs:
    """Arguments for Claude Agent SDK execution

    API-compatible with OpenHandSDKArgs for easy drop-in replacement.
    """
    # ----- Core input -----
    prompt: str
    topic_type: str          # e.g., "data", "info", "knowledge", "wisdom"
    topic_idx: int           # e.g., 1, 2, 3
    topic_name: str          # e.g., "customer_analysis"

    # ----- Root workspace (host) -----
    workspace_dir: str       # e.g., ".../projspace/proj_x"

    # ----- Subpaths (auto-derived in __post_init__) -----
    workspace_code_dir: str | None = None
    workspace_data_dir: str | None = None
    workspace_reports_dir: str | None = None
    workspace_runs_dir: str | None = None

    # ----- Run folder (optional - use existing run folder instead of creating new one) -----
    run_folder_path: str | None = None      # If provided, use this run folder instead of creating new one

    # ----- Runtime options -----
    max_turns: int = 30  # Claude Agent SDK uses "turns" instead of "iterations"

    # LLM configuration
    llm_model: str | None = None            # Default: "sonnet" (Claude Sonnet 4.5)
    llm_api_key: str | None = None          # Optional: if not set, uses ~/.claude subscription auth

    # Agent configuration
    agent_name: str | None = None           # Default: f"{topic_type}_{topic_name}_agent"

    # Tool permissions
    allowed_tools: list[str] | None = None  # Default: ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]
    permission_mode: str = "acceptEdits"    # Default: auto-accept edits

    # Output probing
    output_probe_files: list[str] | None = None  # default: ["report.md", "analysis_report.md"]

    def __post_init__(self) -> None:
        # Normalize topic fields for safe folder naming
        self.topic_type = _slugify(self.topic_type)
        self.topic_name = _slugify(self.topic_name)

        # Convert workspace_dir to absolute path
        self.workspace_dir = str(Path(self.workspace_dir).absolute())

        # Derive missing subpaths from workspace_dir
        root = Path(self.workspace_dir)
        if not self.workspace_code_dir:
            self.workspace_code_dir = str(root / "code")
        if not self.workspace_data_dir:
            self.workspace_data_dir = str(root / "data")
        if not self.workspace_reports_dir:
            self.workspace_reports_dir = str(root / "report")  # singular, not "reports"
        if not self.workspace_runs_dir:
            self.workspace_runs_dir = str(root / "runs")

        # Ensure required directories exist
        for d in (
            self.workspace_code_dir,
            self.workspace_data_dir,
            self.workspace_reports_dir,
            self.workspace_runs_dir,
        ):
            Path(d).mkdir(parents=True, exist_ok=True)

        # Ensure workspace root directory exists
        Path(self.workspace_dir).mkdir(parents=True, exist_ok=True)

        # Set default agent name if not provided
        if not self.agent_name:
            self.agent_name = f"{self.topic_type}_{self.topic_name}_agent"

        # Set default allowed tools if not provided
        if not self.allowed_tools:
            self.allowed_tools = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ClaudeSDKResults:
    """Results from Claude Agent SDK execution

    API-compatible with OpenHandSDKResults for easy drop-in replacement.
    """
    success: bool
    message: str
    run_dir: str
    logs_dir: str
    log_file: str
    prompt_file_host: str
    analysis_file: str | None = None
    analysis_text: str | None = None
    config_echo: dict[str, Any] | None = None
    log_tail: str | None = None

    # SDK-specific fields
    conversation_id: str | None = None      # Session ID from SDK
    execution_status: str | None = None     # Final execution status
    final_response: str | None = None       # Agent's final response
    total_cost: float | None = None         # Total LLM cost in USD
    total_tokens: int | None = None         # Total tokens used (not available in Claude Agent SDK)
    num_turns: int | None = None            # Number of conversation turns


def claude_sdk_tool(args: ClaudeSDKArgs) -> ClaudeSDKResults:
    """Execute task using Claude Agent SDK

    This uses the official Claude Agent SDK (wraps Claude Code CLI).

    Args:
        args: ClaudeSDKArgs configuration

    Returns:
        ClaudeSDKResults with execution details

    Raises:
        Exception: If SDK execution fails
    """
    logger.info("=================== STARTING CLAUDE AGENT SDK EXECUTION ===================")
    logger.info(f"Workspace dir: {args.workspace_dir}")
    logger.info(f"Topic: {args.topic_type}_{args.topic_idx}_{args.topic_name}")
    logger.info(f"Agent name: {args.agent_name}")

    # Use provided run folder or create new one
    if args.run_folder_path:
        # Use existing run folder from nodes.py (clean datetime name)
        run_dir = Path(args.run_folder_path)
        run_name = run_dir.name  # Just the datetime folder name
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using provided run folder: {run_dir}")
    else:
        # Create new run folder with datetime-only name (legacy behavior)
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = ts  # Just use timestamp as folder name
        run_dir = Path(args.workspace_runs_dir) / run_name
        logs_dir = run_dir / "logs"
        run_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created new run folder: {run_dir}")

    logger.info(f"Run name: {run_name}")
    logger.info(f"Run directory: {run_dir}")

    # Write prompt file for record-keeping
    prompt_file_host = str(run_dir / "prompt_file.txt")
    Path(prompt_file_host).write_text(args.prompt, encoding="utf-8")
    logger.info(f"📝 Prompt saved to: {prompt_file_host}")

    # Create log file path
    log_file = str(logs_dir / "sdk_execution.log")

    start_time = time.time()

    try:
        # Run async execution
        result = asyncio.run(_execute_claude_sdk_async(
            args=args,
            run_dir=run_dir,
            logs_dir=logs_dir,
            run_name=run_name,
            log_file=log_file,
            prompt_file_host=prompt_file_host,
            start_time=start_time
        ))
        return result

    except Exception as e:
        elapsed_time = time.time() - start_time
        error_message = f"Claude SDK execution failed: {type(e).__name__}: {e}"
        logger.error(error_message)

        # Write error log
        import traceback
        error_log = ERROR_LOG_TEMPLATE.format(
            run_name=run_name,
            agent_name=args.agent_name,
            elapsed_time=elapsed_time,
            error_message=error_message,
            traceback=traceback.format_exc()
        )
        Path(log_file).write_text(error_log, encoding="utf-8")

        logger.info("=================== CLAUDE AGENT SDK EXECUTION FAILED ===================")

        # Note: API key restoration is handled by EnsureClaudeSDKAuth context manager

        return ClaudeSDKResults(
            success=False,
            message=error_message,
            run_dir=str(run_dir),
            logs_dir=str(logs_dir),
            log_file=log_file,
            prompt_file_host=prompt_file_host,
            analysis_file=None,
            analysis_text=None,
            config_echo=args.to_dict(),
            log_tail=error_log[-500:] if len(error_log) > 500 else error_log,
            conversation_id=None,
            execution_status="ERROR",
            final_response=None,
            total_cost=None,
            total_tokens=None,
            num_turns=None
        )


async def _execute_claude_sdk_async(
    args: ClaudeSDKArgs,
    run_dir: Path,
    logs_dir: Path,
    run_name: str,
    log_file: str,
    prompt_file_host: str,
    start_time: float
) -> ClaudeSDKResults:
    """Async execution wrapper for Claude Agent SDK"""

    # ===================== Setup SDK Components =====================
    logger.info("🔧 Setting up Claude Agent SDK components...")

    # 1. Determine model (get from YAML config - no fallbacks)
    model = args.llm_model
    if not model:
        from eventglucose.state.llm_config import get_llm_config
        config = get_llm_config(agent="data_agent", node="execute_task")
        model = config.model

    # Model should be SDK-compatible name from YAML config (haiku/sonnet/opus)
    logger.info(f"   Using model: {model}")

    # 2. Authentication
    #    Use SubscriptionAuthContext locally when calling ClaudeSDKClient.
    #    This preserves ANTHROPIC_API_KEY for other components (ChatAnthropic).
    #
    #    NOTE: We no longer call configure_claude_auth() here because it
    #    permanently removes ANTHROPIC_API_KEY, breaking ChatAnthropic.
    #    Instead, SubscriptionAuthContext temporarily removes/restores the key.
    claude_credentials = Path.home() / ".claude"
    if claude_credentials.exists():
        logger.info("   Using subscription auth (~/.claude) - handled by SubscriptionAuthContext")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        logger.info("   Using API key auth (ANTHROPIC_API_KEY)")
    else:
        logger.warning("   ⚠️  No Claude authentication found!")

    # 3. Setup Python Virtual Environment (if available)
    python_venv = os.getenv("PYTHON_VENV")
    python_executable = None
    activation_script_path = None

    if python_venv:
        python_venv_path = Path(python_venv)
        if python_venv_path.exists():
            python_executable = str(python_venv_path / "bin" / "python")

            # Create activation script in workspace
            activation_script_path = Path(args.workspace_dir) / "activate_venv.sh"
            activation_content = f"""#!/bin/bash
# Auto-generated Python virtual environment activation script
# Created by Claude Agent SDK tool for environment consistency

source {python_venv}/bin/activate
export PATH="{python_venv}/bin:$PATH"
export VIRTUAL_ENV="{python_venv}"
export PYTHON_VENV="{python_venv}"

echo "✅ Virtual environment activated: {python_venv}"
echo "✅ Python: $(which python)"
echo "✅ Python version: $(python --version)"
"""
            activation_script_path.write_text(activation_content, encoding="utf-8")
            activation_script_path.chmod(0o755)
            logger.info(f"✅ Created activation script: {activation_script_path}")
            logger.info(f"✅ Python venv: {python_venv}")
            logger.info(f"✅ Python executable: {python_executable}")
        else:
            logger.warning(f"⚠️  PYTHON_VENV path does not exist: {python_venv}")
    else:
        logger.warning("⚠️  PYTHON_VENV not set - agent will use system Python")

    # 4. Enhance prompt with Python venv instructions (if available)
    enhanced_prompt = args.prompt

    if python_executable and activation_script_path:
        venv_instructions = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ CRITICAL: Python Virtual Environment Configuration                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

BEFORE running ANY Python commands, you MUST use the correct Python environment:

Method 1 (Recommended): Use full Python path
    {python_executable} your_script.py
    {python_executable} -m pip install package_name

Method 2: Activate environment first
    source {activation_script_path}
    python your_script.py

⚠️  NEVER use just 'python' or 'python3' - always use the full path!
⚠️  The correct Python is: {python_executable}

You can verify the Python environment anytime with:
    {python_executable} --version
    {python_executable} -c "import sys; print(sys.executable)"

╔══════════════════════════════════════════════════════════════════════════════╗
║ Original Task                                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

"""
        enhanced_prompt = venv_instructions + args.prompt
        logger.info("✅ Enhanced prompt with Python venv instructions")

    # 5. Configure Claude Agent SDK options
    options = ClaudeAgentOptions(
        cwd=args.workspace_dir,  # Set working directory to workspace
        allowed_tools=args.allowed_tools,
        permission_mode=args.permission_mode,
        max_turns=args.max_turns,
        model=model,
        # Load project settings to include CLAUDE.md (if it exists)
        setting_sources=["project"]
    )

    logger.info(f"✅ Claude Agent SDK configured")
    logger.info(f"   - Model: {model}")
    logger.info(f"   - Workspace: {args.workspace_dir}")
    logger.info(f"   - Max turns: {args.max_turns}")
    logger.info(f"   - Allowed tools: {', '.join(args.allowed_tools)}")

    # ===================== Execute Task =====================
    logger.info(f"🚀 Starting Claude Agent SDK execution...")
    logger.info(f"⏳ This may take a while...")

    # Track all messages for logging
    all_messages = []
    final_response_text = None
    result_info = None

    # EnsureClaudeSDKAuth handles auth:
    # - If ~/.claude exists: SDK uses subscription (preferred)
    # - If not: temporarily restore API key to env
    from ._auth import EnsureClaudeSDKAuth

    with EnsureClaudeSDKAuth():
        async with ClaudeSDKClient(options=options) as client:
            # Send prompt
            await client.query(enhanced_prompt)

            # Collect all messages and show real-time progress
            # Import shared helper for consistent output formatting
            from ._sdk_streaming import print_sdk_message

            async for message in client.receive_response():
                all_messages.append(message)

                # Use shared helper for consistent output formatting
                response_text = print_sdk_message(message, verbose=True)
                if response_text:
                    final_response_text = response_text

                # Extract result information
                if isinstance(message, ResultMessage):
                    result_info = message

    elapsed_time = time.time() - start_time
    logger.info(f"✅ Claude Agent SDK execution completed in {elapsed_time:.2f} seconds")

    # ===================== Process Results =====================

    # Determine success
    success = False
    execution_status = "UNKNOWN"
    total_cost = None
    num_turns = None
    session_id = None

    if result_info:
        success = not result_info.is_error
        execution_status = "FINISHED" if success else "ERROR"
        total_cost = result_info.total_cost_usd
        num_turns = result_info.num_turns
        session_id = result_info.session_id

        logger.info(f"📊 Execution Status: {execution_status}")
        logger.info(f"💰 Total cost: ${total_cost:.4f}" if total_cost else "💰 Total cost: N/A")
        logger.info(f"🔢 Turns: {num_turns}" if num_turns else "🔢 Turns: N/A")

    # Write execution log
    log_content = LOG_CONTENT_TEMPLATE.format(
        run_name=run_name,
        agent_name=args.agent_name,
        model=model,
        max_turns=args.max_turns,
        elapsed_time=elapsed_time,
        execution_status=execution_status,
        success=success,
        final_response=final_response_text or 'No final response available',
        total_cost=f"{total_cost:.4f}" if total_cost else "N/A",
        num_turns=num_turns or "N/A"
    )
    Path(log_file).write_text(log_content, encoding="utf-8")

    # Save complete conversation history
    if all_messages:
        logger.info(f"📝 Saving {len(all_messages)} conversation messages...")

        # 1. Save human-readable transcript
        transcript_file = logs_dir / "conversation_transcript.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write("Claude Agent SDK - Complete Conversation Transcript\n")
            f.write("=" * 80 + "\n")
            f.write(f"Run: {run_name}\n")
            f.write(f"Agent: {args.agent_name}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Total Messages: {len(all_messages)}\n")
            f.write(f"Elapsed Time: {elapsed_time:.2f}s\n")
            f.write("=" * 80 + "\n\n")

            for i, message in enumerate(all_messages, 1):
                try:
                    f.write(f"\n{'─' * 80}\n")
                    f.write(f"Message #{i}: {type(message).__name__}\n")
                    f.write(f"{'─' * 80}\n")

                    if isinstance(message, AssistantMessage):
                        f.write(f"Model: {message.model}\n")
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                f.write(f"Text: {block.text}\n")
                            elif isinstance(block, ThinkingBlock):
                                f.write(f"Thinking: {block.thinking}\n")
                            elif isinstance(block, ToolUseBlock):
                                f.write(f"Tool: {block.name}\n")
                                f.write(f"Input: {json.dumps(block.input, indent=2)}\n")
                            elif isinstance(block, ToolResultBlock):
                                f.write(f"Tool Result (ID: {block.tool_use_id}):\n")
                                f.write(f"Content: {block.content}\n")
                                if block.is_error:
                                    f.write(f"⚠️ Error: {block.is_error}\n")
                    elif isinstance(message, ResultMessage):
                        f.write(f"Subtype: {message.subtype}\n")
                        f.write(f"Duration: {message.duration_ms}ms\n")
                        f.write(f"Turns: {message.num_turns}\n")
                        f.write(f"Cost: ${message.total_cost_usd:.4f}\n" if message.total_cost_usd else "")
                        f.write(f"Session ID: {message.session_id}\n")
                    else:
                        f.write(f"{message}\n")

                except Exception as e:
                    f.write(f"[Could not serialize message: {e}]\n")

        logger.info(f"💾 Saved conversation transcript: {transcript_file}")

        # 2. Save messages as JSON
        messages_json_file = logs_dir / "conversation_messages.json"
        try:
            messages_data = {
                "run_name": run_name,
                "agent_name": args.agent_name,
                "model": model,
                "total_messages": len(all_messages),
                "elapsed_time": elapsed_time,
                "messages": []
            }

            for message in all_messages:
                try:
                    # Convert message to dict (if possible)
                    if hasattr(message, '__dict__'):
                        msg_dict = {"type": type(message).__name__}
                        for key, value in message.__dict__.items():
                            if not key.startswith('_'):
                                try:
                                    json.dumps(value)  # Test if serializable
                                    msg_dict[key] = value
                                except:
                                    msg_dict[key] = str(value)
                        messages_data["messages"].append(msg_dict)
                    else:
                        messages_data["messages"].append({
                            "type": type(message).__name__,
                            "content": str(message)
                        })
                except Exception as e:
                    messages_data["messages"].append({
                        "type": type(message).__name__,
                        "error": f"Could not serialize: {e}"
                    })

            with open(messages_json_file, 'w', encoding='utf-8') as f:
                json.dump(messages_data, f, indent=2, default=str)

            logger.info(f"💾 Saved conversation messages JSON: {messages_json_file}")

        except Exception as e:
            logger.warning(f"Could not save messages JSON: {e}")

    else:
        logger.warning("⚠️ No messages captured during conversation")

    # ===================== Probe for Output Files =====================

    analysis_text: str | None = None
    analysis_file: str | None = None

    # Probe for expected output files in reports folder
    probe_files = args.output_probe_files or [
        "report.md",
        "analysis_report.md",
        "data_analysis.txt",
        f"{args.topic_type}_report.md"
    ]

    for probe_name in probe_files:
        candidate = Path(args.workspace_reports_dir) / probe_name
        if candidate.exists():
            analysis_file = str(candidate)
            analysis_text = candidate.read_text(encoding="utf-8")
            logger.info(f"✅ Found analysis file: {analysis_file}")
            break

    if not analysis_text and final_response_text:
        # Use final response as analysis if no file found
        analysis_text = final_response_text
        logger.info(f"ℹ️  Using final response as analysis (no probe files found)")

    # Get log tail (last 10 lines)
    log_tail = None
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            log_tail = ''.join(lines[-10:]) if len(lines) >= 10 else ''.join(lines)
    except Exception as e:
        log_tail = f"Error reading log: {e}"

    # Prepare result message
    message = f"Claude SDK execution {execution_status}"
    if success:
        message += f" - Task completed in {elapsed_time:.2f}s"
        if total_cost:
            message += f" (cost: ${total_cost:.4f})"
    else:
        message += f" - Check logs for details"

    logger.info(f"✅ Claude Agent SDK execution completed")
    logger.info("=================== CLAUDE AGENT SDK EXECUTION COMPLETED ===================")

    # Note: API key restoration is handled by EnsureClaudeSDKAuth context manager

    return ClaudeSDKResults(
        success=success,
        message=message,
        run_dir=str(run_dir),
        logs_dir=str(logs_dir),
        log_file=log_file,
        prompt_file_host=prompt_file_host,
        analysis_file=analysis_file,
        analysis_text=analysis_text,
        config_echo=args.to_dict(),
        log_tail=log_tail,
        conversation_id=session_id,
        execution_status=execution_status,
        final_response=final_response_text,
        total_cost=total_cost,
        total_tokens=None,  # Not available in Claude Agent SDK
        num_turns=num_turns
    )


# Test prompt
PROMPT = '''
Please create a simple Python script:
1. Save it to code/hello.py
2. The script should print "Hello from Claude Agent SDK!"
3. Create a report at reports/hello_report.md with:
   - Confirmation that the script was created
   - The script contents
   - Instructions on how to run it
'''


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger.info("=================== STARTING CLAUDE SDK TEST ===================")

    # Create test arguments
    args = ClaudeSDKArgs(
        prompt=PROMPT,
        topic_type="test",
        topic_idx=1,
        topic_name="claude_sdk_hello",
        workspace_dir="projspace/proj_claude_sdk_hello",
        max_turns=10
    )

    logger.info(f"Args: {json.dumps(args.to_dict(), indent=4)}")

    # Execute with Claude Agent SDK
    result = claude_sdk_tool(args)

    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS:")
    logger.info(f"{'='*80}")
    logger.info(f"Success: {result.success}")
    logger.info(f"Message: {result.message}")
    logger.info(f"Status: {result.execution_status}")
    logger.info(f"Session ID: {result.conversation_id}")
    logger.info(f"Run dir: {result.run_dir}")
    logger.info(f"Log file: {result.log_file}")

    if result.total_cost:
        logger.info(f"Cost: ${result.total_cost:.4f}")
    if result.num_turns:
        logger.info(f"Turns: {result.num_turns}")

    if result.analysis_file:
        logger.info(f"Analysis file: {result.analysis_file}")

    if result.final_response:
        logger.info(f"\nFinal Response Preview:")
        logger.info(f"{result.final_response[:300]}...")

    logger.info(f"{'='*80}")

    if result.success:
        logger.info("✅ Claude SDK test completed successfully!")
    else:
        logger.error("❌ Claude SDK test failed - check logs")


"""
Usage Examples:

# Basic usage
python code/haiagent/dikwgraph/tools/tools_ccsdk.py

# Or import and use in your code:
from eventglucose.tools.tools_ccsdk import ClaudeSDKArgs, claude_sdk_tool

args = ClaudeSDKArgs(
    prompt="Your detailed task...",
    topic_type="data",
    topic_idx=1,
    topic_name="analysis",
    workspace_dir="projspace/proj_analysis"
)

result = claude_sdk_tool(args)
print(f"Success: {result.success}")
print(f"Cost: ${result.total_cost}")
print(result.final_response)
"""
