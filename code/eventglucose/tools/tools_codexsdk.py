"""
Codex SDK Tool - OpenAI Codex CLI Integration
==============================================

Module Structure
----------------

┌─────────────────────────────────────────────────────────────────────┐
│                       tools_codexsdk.py                             │
│               Codex CLI SDK Integration                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MAIN FUNCTION                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  run_with_codex_sdk(task, metadata, verbose)               │    │
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
│  │  │ 3. Execute via Codex SDK                            │  │    │
│  │  │    └─> codex_client.run(prompt, workspace_dir)      │  │    │
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

Authentication
--------------

Uses codex-client authentication (managed by codex CLI):
1. Run `codex-client login` to authenticate
2. Credentials stored via keyring
3. No API key required in code

Usage
-----

::

    from eventglucose.tools.tools_codexsdk import CodexSDKArgs, codex_sdk_tool

    args = CodexSDKArgs(
        prompt="Your task description",
        topic_type="data",
        topic_idx=1,
        topic_name="analysis_task",
        workspace_dir="projspace/proj_analysis"
    )
    result = codex_sdk_tool(args)
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

# Codex client imports
from codex_client import (
    Client,
    CodexChatConfig,
    CodexProfile,
    ReasoningEffort,
    SandboxMode,
    ApprovalPolicy,
    Verbosity,
    AssistantMessageStream,
    CommandStream,
    ReasoningStream,
    SessionConfiguredEvent,
    TaskCompleteEvent,
    TaskStartedEvent,
    TokenCountEvent,
    CodexError,
)
from codex_client.auth import CodexAuth

# Configure logging
logger = logging.getLogger(__name__)


# Log content templates
LOG_CONTENT_TEMPLATE = """Codex SDK Execution Log
========================

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
- Tokens: {total_tokens}

"""

ERROR_LOG_TEMPLATE = """Codex SDK Execution Error
==========================

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
class CodexSDKArgs:
    """Arguments for Codex SDK execution

    API-compatible with ClaudeSDKArgs for easy drop-in replacement.
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
    max_turns: int = 30  # Maximum conversation turns

    # LLM configuration
    llm_model: str | None = None            # Default: "gpt-5" (Codex default)
    llm_api_key: str | None = None          # Optional: managed by codex-client

    # Agent configuration
    agent_name: str | None = None           # Default: f"{topic_type}_{topic_name}_agent"

    # Codex-specific options
    codex_model: str = "gpt-5"              # Model: gpt-5, claude-opus-4-5, etc.
    reasoning_effort: str = "minimal"       # minimal, low, medium, high
    sandbox_mode: str = "workspace-write"   # read-only, workspace-write, danger-full-access
    approval_policy: str = "never"          # untrusted, on-failure, on-request, never
    verbosity: str = "medium"               # low, medium, high

    # Tool permissions (not directly used by Codex, but kept for compatibility)
    allowed_tools: list[str] | None = None  # For compatibility with other SDKs
    permission_mode: str = "acceptEdits"    # For compatibility

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

        # Set default allowed tools if not provided (for compatibility)
        if not self.allowed_tools:
            self.allowed_tools = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class CodexSDKResults:
    """Results from Codex SDK execution

    API-compatible with ClaudeSDKResults for easy drop-in replacement.
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
    total_cost: float | None = None         # Total LLM cost in USD (estimated)
    total_tokens: int | None = None         # Total tokens used
    num_turns: int | None = None            # Number of conversation turns


def codex_sdk_tool(args: CodexSDKArgs) -> CodexSDKResults:
    """Execute task using Codex SDK

    This uses the codex-client wrapper around OpenAI's Codex CLI.

    Args:
        args: CodexSDKArgs configuration

    Returns:
        CodexSDKResults with execution details

    Raises:
        Exception: If SDK execution fails
    """
    logger.info("=================== STARTING CODEX SDK EXECUTION ===================")
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
        result = asyncio.run(_execute_codex_sdk_async(
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
        error_message = f"Codex SDK execution failed: {type(e).__name__}: {e}"

        # Add helpful message for connection errors
        if "Failed to connect" in str(e) or "Connection closed" in str(e):
            logger.error(error_message)
            logger.error("")
            logger.error("💡 TROUBLESHOOTING:")
            logger.error("   1. Ensure 'codex' CLI is installed and on PATH")
            logger.error("   2. Check authentication: codex-client login")
            logger.error("   3. Test codex CLI: codex mcp --help")
            logger.error("")
        else:
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

        logger.info("=================== CODEX SDK EXECUTION FAILED ===================")

        return CodexSDKResults(
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


async def _execute_codex_sdk_async(
    args: CodexSDKArgs,
    run_dir: Path,
    logs_dir: Path,
    run_name: str,
    log_file: str,
    prompt_file_host: str,
    start_time: float
) -> CodexSDKResults:
    """Async execution wrapper for Codex SDK"""

    # ===================== Setup SDK Components =====================
    logger.info("🔧 Setting up Codex SDK components...")

    # 1. Determine model
    model = args.llm_model or args.codex_model
    if not model:
        from eventglucose.state.llm_config import get_llm_config
        config = get_llm_config(agent="data_agent", node="execute_task")
        model = config.model

    logger.info(f"   Using model: {model}")

    # 2. Authentication check
    auth = CodexAuth()
    try:
        token = auth.read()
        if token:
            logger.info("   ✅ Codex authentication configured")
        else:
            logger.warning("   ⚠️  No Codex authentication found! Run: codex-client login")
    except Exception as e:
        logger.warning(f"   ⚠️  Could not verify Codex auth: {e}")

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
# Created by Codex SDK tool for environment consistency

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

    # 5. Configure Codex profile
    profile = CodexProfile(
        model=model,
        reasoning_effort=ReasoningEffort(args.reasoning_effort),
        sandbox=SandboxMode(args.sandbox_mode),
        verbosity=Verbosity(args.verbosity),
    )

    config = CodexChatConfig(
        profile=profile,
        approval_policy=ApprovalPolicy(args.approval_policy),
        cwd=args.workspace_dir,
        sandbox=SandboxMode(args.sandbox_mode),  # Set sandbox at config level too
    )

    logger.info(f"✅ Codex SDK configured")
    logger.info(f"   - Model: {model}")
    logger.info(f"   - Workspace: {args.workspace_dir}")
    logger.info(f"   - Max turns: {args.max_turns}")
    logger.info(f"   - Sandbox: {args.sandbox_mode}")
    logger.info(f"   - Reasoning: {args.reasoning_effort}")

    # ===================== Execute Task =====================
    logger.info(f"🚀 Starting Codex SDK execution...")
    logger.info(f"⏳ This may take a while...")

    # Track execution state
    all_events = []
    final_response_text = None
    total_tokens = 0
    num_turns = 0
    session_configured = False

    # Change to workspace directory for execution
    original_cwd = os.getcwd()
    os.chdir(args.workspace_dir)
    logger.info(f"📂 Changed working directory to: {args.workspace_dir}")

    try:
        # Note: Client defaults to args=["mcp-server"] but Codex CLI uses "mcp" subcommand
        # Override to use correct command: "codex mcp"
        async with Client(args=["mcp"]) as client:
            # Create chat
            chat = await client.create_chat(enhanced_prompt, config=config)

            # Stream events
            async for event in chat:
                all_events.append(event)

                # Handle different event types
                if isinstance(event, SessionConfiguredEvent):
                    session_configured = True
                    logger.info(f"🔧 Session configured with model '{event.model}'")
                    if event.reasoning_effort:
                        logger.info(f"   Reasoning effort: {event.reasoning_effort.value}")

                elif isinstance(event, TaskStartedEvent):
                    context_size = event.model_context_window or "unknown"
                    logger.info(f"🚀 Task started (context: {context_size})")

                elif isinstance(event, AssistantMessageStream):
                    logger.info("🤖 Assistant response:")
                    chunks = []
                    async for chunk in event.stream():
                        chunks.append(chunk)
                        print(chunk, end="", flush=True)
                    final_response_text = "".join(chunks)
                    print()  # Newline after response
                    num_turns += 1

                elif isinstance(event, ReasoningStream):
                    logger.info("🧐 Reasoning:")
                    async for chunk in event.stream():
                        logger.info(f"   {chunk}")

                elif isinstance(event, CommandStream):
                    command_str = " ".join(event.command)
                    logger.info(f"⚡ Command: {command_str}")
                    async for chunk in event.stream():
                        if chunk.text is not None:
                            print(chunk.text, end="", flush=True)
                    if event.exit_code is not None:
                        status = "✅" if event.exit_code == 0 else "❌"
                        logger.info(f"{status} Exit code: {event.exit_code}")

                elif isinstance(event, TokenCountEvent):
                    if event.info and event.info.total_token_usage:
                        total_tokens = event.info.total_token_usage.total_tokens
                        delta = event.info.last_token_usage.total_tokens if event.info.last_token_usage else 0
                        logger.info(f"💰 Tokens: {total_tokens:,} total (+{delta:,})")

                elif isinstance(event, TaskCompleteEvent):
                    logger.info("🎉 Task complete")
                    session_configured = True  # Mark as successful completion

            # Get final message
            final_message = await chat.get()
            if final_message and not final_response_text:
                final_response_text = str(final_message)

    finally:
        # Restore original working directory
        os.chdir(original_cwd)
        logger.info(f"📂 Restored working directory to: {original_cwd}")

    elapsed_time = time.time() - start_time
    logger.info(f"✅ Codex SDK execution completed in {elapsed_time:.2f} seconds")

    # ===================== Process Results =====================

    # Determine success
    success = session_configured and (final_response_text is not None)
    execution_status = "FINISHED" if success else "ERROR"

    # Estimate cost (rough approximation - Codex doesn't provide exact costs)
    total_cost = None
    if total_tokens > 0:
        # Rough estimate: $0.01 per 1K tokens (adjust based on actual model pricing)
        total_cost = (total_tokens / 1000.0) * 0.01

    logger.info(f"📊 Execution Status: {execution_status}")
    logger.info(f"💰 Total cost: ${total_cost:.4f}" if total_cost else "💰 Total cost: N/A")
    logger.info(f"🔢 Tokens: {total_tokens:,}" if total_tokens else "🔢 Tokens: N/A")
    logger.info(f"🔢 Turns: {num_turns}")

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
        num_turns=num_turns,
        total_tokens=f"{total_tokens:,}" if total_tokens else "N/A"
    )
    Path(log_file).write_text(log_content, encoding="utf-8")

    # Save complete event history
    if all_events:
        logger.info(f"📝 Saving {len(all_events)} events...")

        # 1. Save human-readable transcript
        transcript_file = logs_dir / "conversation_transcript.txt"
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write("Codex SDK - Complete Conversation Transcript\n")
            f.write("=" * 80 + "\n")
            f.write(f"Run: {run_name}\n")
            f.write(f"Agent: {args.agent_name}\n")
            f.write(f"Model: {model}\n")
            f.write(f"Total Events: {len(all_events)}\n")
            f.write(f"Elapsed Time: {elapsed_time:.2f}s\n")
            f.write("=" * 80 + "\n\n")

            for i, event in enumerate(all_events, 1):
                try:
                    f.write(f"\n{'─' * 80}\n")
                    f.write(f"Event #{i}: {type(event).__name__}\n")
                    f.write(f"{'─' * 80}\n")
                    f.write(f"{event}\n")
                except Exception as e:
                    f.write(f"[Could not serialize event: {e}]\n")

        logger.info(f"💾 Saved conversation transcript: {transcript_file}")

        # 2. Save events as JSON
        events_json_file = logs_dir / "conversation_events.json"
        try:
            events_data = {
                "run_name": run_name,
                "agent_name": args.agent_name,
                "model": model,
                "total_events": len(all_events),
                "elapsed_time": elapsed_time,
                "events": []
            }

            for event in all_events:
                try:
                    event_dict = {"type": type(event).__name__}
                    if hasattr(event, '__dict__'):
                        for key, value in event.__dict__.items():
                            if not key.startswith('_'):
                                try:
                                    json.dumps(value)
                                    event_dict[key] = value
                                except:
                                    event_dict[key] = str(value)
                    events_data["events"].append(event_dict)
                except Exception as e:
                    events_data["events"].append({
                        "type": type(event).__name__,
                        "error": f"Could not serialize: {e}"
                    })

            with open(events_json_file, 'w', encoding='utf-8') as f:
                json.dump(events_data, f, indent=2, default=str)

            logger.info(f"💾 Saved conversation events JSON: {events_json_file}")

        except Exception as e:
            logger.warning(f"Could not save events JSON: {e}")

    else:
        logger.warning("⚠️ No events captured during conversation")

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
    message = f"Codex SDK execution {execution_status}"
    if success:
        message += f" - Task completed in {elapsed_time:.2f}s"
        if total_cost:
            message += f" (cost: ${total_cost:.4f})"
    else:
        message += f" - Check logs for details"

    logger.info(f"✅ Codex SDK execution completed")
    logger.info("=================== CODEX SDK EXECUTION COMPLETED ===================")

    return CodexSDKResults(
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
        conversation_id=None,  # Codex doesn't provide session IDs in the same way
        execution_status=execution_status,
        final_response=final_response_text,
        total_cost=total_cost,
        total_tokens=total_tokens,
        num_turns=num_turns
    )


# Test prompt
PROMPT = '''
Please create a simple Python script:
1. Save it to code/hello.py
2. The script should print "Hello from Codex SDK!"
3. Create a report at report/hello_report.md with:
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

    logger.info("=================== STARTING CODEX SDK TEST ===================")
    logger.info("")
    logger.info("⚠️  REQUIREMENTS:")
    logger.info("   1. Codex CLI installed and on PATH")
    logger.info("   2. Authenticated with: codex-client login")
    logger.info("")
    logger.info("   The Client will automatically start the Codex MCP process.")
    logger.info("")

    # Create test arguments
    args = CodexSDKArgs(
        prompt=PROMPT,
        topic_type="test",
        topic_idx=1,
        topic_name="codex_sdk_hello",
        workspace_dir="projspace/proj_codex_sdk_hello",
        max_turns=10,
        codex_model="gpt-5",
        reasoning_effort="minimal",
        sandbox_mode="workspace-write"
    )

    logger.info(f"Args: {json.dumps(args.to_dict(), indent=4)}")

    # Execute with Codex SDK
    result = codex_sdk_tool(args)

    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS:")
    logger.info(f"{'='*80}")
    logger.info(f"Success: {result.success}")
    logger.info(f"Message: {result.message}")
    logger.info(f"Status: {result.execution_status}")
    logger.info(f"Run dir: {result.run_dir}")
    logger.info(f"Log file: {result.log_file}")

    if result.total_cost:
        logger.info(f"Cost: ${result.total_cost:.4f}")
    if result.total_tokens:
        logger.info(f"Tokens: {result.total_tokens:,}")
    if result.num_turns:
        logger.info(f"Turns: {result.num_turns}")

    if result.analysis_file:
        logger.info(f"Analysis file: {result.analysis_file}")

    if result.final_response:
        logger.info(f"\nFinal Response Preview:")
        logger.info(f"{result.final_response[:300]}...")

    logger.info(f"{'='*80}")

    if result.success:
        logger.info("✅ Codex SDK test completed successfully!")
        logger.info("")
        logger.info("Check the workspace for generated files:")
        logger.info(f"  Code: {result.run_dir}/../code/")
        logger.info(f"  Report: {result.run_dir}/../report/")
    else:
        logger.error("❌ Codex SDK test failed - check logs")
        if "Failed to connect" in result.message or "Connection" in result.message:
            logger.error("")
            logger.error("Possible causes:")
            logger.error("  1. Codex CLI not installed - install from https://docs.codex.com")
            logger.error("  2. Codex CLI not authenticated - run: codex-client login")
            logger.error("  3. Codex CLI not on PATH - run: which codex")
            logger.error("")
            logger.error("Verify installation: codex mcp --help")


"""
Usage Examples:

# Basic usage
python code/eventglucose/tools/tools_codexsdk.py

# Or import and use in your code:
from eventglucose.tools.tools_codexsdk import CodexSDKArgs, codex_sdk_tool

args = CodexSDKArgs(
    prompt="Your detailed task...",
    topic_type="data",
    topic_idx=1,
    topic_name="analysis",
    workspace_dir="projspace/proj_analysis",
    codex_model="gpt-5",
    sandbox_mode="workspace-write"
)

result = codex_sdk_tool(args)
print(f"Success: {result.success}")
print(f"Cost: ${result.total_cost}")
print(result.final_response)
"""
