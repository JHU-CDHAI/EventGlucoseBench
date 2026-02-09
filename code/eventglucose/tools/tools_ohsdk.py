"""
OpenHands SDK Tool - Official SDK Implementation
================================================

Module Structure
----------------

┌─────────────────────────────────────────────────────────────────────┐
│                       tools_ohsdk.py                                │
│               OpenHands SDK Integration                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  MAIN FUNCTION                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  openhand_sdk_tool(args: OpenHandSDKArgs) -> dict          │    │
│  │    │                                                       │    │
│  │    ▼                                                       │    │
│  │  Flow:                                                     │    │
│  │  ┌─────────────────────────────────────────────────────┐  │    │
│  │  │ 1. Set up workspace and tools                       │  │    │
│  │  │ 2. Create LLM + Agent + Conversation                │  │    │
│  │  │ 3. Execute via agent.run(prompt)                    │  │    │
│  │  │ 4. Parse outputs (code, reports)                    │  │    │
│  │  │ 5. Return result dict with paths                    │  │    │
│  │  └─────────────────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  SDK COMPONENTS                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  LLM ──────────> Language model interface                  │    │
│  │  Agent ────────> Task execution agent                      │    │
│  │  Conversation ─> Stateful conversation manager             │    │
│  │  Tools ────────> Terminal, FileEditor, TaskTracker         │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  OpenHandSDKArgs DATACLASS                                          │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  prompt ─────────> Task description                        │    │
│  │  topic_type ─────> "data" | "information" | ...            │    │
│  │  topic_idx ──────> Task index                              │    │
│  │  topic_name ─────> Task name                               │    │
│  │  workspace_dir ──> Execution workspace path                │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Based on: https://github.com/openhands/software-agent-sdk
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

# OpenHands SDK imports
from openhands.sdk import LLM, Agent, Conversation, Event, LLMConvertibleEvent
from openhands.sdk.tool import Tool
from openhands.tools.terminal import TerminalTool
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool

# Configure logging
logger = logging.getLogger(__name__)


# Log content templates
LOG_CONTENT_TEMPLATE = """OpenHands SDK Execution Log
============================

Run: {run_name}
Agent: {agent_name}
Model: {model}
Max Iterations: {max_iterations}
Elapsed Time: {elapsed_time:.2f} seconds

Status: {execution_status}
Success: {success}

Final Response:
{final_response}

Metrics:
- Cost: ${total_cost}

"""

ERROR_LOG_TEMPLATE = """OpenHands SDK Execution Error
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
class OpenHandSDKArgs:
    """Arguments for OpenHands SDK execution

    Simplified API focused on SDK features, not Docker internals.
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
    max_iterations: int = 30

    # LLM configuration
    llm_model: str | None = None            # Default: uses $MODEL env var
    llm_api_key: str | None = None          # Default: uses appropriate API key from env

    # Agent configuration
    agent_name: str | None = None           # Default: f"{topic_type}_{topic_name}_agent"

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

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class OpenHandSDKResults:
    """Results from OpenHands SDK execution"""
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
    conversation_id: str | None = None      # Conversation ID from SDK
    execution_status: str | None = None     # Final execution status
    final_response: str | None = None       # Agent's final response
    total_cost: float | None = None         # Total LLM cost
    total_tokens: int | None = None         # Total tokens used


def openhand_sdk_tool(args: OpenHandSDKArgs) -> OpenHandSDKResults:
    """Execute OpenHands task using official SDK

    This uses the official OpenHands SDK API (v1.1.0) for clean, Python-native execution.

    Args:
        args: OpenHandSDKArgs configuration

    Returns:
        OpenHandSDKResults with execution details

    Raises:
        Exception: If SDK execution fails
    """

    logger.info("=================== STARTING OPENHANDS SDK EXECUTION ===================")
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
    logger.info(f"Saved prompt to: {prompt_file_host}")

    # Create log file path
    log_file = str(logs_dir / "sdk_execution.log")

    start_time = time.time()

    try:
        # ===================== Setup SDK Components =====================
        logger.info("🔧 Setting up OpenHands SDK components...")

        # 1. Create LLM (get model from YAML config - no fallbacks)
        model = args.llm_model
        if not model:
            from eventglucose.state.llm_config import get_llm_config
            config = get_llm_config(agent="data_agent", node="execute_task")
            model = config.model

        # Auto-add "anthropic/" prefix if not present (for LiteLLM compatibility)
        if not model.startswith(("anthropic/", "openai/", "cohere/", "azure/")):
            logger.info(f"   Adding 'anthropic/' prefix to model: {model}")
            model = f"anthropic/{model}"

        # Get API key (auto-detect from model name) - use shared auth utility
        from ._auth import get_api_key_for_model

        api_key = args.llm_api_key or get_api_key_for_model(model)

        if not api_key:
            raise ValueError(f"No API key provided for model {model}")

        # NOTE: Use direct model names (e.g., "anthropic/claude-sonnet-4-20250514")
        # Do NOT add 'openhands/' prefix - that routes through their proxy server
        # For direct API access, LiteLLM will handle the provider routing

        llm = LLM(model=model, api_key=api_key)
        logger.info(f"✅ LLM: {model}")

        # 2. Create Agent with default tools
        # NOTE: Disable llm_security_analyzer to avoid SDK bug with undefined *WithRisk action classes
        # See: https://github.com/All-Hands-AI/OpenHands/issues (PydanticUndefinedAnnotation errors)
        agent = Agent(
            llm=llm,
            tools=[
                Tool(name=TerminalTool.name),      # Bash execution
                Tool(name=FileEditorTool.name),    # File operations
                Tool(name=TaskTrackerTool.name),   # Task management
            ],
            system_prompt_kwargs={
                'llm_security_analyzer': False  # Disable to avoid FinishActionWithRisk/TaskTrackerActionWithRisk errors
            }
        )
        logger.info(f"✅ Agent: {args.agent_name} with 3 tools")

        # 3. Setup event capture for full conversation logging
        all_events = []
        llm_messages = []

        def event_callback(event: Event):
            """Capture all events from the conversation"""
            all_events.append(event)
            if isinstance(event, LLMConvertibleEvent):
                llm_messages.append(event.to_llm_message())
                logger.debug(f"Event: {type(event).__name__}")

        # 4. Setup Python Virtual Environment (Solution 4)
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
# Created by OpenHands SDK tool for environment consistency

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

        # 5. Create Conversation with event callback
        conversation = Conversation(
            agent=agent,
            workspace=args.workspace_dir,  # SDK takes path string, not object!
            callbacks=[event_callback],    # Capture all events!
        )
        logger.info(f"✅ Conversation initialized in workspace: {args.workspace_dir}")
        logger.info(f"✅ Event callback attached for full conversation logging")

        # ===================== Execute Task =====================
        logger.info(f"🚀 Sending message to agent (max_iterations={args.max_iterations})")
        logger.info(f"⏳ This may take a while...")

        # 6. Enhance prompt with Python venv instructions (Solution 1)
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

        # Send message to conversation
        conversation.send_message(enhanced_prompt)

        # Run the conversation
        conversation.run()

        elapsed_time = time.time() - start_time
        logger.info(f"✅ SDK execution completed in {elapsed_time:.2f} seconds")

        # ===================== Process Results =====================

        # Access conversation state
        state = conversation.state
        execution_status = str(state.execution_status) if hasattr(state, 'execution_status') else "UNKNOWN"

        logger.info(f"📊 Execution Status: {execution_status}")

        # Check if conversation has final response
        final_response = None
        if hasattr(state, 'events') and state.events:
            # Get agent's final response from events (last agent message)
            for event in reversed(state.events):
                if hasattr(event, 'source') and event.source == 'agent':
                    if hasattr(event, 'message'):
                        final_response = event.message
                        break

        if final_response:
            logger.info(f"💬 Final response length: {len(final_response)} characters")

        # Get metrics if available
        total_cost = None
        total_tokens = None
        if hasattr(conversation, 'conversation_stats'):
            stats = conversation.conversation_stats.get_combined_metrics()
            total_cost = stats.accumulated_cost if hasattr(stats, 'accumulated_cost') else None
            # Note: accumulated_tokens not available in Metrics object
            # Token info is in individual message stats, not combined metrics
            logger.info(f"💰 Total cost: ${total_cost:.4f}" if total_cost else "💰 Total cost: N/A")

        # Determine success (SDK doesn't have explicit success flag)
        success = (
            execution_status == "FINISHED" or
            execution_status == "ConversationExecutionStatus.FINISHED"
        )

        # Write execution log
        log_content = LOG_CONTENT_TEMPLATE.format(
            run_name=run_name,
            agent_name=args.agent_name,
            model=model,
            max_iterations=args.max_iterations,
            elapsed_time=elapsed_time,
            execution_status=execution_status,
            success=success,
            final_response=final_response or 'No final response available',
            total_cost=f"{total_cost:.4f}" if total_cost else "N/A"
        )
        Path(log_file).write_text(log_content, encoding="utf-8")

        # Save complete conversation history (captured via event callbacks)
        if all_events:
            logger.info(f"📝 Saving {len(all_events)} conversation events...")

            # 1. Save human-readable transcript
            transcript_file = logs_dir / "conversation_transcript.txt"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write("OpenHands SDK - Complete Conversation Transcript\n")
                f.write("=" * 80 + "\n")
                f.write(f"Run: {run_name}\n")
                f.write(f"Agent: {args.agent_name}\n")
                f.write(f"Model: {model}\n")
                f.write(f"Total Events: {len(all_events)}\n")
                f.write(f"LLM Messages: {len(llm_messages)}\n")
                f.write(f"Elapsed Time: {elapsed_time:.2f}s\n")
                f.write("=" * 80 + "\n\n")

                for i, event in enumerate(all_events, 1):
                    try:
                        f.write(f"\n{'─' * 80}\n")
                        f.write(f"Event #{i}: {type(event).__name__}\n")
                        f.write(f"{'─' * 80}\n")

                        # Extract all event information
                        if hasattr(event, 'model_dump'):
                            event_data = event.model_dump()
                            for key, value in event_data.items():
                                if value is not None:
                                    f.write(f"{key}: {value}\n")
                        elif hasattr(event, '__dict__'):
                            for key, value in event.__dict__.items():
                                if not key.startswith('_') and value is not None:
                                    # Truncate long values
                                    value_str = str(value)
                                    if len(value_str) > 500:
                                        value_str = value_str[:500] + "... (truncated)"
                                    f.write(f"{key}: {value_str}\n")
                        else:
                            f.write(f"{event}\n")

                    except Exception as e:
                        f.write(f"[Could not serialize event: {e}]\n")

            logger.info(f"💾 Saved conversation transcript ({len(all_events)} events): {transcript_file}")

            # 2. Save complete events as JSON
            events_json_file = logs_dir / "conversation_events.json"
            try:
                events_data = {
                    "run_name": run_name,
                    "agent_name": args.agent_name,
                    "model": model,
                    "total_events": len(all_events),
                    "llm_messages_count": len(llm_messages),
                    "elapsed_time": elapsed_time,
                    "events": []
                }

                for event in all_events:
                    try:
                        # Try Pydantic v2 method
                        if hasattr(event, 'model_dump'):
                            events_data["events"].append(event.model_dump(mode='json'))
                        # Try Pydantic v1 method
                        elif hasattr(event, 'dict'):
                            events_data["events"].append(event.dict())
                        # Fallback to manual conversion
                        else:
                            event_dict = {"type": type(event).__name__}
                            if hasattr(event, '__dict__'):
                                for key, value in event.__dict__.items():
                                    if not key.startswith('_'):
                                        try:
                                            json.dumps(value)  # Test if serializable
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

            # 3. Save LLM messages separately (for replay/analysis)
            llm_messages_file = logs_dir / "llm_messages.json"
            try:
                llm_data = {
                    "run_name": run_name,
                    "total_messages": len(llm_messages),
                    "messages": [str(msg) for msg in llm_messages]
                }

                with open(llm_messages_file, 'w', encoding='utf-8') as f:
                    json.dump(llm_data, f, indent=2)

                logger.info(f"💾 Saved LLM messages ({len(llm_messages)} messages): {llm_messages_file}")

            except Exception as e:
                logger.warning(f"Could not save LLM messages: {e}")

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

        if not analysis_text and final_response:
            # Use final response as analysis if no file found
            analysis_text = final_response
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
        message = f"SDK execution {execution_status}"
        if success:
            message += f" - Task completed in {elapsed_time:.2f}s"
            if total_cost:
                message += f" (cost: ${total_cost:.4f})"
        else:
            message += f" - Check logs for details"

        logger.info(f"✅ OpenHands SDK execution completed")
        logger.info("=================== OPENHANDS SDK EXECUTION COMPLETED ===================")

        return OpenHandSDKResults(
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
            conversation_id=str(state.id) if hasattr(state, 'id') else None,
            execution_status=execution_status,
            final_response=final_response,
            total_cost=total_cost,
            total_tokens=total_tokens
        )

    except Exception as e:
        elapsed_time = time.time() - start_time
        error_message = f"SDK execution failed: {type(e).__name__}: {e}"
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

        logger.info("=================== OPENHANDS SDK EXECUTION FAILED ===================")

        return OpenHandSDKResults(
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
            total_tokens=None
        )


# Test prompt
PROMPT = '''
Please create a simple Python script:
1. Save it to code/hello.py
2. The script should print "Hello from OpenHands SDK!"
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

    logger.info("=================== STARTING SDK TEST ===================")

    # Create test arguments
    args = OpenHandSDKArgs(
        prompt=PROMPT,
        topic_type="test",
        topic_idx=1,
        topic_name="sdk_hello",
        workspace_dir="projspace/proj_sdk_hello",
        max_iterations=10
    )

    logger.info(f"Args: {json.dumps(args.to_dict(), indent=4)}")

    # Execute with SDK
    result = openhand_sdk_tool(args)

    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS:")
    logger.info(f"{'='*80}")
    logger.info(f"Success: {result.success}")
    logger.info(f"Message: {result.message}")
    logger.info(f"Status: {result.execution_status}")
    logger.info(f"Conversation ID: {result.conversation_id}")
    logger.info(f"Run dir: {result.run_dir}")
    logger.info(f"Log file: {result.log_file}")

    if result.total_cost:
        logger.info(f"Cost: ${result.total_cost:.4f}")
    if result.total_tokens:
        logger.info(f"Tokens: {result.total_tokens}")

    if result.analysis_file:
        logger.info(f"Analysis file: {result.analysis_file}")

    if result.final_response:
        logger.info(f"\nFinal Response Preview:")
        logger.info(f"{result.final_response[:300]}...")

    logger.info(f"{'='*80}")

    if result.success:
        logger.info("✅ SDK test completed successfully!")
    else:
        logger.error("❌ SDK test failed - check logs")


"""
Usage Examples:

# Basic usage
python digitalme/tools/tools_ohsdk.py

# Or import and use in your code:
from eventglucose.tools.tools_ohsdk import OpenHandSDKArgs, openhand_sdk_tool

args = OpenHandSDKArgs(
    prompt="Your detailed task...",
    topic_type="data",
    topic_idx=1,
    topic_name="analysis",
    workspace_dir="projspace/proj_analysis"
)

result = openhand_sdk_tool(args)
print(f"Success: {result.success}")
print(f"Cost: ${result.total_cost}")
print(result.final_response)
"""
