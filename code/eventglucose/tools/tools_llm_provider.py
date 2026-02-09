"""
Unified LLM Provider - Direct LLM vs CC-SDK vs Codex-SDK
=========================================================

Module Structure
----------------

┌─────────────────────────────────────────────────────────────────────┐
│                   tools_llm_provider.py                             │
│              Unified LLM Provider Abstraction                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  THREE BACKENDS                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  "direct" ───> langchain ChatAnthropic/ChatOpenAI          │    │
│  │                Fast, pay-per-token                         │    │
│  │                                                            │    │
│  │  "ccsdk" ────> Claude Code SDK                             │    │
│  │                Subscription-based, free with Pro/Max       │    │
│  │                                                            │    │
│  │  "codexsdk" ─> Codex Client SDK                            │    │
│  │                Codex-based execution                       │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  LLMProvider CLASS                                                  │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  invoke(prompt, output_schema?) ──> Response/BaseModel     │    │
│  │    └─> Simple chat (one round, no tools) - ALL backends   │    │
│  │                                                            │    │
│  │  run_agentic_task(task, workspace, output_path)            │    │
│  │    └─> Full agentic execution - CC-SDK & Codex-SDK        │    │
│  │                                                            │    │
│  │  gather_context(query, workspace, allowed_paths)           │    │
│  │    └─> Explore workspace - CC-SDK & Codex-SDK             │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  FACTORY FUNCTION                                                   │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  get_llm_provider(agent, node) ─> LLMProvider              │    │
│  │    └─> Loads config from YAML, returns configured provider │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  CONFIG SOURCE: config/dikw-agent.yaml                              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  backend: "direct" | "ccsdk" | "codexsdk"                  │    │
│  │  model: various (see backend-specific models)             │    │
│  │  temperature, max_tokens, enable_fallback                  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Usage
-----

::

    # Get provider with YAML config
    provider = get_llm_provider(agent="planner_agent", node="generate_plan")
    response = provider.invoke("Generate a plan for...")

    # Structured output
    class MathResult(BaseModel):
        answer: int
        explanation: str

    result = provider.invoke("Calculate 2+2", output_schema=MathResult)
    print(result.answer)  # 4

    # Agentic execution (ccsdk or codexsdk)
    result = provider.run_agentic_task(
        task="Analyze data and create report",
        workspace_root="./data",
        output_path="report.md",
    )
"""

from __future__ import annotations

import os
import json
import logging
import asyncio
from typing import Any, Dict, List, Optional, Type, Union, Literal
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# JSON schema instruction for structured output (used with CC-SDK)
JSON_SCHEMA_INSTRUCTION = """
IMPORTANT: Return your response as valid JSON matching this schema:
{schema_json}

Return ONLY the JSON object, no additional text or markdown code blocks."""

# Output path instruction for agentic tasks
OUTPUT_PATH_INSTRUCTION = """

IMPORTANT: Save your final output to: {output_path}
Make sure to write the complete analysis/report to this file."""


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

BackendType = Literal["direct", "ccsdk", "codexsdk"]


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class LLMResult:
    """Result from LLM invocation."""
    success: bool
    content: str
    structured_output: Optional[BaseModel] = None
    backend_used: str = "unknown"
    cost_usd: Optional[float] = None
    tokens_used: Optional[int] = None
    error_message: Optional[str] = None


@dataclass
class AgenticTaskResult:
    """Result from agentic task execution."""
    success: bool
    message: str
    output_path: Optional[str] = None
    output_content: Optional[str] = None
    backend_used: str = "ccsdk"
    cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    error_message: Optional[str] = None


# =============================================================================
# PROVIDER CONFIGURATION
# =============================================================================

@dataclass
class LLMProviderConfig:
    """
    Configuration for LLM Provider.

    All values come from config/dikw-agent.yaml - no hardcoded defaults.
    """

    # LLM Backend selection (from YAML)
    llm_backend: BackendType

    # Model configuration (from YAML)
    model: str
    temperature: float
    max_tokens: int

    # Fallback settings (from YAML)
    enable_fallback: bool

    # CC-SDK specific settings
    ccsdk_max_turns: int = 30
    ccsdk_workspace_root: Optional[str] = None
    ccsdk_allowed_tools: List[str] = field(default_factory=lambda: [
        "Read", "Glob", "Grep"  # Safe tools for context gathering
    ])
    ccsdk_system_prompt: Optional[str] = None  # Custom system prompt for SDK

    # Codex-SDK specific settings
    codexsdk_max_turns: int = 30
    codexsdk_workspace_root: Optional[str] = None
    codexsdk_reasoning_effort: str = "minimal"  # minimal/low/medium/high
    codexsdk_sandbox_mode: str = "read_only"    # read_only/workspace_write/danger_full_access
    codexsdk_approval_policy: str = "auto"      # auto/manual
    codexsdk_verbosity: str = "normal"          # quiet/normal/verbose

    # Source tracking
    resolved_from: str = "default"  # Where config came from (e.g., "planner.nodes.generate_plan")

    # Alias for backward compatibility
    @property
    def backend(self) -> BackendType:
        """Alias for llm_backend (backward compatibility)."""
        return self.llm_backend

    @backend.setter
    def backend(self, value: BackendType) -> None:
        """Setter for backward compatibility."""
        self.llm_backend = value

    @classmethod
    def from_yaml(
        cls,
        agent: Optional[str] = None,
        node: Optional[str] = None,
    ) -> "LLMProviderConfig":
        """
        Create config from YAML file (single source of truth).

        Priority order:
        1. Per-node setting (agent.nodes.node.key)
        2. Per-agent setting (agent.key)
        3. Global default (default.key)

        Args:
            agent: Agent type ("planner", "data", "information", "knowledge", "wisdom", "decision")
            node: Node/function name (e.g., "generate_plan", "reasoning_synthesis")

        Returns:
            LLMProviderConfig with resolved settings

        Raises:
            FileNotFoundError: If config/dikw-agent.yaml not found
            ValueError: If required config keys missing
        """
        from eventglucose.state.llm_config import get_llm_config

        yaml_config = get_llm_config(agent=agent, node=node)

        return cls(
            llm_backend=yaml_config.llm_backend,
            model=yaml_config.model,
            temperature=yaml_config.temperature,
            max_tokens=yaml_config.max_tokens,
            enable_fallback=yaml_config.enable_fallback,
            ccsdk_max_turns=30,  # CC-SDK specific, not in main config
            ccsdk_workspace_root=None,
            resolved_from=yaml_config.resolved_from,
        )


# =============================================================================
# LLM PROVIDER (Main Class)
# =============================================================================

class LLMProvider:
    """
    Unified LLM Provider with switchable backend.

    Configuration comes from config/dikw-agent.yaml (single source of truth).

    Supports:
    - Direct LLM calls (langchain ChatAnthropic/ChatOpenAI)
    - CC-SDK calls (Claude Code SDK with subscription)
    - Automatic fallback from CC-SDK to Direct LLM
    - Structured output for both backends
    """

    _instances: Dict[str, "LLMProvider"] = {}

    def __init__(self, config: Optional[LLMProviderConfig] = None):
        """
        Initialize provider with configuration.

        If no config provided, loads from YAML file.
        """
        self.config = config or LLMProviderConfig.from_yaml()
        self._direct_llm = None
        self._ccsdk_client = None
        logger.info(f"LLMProvider initialized: llm_backend={self.config.llm_backend}, from={self.config.resolved_from}")

    @classmethod
    def get_instance(
        cls,
        backend: Optional[BackendType] = None,
        config: Optional[LLMProviderConfig] = None,
        agent: Optional[str] = None,
        node: Optional[str] = None,
    ) -> "LLMProvider":
        """
        Get singleton instance of LLMProvider.

        Configuration loaded from config/dikw-agent.yaml.

        Args:
            backend: Override backend type ("direct" or "ccsdk")
            config: Full configuration override
            agent: Agent type for YAML config lookup
            node: Node/function name for YAML config lookup

        Returns:
            LLMProvider instance
        """
        if config:
            key = f"{config.llm_backend}_{config.model}_{config.resolved_from}"
        elif agent or node:
            key = f"{agent}_{node}"
        elif backend:
            key = f"{backend}_override"
        else:
            key = "default"

        if key not in cls._instances:
            if config:
                cls._instances[key] = cls(config)
            elif agent or node:
                cfg = LLMProviderConfig.from_yaml(agent=agent, node=node)
                if backend:
                    cfg.llm_backend = backend  # Allow override
                cls._instances[key] = cls(cfg)
            elif backend:
                cfg = LLMProviderConfig.from_yaml()
                cfg.llm_backend = backend
                cls._instances[key] = cls(cfg)
            else:
                cls._instances[key] = cls()

        return cls._instances[key]

    @classmethod
    def reset_instances(cls) -> None:
        """Reset all singleton instances (useful for testing)."""
        cls._instances.clear()

    # -------------------------------------------------------------------------
    # MAIN INVOCATION METHOD
    # -------------------------------------------------------------------------

    def invoke(
        self,
        prompt: str,
        output_schema: Optional[Type[BaseModel]] = None,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        force_backend: Optional[BackendType] = None,
        workspace_root: Optional[str] = None,
    ) -> Union[str, BaseModel, LLMResult]:
        """
        Invoke LLM with prompt and optional structured output.

        Args:
            prompt: The prompt to send to the LLM
            output_schema: Optional Pydantic model for structured output
            system_message: Optional system message
            temperature: Override temperature for this call
            force_backend: Force specific backend ("direct", "ccsdk", or "codexsdk")
            workspace_root: Workspace root (ignored for simple chat mode)

        Returns:
            - If output_schema: Returns instance of output_schema
            - If no output_schema: Returns string response
            - On error with fallback disabled: Returns LLMResult with error

        Raises:
            Exception: If both backends fail and fallback is enabled
        """
        backend = force_backend or self.config.llm_backend
        temp = temperature if temperature is not None else self.config.temperature

        try:
            if backend == "ccsdk":
                result = self._invoke_ccsdk(
                    prompt=prompt,
                    output_schema=output_schema,
                    system_message=system_message,
                    temperature=temp,
                    workspace_root=workspace_root,
                )
            elif backend == "codexsdk":
                result = self._invoke_codexsdk(
                    prompt=prompt,
                    output_schema=output_schema,
                    system_message=system_message,
                    temperature=temp,
                    workspace_root=workspace_root,
                )
            else:
                result = self._invoke_direct(
                    prompt=prompt,
                    output_schema=output_schema,
                    system_message=system_message,
                    temperature=temp,
                )

            if result.success:
                if output_schema and result.structured_output:
                    return result.structured_output
                return result.content

            # Handle error
            raise Exception(result.error_message)

        except Exception as e:
            logger.warning(f"LLM invocation failed with {backend}: {e}")

            # Try fallback if enabled and not already using direct
            if self.config.enable_fallback and backend in ["ccsdk", "codexsdk"]:
                logger.info("Falling back to direct LLM...")
                try:
                    result = self._invoke_direct(
                        prompt=prompt,
                        output_schema=output_schema,
                        system_message=system_message,
                        temperature=temp,
                    )
                    if result.success:
                        if output_schema and result.structured_output:
                            return result.structured_output
                        return result.content
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
                    raise

            raise

    # -------------------------------------------------------------------------
    # DIRECT LLM BACKEND
    # -------------------------------------------------------------------------

    def _get_direct_llm(self, temperature: float = 0.0):
        """Get or create direct LLM client."""
        model = self.config.model

        # Try to restore API key if it was saved (after CCSDK calls)
        # API key is saved locally (removed from env so ClaudeSDKClient uses subscription)
        from ._auth import get_anthropic_api_key, has_anthropic_api_key

        # Determine if using Anthropic or OpenAI
        if "claude" in model.lower() or has_anthropic_api_key():
            from langchain_anthropic import ChatAnthropic
            api_key = get_anthropic_api_key()
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not available. "
                    "Set it in env.sh or use CCSDK backend instead."
                )
            return ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=self.config.max_tokens,
                api_key=api_key,
            )
        else:
            from langchain_openai import ChatOpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=self.config.max_tokens,
                api_key=api_key,  # Explicitly pass API key
            )

    def _invoke_direct(
        self,
        prompt: str,
        output_schema: Optional[Type[BaseModel]] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResult:
        """Invoke using direct LLM (langchain)."""
        try:
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = self._get_direct_llm(temperature)

            # Build messages
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))

            # Handle structured output
            if output_schema:
                structured_llm = llm.with_structured_output(output_schema)
                response = structured_llm.invoke(messages)
                return LLMResult(
                    success=True,
                    content=str(response),
                    structured_output=response,
                    backend_used="direct",
                )
            else:
                response = llm.invoke(messages)
                return LLMResult(
                    success=True,
                    content=response.content,
                    backend_used="direct",
                )

        except Exception as e:
            logger.error(f"Direct LLM error: {e}")
            return LLMResult(
                success=False,
                content="",
                backend_used="direct",
                error_message=str(e),
            )

    # -------------------------------------------------------------------------
    # CC-SDK BACKEND
    # -------------------------------------------------------------------------

    def _invoke_ccsdk(
        self,
        prompt: str,
        output_schema: Optional[Type[BaseModel]] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.0,
        workspace_root: Optional[str] = None,
    ) -> LLMResult:
        """Invoke using Claude Code SDK - Simple Chat Mode (one round, no tools)."""
        try:
            from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
            from claude_agent_sdk.types import AssistantMessage, TextBlock, ResultMessage

            # Build enhanced prompt with schema if needed
            enhanced_prompt = prompt
            if system_message:
                enhanced_prompt = f"{system_message}\n\n{prompt}"

            if output_schema:
                schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
                enhanced_prompt += JSON_SCHEMA_INSTRUCTION.format(schema_json=schema_json)

            # Configure SDK options - Simple Chat Mode
            # No workspace, no tools, single round
            options = ClaudeAgentOptions(
                cwd=None,  # No workspace for simple chat
                allowed_tools=[],  # No tools for simple chat
                permission_mode="acceptEdits",
                max_turns=1,  # Single round only
                model=self.config.model,
                system_prompt=self.config.ccsdk_system_prompt,  # Custom system prompt
            )

            # Run SDK
            final_response = None
            total_cost = None
            num_turns = None

            async def run_sdk():
                nonlocal final_response, total_cost, num_turns

                # EnsureClaudeSDKAuth handles auth:
                # - If ~/.claude exists: SDK uses subscription (preferred)
                # - If not: temporarily restore API key to env
                from ._auth import EnsureClaudeSDKAuth

                from ._sdk_streaming import print_sdk_message

                with EnsureClaudeSDKAuth():
                    async with ClaudeSDKClient(options=options) as client:
                        await client.query(enhanced_prompt)

                        async for message in client.receive_response():
                            # Use shared helper for consistent output formatting
                            response_text = print_sdk_message(message, verbose=True)
                            if response_text:
                                final_response = response_text

                            if isinstance(message, ResultMessage):
                                total_cost = message.total_cost_usd
                                num_turns = message.num_turns

            asyncio.run(run_sdk())

            if not final_response:
                return LLMResult(
                    success=False,
                    content="",
                    backend_used="ccsdk",
                    error_message="No response from CC-SDK",
                )

            # Parse structured output if needed
            structured_output = None
            if output_schema:
                try:
                    # Try to extract JSON from response
                    json_content = self._extract_json(final_response)
                    structured_output = output_schema.model_validate_json(json_content)
                except Exception as parse_error:
                    logger.warning(f"Failed to parse structured output: {parse_error}")
                    # Return raw response, let caller handle

            return LLMResult(
                success=True,
                content=final_response,
                structured_output=structured_output,
                backend_used="ccsdk",
                cost_usd=total_cost,
            )

        except Exception as e:
            logger.error(f"CC-SDK error: {e}")
            return LLMResult(
                success=False,
                content="",
                backend_used="ccsdk",
                error_message=str(e),
            )

    # -------------------------------------------------------------------------
    # Codex-SDK BACKEND
    # -------------------------------------------------------------------------

    def _invoke_codexsdk(
        self,
        prompt: str,
        output_schema: Optional[Type[BaseModel]] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.0,
        workspace_root: Optional[str] = None,
    ) -> LLMResult:
        """Invoke using Codex SDK - Simple Chat Mode (one round, no tools)."""
        try:
            from codex_client import (
                Client,
                CodexChatConfig,
                CodexProfile,
                ReasoningEffort,
                SandboxMode,
                ApprovalPolicy,
                Verbosity,
                AssistantMessageStream,
                TokenCountEvent,
            )

            # Build enhanced prompt
            enhanced_prompt = prompt
            if system_message:
                enhanced_prompt = f"{system_message}\n\n{prompt}"

            if output_schema:
                schema_json = json.dumps(output_schema.model_json_schema(), indent=2)
                enhanced_prompt += JSON_SCHEMA_INSTRUCTION.format(schema_json=schema_json)

            # Configure Codex profile - Simple Chat Mode
            profile = CodexProfile(
                model=self.config.model,  # "gpt-5" or "claude-opus-4-5"
                reasoning_effort=ReasoningEffort.MINIMAL,  # Fast mode for simple chat
                sandbox=SandboxMode.READ_ONLY,  # No file access for simple chat
                verbosity=Verbosity.NORMAL,
            )

            config = CodexChatConfig(
                profile=profile,
                approval_policy=ApprovalPolicy.AUTO,  # Auto-approve for simple chat
                cwd=None,  # No workspace for simple chat
                sandbox=SandboxMode.READ_ONLY,
            )

            # Execute async (single round)
            final_response = None
            total_tokens = None

            async def run_codex():
                nonlocal final_response, total_tokens

                async with Client(args=["mcp"]) as client:
                    chat = await client.create_chat(enhanced_prompt, config=config)

                    # Single round - just get response
                    async for event in chat:
                        if isinstance(event, AssistantMessageStream):
                            chunks = []
                            async for chunk in event.stream():
                                chunks.append(chunk)
                            final_response = "".join(chunks)

                        elif isinstance(event, TokenCountEvent):
                            if event.info and event.info.total_token_usage:
                                total_tokens = event.info.total_token_usage.total_tokens

            asyncio.run(run_codex())

            if not final_response:
                return LLMResult(
                    success=False,
                    content="",
                    backend_used="codexsdk",
                    error_message="No response from Codex SDK",
                )

            # Parse structured output if needed
            structured_output = None
            if output_schema:
                try:
                    json_content = self._extract_json(final_response)
                    structured_output = output_schema.model_validate_json(json_content)
                except Exception as parse_error:
                    logger.warning(f"Failed to parse structured output: {parse_error}")

            return LLMResult(
                success=True,
                content=final_response,
                structured_output=structured_output,
                backend_used="codexsdk",
                tokens_used=total_tokens,
            )

        except Exception as e:
            logger.error(f"CodexSDK error: {e}")
            return LLMResult(
                success=False,
                content="",
                backend_used="codexsdk",
                error_message=str(e),
            )

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text response."""
        # Try to find JSON in the response
        text = text.strip()

        # If wrapped in markdown code block
        if text.startswith("```json"):
            text = text[7:]
            end = text.find("```")
            if end != -1:
                text = text[:end]
        elif text.startswith("```"):
            text = text[3:]
            end = text.find("```")
            if end != -1:
                text = text[:end]

        # Find JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

        return text.strip()

    # -------------------------------------------------------------------------
    # AGENTIC TASK METHODS (CC-SDK & Codex-SDK)
    # -------------------------------------------------------------------------

    def run_agentic_task(
        self,
        task: str,
        workspace_root: str,
        output_path: Optional[str] = None,
        allow_code_execution: bool = False,
        max_turns: Optional[int] = None,
        force_backend: Optional[BackendType] = None,
    ) -> AgenticTaskResult:
        """
        Run an agentic task with full workspace access.

        This uses CC-SDK or Codex-SDK to run a task that can:
        - Read files from the workspace
        - Write output reports
        - Optionally execute code (if allow_code_execution=True)

        Args:
            task: Task description
            workspace_root: Root directory for workspace
            output_path: Optional path for output file (relative to workspace)
            allow_code_execution: Allow Bash/code execution (default: False)
            max_turns: Maximum turns for SDK execution
            force_backend: Force specific backend ("ccsdk" or "codexsdk")

        Returns:
            AgenticTaskResult with execution details
        """
        backend = force_backend or self.config.llm_backend

        # Route to appropriate backend
        if backend == "codexsdk":
            return self._run_agentic_task_codexsdk(
                task=task,
                workspace_root=workspace_root,
                output_path=output_path,
                allow_code_execution=allow_code_execution,
                max_turns=max_turns,
            )
        elif backend == "ccsdk":
            return self._run_agentic_task_ccsdk(
                task=task,
                workspace_root=workspace_root,
                output_path=output_path,
                allow_code_execution=allow_code_execution,
                max_turns=max_turns,
            )
        else:
            return AgenticTaskResult(
                success=False,
                message=f"Agentic tasks not supported for backend: {backend}",
                error_message="Use 'ccsdk' or 'codexsdk' backend for agentic tasks",
            )

    def _run_agentic_task_ccsdk(
        self,
        task: str,
        workspace_root: str,
        output_path: Optional[str] = None,
        allow_code_execution: bool = False,
        max_turns: Optional[int] = None,
    ) -> AgenticTaskResult:
        """Run agentic task using Claude Code SDK."""
        try:
            from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
            from claude_agent_sdk.types import AssistantMessage, TextBlock, ResultMessage

            # Configure allowed tools
            allowed_tools = ["Read", "Glob", "Grep"]
            if allow_code_execution:
                allowed_tools.extend(["Bash", "Write", "Edit"])
            elif output_path:
                allowed_tools.append("Write")  # Need Write for output

            # Build task prompt
            task_prompt = task
            if output_path:
                task_prompt += OUTPUT_PATH_INSTRUCTION.format(output_path=output_path)

            # Use model directly from config (YAML should specify SDK model name: haiku/sonnet/opus)
            options = ClaudeAgentOptions(
                cwd=workspace_root,
                allowed_tools=allowed_tools,
                permission_mode="acceptEdits",
                max_turns=max_turns or self.config.ccsdk_max_turns,
                model=self.config.model,
            )

            final_response = None
            total_cost = None
            num_turns = None

            async def run_sdk():
                nonlocal final_response, total_cost, num_turns

                # EnsureClaudeSDKAuth handles auth:
                # - If ~/.claude exists: SDK uses subscription (preferred)
                # - If not: temporarily restore API key to env
                from ._auth import EnsureClaudeSDKAuth

                from ._sdk_streaming import print_sdk_message

                with EnsureClaudeSDKAuth():
                    async with ClaudeSDKClient(options=options) as client:
                        await client.query(task_prompt)

                        async for message in client.receive_response():
                            # Use shared helper for consistent output formatting
                            response_text = print_sdk_message(message, verbose=True)
                            if response_text:
                                final_response = response_text

                            if isinstance(message, ResultMessage):
                                total_cost = message.total_cost_usd
                                num_turns = message.num_turns

            asyncio.run(run_sdk())

            # Check if output file was created
            output_content = None
            if output_path:
                full_path = Path(workspace_root) / output_path
                if full_path.exists():
                    output_content = full_path.read_text(encoding="utf-8")

            return AgenticTaskResult(
                success=True,
                message=f"Task completed in {num_turns} turns",
                output_path=output_path,
                output_content=output_content or final_response,
                backend_used="ccsdk",
                cost_usd=total_cost,
                num_turns=num_turns,
            )

        except Exception as e:
            logger.error(f"Agentic task error: {e}")
            return AgenticTaskResult(
                success=False,
                message=f"Task failed: {e}",
                backend_used="ccsdk",
                error_message=str(e),
            )

    def _run_agentic_task_codexsdk(
        self,
        task: str,
        workspace_root: str,
        output_path: Optional[str] = None,
        allow_code_execution: bool = False,
        max_turns: Optional[int] = None,
    ) -> AgenticTaskResult:
        """Run agentic task using Codex SDK."""
        try:
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
                TokenCountEvent,
                TaskCompleteEvent,
            )

            # Build task prompt
            task_prompt = task
            if output_path:
                task_prompt += OUTPUT_PATH_INSTRUCTION.format(output_path=output_path)

            # Determine sandbox mode based on permissions
            if allow_code_execution:
                sandbox = SandboxMode.WORKSPACE_WRITE  # Can write and execute
            elif output_path:
                sandbox = SandboxMode.WORKSPACE_WRITE  # Need write for output
            else:
                sandbox = SandboxMode.READ_ONLY  # Read-only access

            # Configure Codex profile - Agentic Mode
            profile = CodexProfile(
                model=self.config.model,  # "gpt-5" or "claude-opus-4-5"
                reasoning_effort=ReasoningEffort(self.config.codexsdk_reasoning_effort),
                sandbox=sandbox,
                verbosity=Verbosity(self.config.codexsdk_verbosity),
            )

            config = CodexChatConfig(
                profile=profile,
                approval_policy=ApprovalPolicy(self.config.codexsdk_approval_policy),
                cwd=workspace_root,
                sandbox=sandbox,
            )

            # Execute async (multi-turn)
            final_response = None
            total_tokens = None
            num_turns = 0
            session_configured = False

            async def run_codex():
                nonlocal final_response, total_tokens, num_turns, session_configured

                async with Client(args=["mcp"]) as client:
                    chat = await client.create_chat(task_prompt, config=config)

                    # Multi-turn execution with events
                    async for event in chat:
                        if isinstance(event, AssistantMessageStream):
                            chunks = []
                            async for chunk in event.stream():
                                chunks.append(chunk)
                            final_response = "".join(chunks)
                            num_turns += 1

                        elif isinstance(event, CommandStream):
                            # Track command execution (indicates active work)
                            num_turns += 1

                        elif isinstance(event, TokenCountEvent):
                            if event.info and event.info.total_token_usage:
                                total_tokens = event.info.total_token_usage.total_tokens

                        elif isinstance(event, TaskCompleteEvent):
                            session_configured = True

            asyncio.run(run_codex())

            # Check if output file was created
            output_content = None
            if output_path:
                full_path = Path(workspace_root) / output_path
                if full_path.exists():
                    output_content = full_path.read_text(encoding="utf-8")

            return AgenticTaskResult(
                success=session_configured or final_response is not None,
                message=f"Task completed in {num_turns} turns" if session_configured else "Task execution finished",
                output_path=output_path,
                output_content=output_content or final_response,
                backend_used="codexsdk",
                cost_usd=None,  # Codex SDK doesn't provide cost info
                num_turns=num_turns,
            )

        except Exception as e:
            logger.error(f"Agentic task error (Codex): {e}")
            return AgenticTaskResult(
                success=False,
                message=f"Task failed: {e}",
                backend_used="codexsdk",
                error_message=str(e),
            )

    def gather_context(
        self,
        query: str,
        workspace_root: str,
        allowed_paths: Optional[List[str]] = None,
        max_files: int = 10,
        output_format: str = "markdown",
    ) -> str:
        """
        Use CC-SDK to explore workspace and gather relevant context.

        This is useful for the HYBRID pattern where CC-SDK gathers context
        and then direct LLM generates the final output.

        Args:
            query: What context to gather (e.g., "Find all D/I level reports about customer churn")
            workspace_root: Root directory for workspace
            allowed_paths: List of paths to search (e.g., ["reports/d/", "reports/i/"])
            max_files: Maximum files to read
            output_format: "markdown" or "json"

        Returns:
            Gathered context as string
        """
        task = f"""Explore the workspace and gather context relevant to: {query}

Instructions:
1. Use Glob to find relevant files"""

        if allowed_paths:
            task += f"\n2. Focus on these directories: {', '.join(allowed_paths)}"
        else:
            task += "\n2. Search the entire workspace"

        task += f"""
3. Use Read to examine the content of up to {max_files} relevant files
4. Summarize the key findings in {output_format} format

Return ONLY the gathered context, no additional commentary."""

        result = self.run_agentic_task(
            task=task,
            workspace_root=workspace_root,
            allow_code_execution=False,
            max_turns=min(20, self.config.ccsdk_max_turns),  # Limit turns for context gathering
        )

        if result.success:
            return result.output_content or ""
        else:
            logger.warning(f"Context gathering failed: {result.error_message}")
            return ""

    def answer_question(
        self,
        question: str,
        workspace_root: str,
        context_hints: Optional[List[str]] = None,
    ) -> str:
        """
        Answer a question with workspace file access.

        Args:
            question: The question to answer
            workspace_root: Root directory for workspace
            context_hints: Optional list of files/folders to check first

        Returns:
            Answer string
        """
        task = f"""Answer this question: {question}

You have access to the workspace files. """

        if context_hints:
            task += f"Start by checking these files/folders: {', '.join(context_hints)}"
        else:
            task += "Use Glob and Read to find relevant information."

        task += """

Provide a clear, concise answer based on what you find in the workspace."""

        result = self.run_agentic_task(
            task=task,
            workspace_root=workspace_root,
            allow_code_execution=False,
            max_turns=min(15, self.config.ccsdk_max_turns),
        )

        if result.success:
            return result.output_content or "Unable to find answer."
        else:
            return f"Error: {result.error_message}"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_llm_provider(
    backend: Optional[BackendType] = None,
    agent: Optional[str] = None,
    node: Optional[str] = None,
) -> LLMProvider:
    """
    Get LLM provider instance with configuration.

    Args:
        backend: Override backend type ("direct" or "ccsdk")
        agent: Agent type for YAML config lookup ("planner", "data", "knowledge", etc.)
        node: Node/function name for YAML config lookup

    Returns:
        LLMProvider configured for the specified agent/node

    Examples:
        # Get provider with YAML config for planner's generate_plan
        provider = get_llm_provider(agent="planner_agent", node="generate_plan")

        # Get provider with default config
        provider = get_llm_provider()

        # Force specific backend
        provider = get_llm_provider(backend="ccsdk")
    """
    if agent or node:
        # Use YAML config with agent/node resolution
        config = LLMProviderConfig.from_yaml(agent=agent, node=node)
        if backend:
            config.llm_backend = backend  # Allow override
        return LLMProvider(config)
    elif backend:
        # Override backend only
        config = LLMProviderConfig.from_yaml()
        config.llm_backend = backend
        return LLMProvider(config)
    else:
        # Use default singleton
        return LLMProvider.get_instance()


def invoke_llm(
    prompt: str,
    output_schema: Optional[Type[BaseModel]] = None,
    system_message: Optional[str] = None,
    temperature: float = 0.0,
    backend: Optional[BackendType] = None,
) -> Union[str, BaseModel]:
    """
    Convenience function for quick LLM invocation.

    Usage:
        response = invoke_llm("What is 2+2?")

        # With structured output
        class Result(BaseModel):
            answer: int

        result = invoke_llm("Calculate 2+2", output_schema=Result)
    """
    provider = LLMProvider.get_instance(backend=backend)
    return provider.invoke(
        prompt=prompt,
        output_schema=output_schema,
        system_message=system_message,
        temperature=temperature,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "BackendType",
    "LLMResult",
    "AgenticTaskResult",
    "LLMProviderConfig",
    "LLMProvider",
    "get_llm_provider",
    "invoke_llm",
]
