"""
SDK Streaming Output Helper
===========================

Shared helper for formatting and printing Claude SDK messages.
Used by both tools_llm_provider.py and tools_ccsdk.py for consistent output.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional


def _print_flush(text: str) -> None:
    """Print with immediate flush to ensure real-time streaming output."""
    print(text, flush=True)


def print_sdk_message(message: Any, verbose: bool = True) -> Optional[str]:
    """
    Print formatted SDK message to console with real-time streaming.

    Args:
        message: SDK message (AssistantMessage, ResultMessage, etc.)
        verbose: If True, print detailed output. If False, minimal output.

    Returns:
        Final response text if this message contains it, None otherwise.
    """
    # Import SDK types locally to avoid import errors if SDK not installed
    try:
        from claude_agent_sdk.types import (
            AssistantMessage,
            TextBlock,
            ThinkingBlock,
            ToolUseBlock,
            ToolResultBlock,
            ResultMessage,
        )
    except ImportError:
        return None

    final_response = None

    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                final_response = block.text
                if verbose:
                    # Truncate long responses
                    display_text = block.text[:500] + "..." if len(block.text) > 500 else block.text
                    _print_flush(f"\n💬 Claude: {display_text}")

            elif isinstance(block, ThinkingBlock):
                if verbose:
                    thinking_preview = block.thinking[:200] + "..." if len(block.thinking) > 200 else block.thinking
                    _print_flush(f"\n🤔 Thinking: {thinking_preview}")

            elif isinstance(block, ToolUseBlock):
                if verbose:
                    _print_flush(f"\n🔧 Using tool: {block.name}")
                    # Print tool inputs in a readable format
                    if block.name == "Bash":
                        _print_flush(f"   Command: {block.input.get('command', 'N/A')}")
                    elif block.name in ["Read", "Write", "Edit"]:
                        _print_flush(f"   File: {block.input.get('file_path', 'N/A')}")
                    elif block.name == "Glob":
                        _print_flush(f"   Pattern: {block.input.get('pattern', 'N/A')}")
                    elif block.name == "Grep":
                        _print_flush(f"   Pattern: {block.input.get('pattern', 'N/A')}")
                    else:
                        try:
                            input_str = json.dumps(block.input, indent=2)[:150]
                            _print_flush(f"   Input: {input_str}...")
                        except Exception:
                            _print_flush(f"   Input: {str(block.input)[:150]}...")

            elif isinstance(block, ToolResultBlock):
                if verbose:
                    if block.is_error:
                        _print_flush(f"   ❌ Tool error: {str(block.content)[:200]}")
                    else:
                        _print_flush(f"   ✅ Tool completed")

    elif isinstance(message, ResultMessage):
        if verbose:
            _print_flush(f"\n{'='*60}")
            _print_flush(f"✅ Task completed!")
            _print_flush(f"   Turns: {message.num_turns}")
            if message.total_cost_usd:
                _print_flush(f"   Cost: ${message.total_cost_usd:.4f}")
            _print_flush(f"{'='*60}\n")

    return final_response


def print_sdk_result(num_turns: int, total_cost: Optional[float] = None) -> None:
    """
    Print SDK execution result summary.

    Args:
        num_turns: Number of turns used
        total_cost: Total cost in USD (optional)
    """
    if total_cost:
        _print_flush(f"\n✅ Complete (turns: {num_turns}, cost: ${total_cost:.4f})")
    else:
        _print_flush(f"\n✅ Complete (turns: {num_turns})")
