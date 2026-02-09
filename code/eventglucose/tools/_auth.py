"""
Authentication Utilities for DIKW Tools
========================================

This module provides shared authentication logic for Claude and OpenAI APIs.

Authentication Priority
-----------------------

For Claude Agent SDK (tools_ccsdk.py, tools_llm_provider.py):

1. ~/.claude subscription (HIGHEST PRIORITY - cost effective!)
   - Free with Claude Pro/Max subscription
   - No API key management needed

2. ANTHROPIC_API_KEY env var (fallback - pay-per-token)
   - Standard API access
   - Billed per token

For OpenAI (tools_llm_provider.py direct backend):
   - OPENAI_API_KEY env var

Usage
-----

::

    from eventglucose.tools._auth import configure_claude_auth, get_api_key_for_model

    # Configure Claude auth (subscription preferred)
    auth_method = configure_claude_auth()
    # Returns: "subscription" | "api_key" | "none"

    # Get API key for a model
    api_key = get_api_key_for_model("claude-3-5-sonnet")
    # Returns: ANTHROPIC_API_KEY or None
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

AuthMethod = Literal["subscription", "api_key", "none"]


# =============================================================================
# API KEY MANAGEMENT
# =============================================================================
#
# Strategy: Save API key locally, remove from environment.
# - ChatAnthropic: Pass api_key explicitly via get_anthropic_api_key()
# - ClaudeSDKClient: No key in env → automatically uses ~/.claude subscription
#
# This allows:
# - ClaudeSDKClient to prefer subscription auth (free with Pro/Max)
# - ChatAnthropic to still use API key when needed

# Save API key at module load time and remove from environment
_ANTHROPIC_API_KEY: Optional[str] = os.environ.pop("ANTHROPIC_API_KEY", None)

if _ANTHROPIC_API_KEY:
    logger.info("API Key Management: Saved ANTHROPIC_API_KEY locally, removed from environment")
    logger.info("   → ClaudeSDKClient will use ~/.claude subscription (if available)")
    logger.info("   → ChatAnthropic will receive key via get_anthropic_api_key()")


def get_anthropic_api_key() -> Optional[str]:
    """
    Get the saved Anthropic API key.

    This key was saved at module load time and removed from environment
    so that ClaudeSDKClient prefers ~/.claude subscription auth.

    Use this when creating ChatAnthropic instances:
        llm = ChatAnthropic(api_key=get_anthropic_api_key())

    Returns:
        The API key string, or None if not available.
    """
    return _ANTHROPIC_API_KEY


def has_anthropic_api_key() -> bool:
    """Check if an Anthropic API key is available."""
    return _ANTHROPIC_API_KEY is not None


def has_subscription_auth() -> bool:
    """Check if ~/.claude subscription auth is available."""
    return (Path.home() / ".claude").exists()


class EnsureClaudeSDKAuth:
    """
    Context manager to ensure ClaudeSDKClient has authentication.

    Priority:
    1. ~/.claude subscription (preferred, free with Pro/Max)
    2. ANTHROPIC_API_KEY (fallback, temporarily restored to env)

    Usage::

        from eventglucose.tools._auth import EnsureClaudeSDKAuth

        with EnsureClaudeSDKAuth():
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)

    How it works:
    - If ~/.claude exists: Do nothing (SDK uses subscription)
    - If ~/.claude doesn't exist but API key was saved: Temporarily restore key to env
    - On exit: Remove key from env (if we added it)
    """

    def __init__(self, quiet: bool = True):
        self._api_key_restored = False
        self._quiet = quiet
        self._auth_method = "unknown"

    def __enter__(self) -> "EnsureClaudeSDKAuth":
        """Ensure ClaudeSDKClient has auth available."""
        if has_subscription_auth():
            # Subscription available - SDK will use it
            logger.info("🔑 ClaudeSDKClient auth: ~/.claude subscription (free with Pro/Max)")
            self._auth_method = "subscription"
            return self

        # No subscription - check if we have API key
        if has_anthropic_api_key():
            # Restore API key to environment temporarily
            os.environ["ANTHROPIC_API_KEY"] = get_anthropic_api_key()
            self._api_key_restored = True
            logger.info("🔑 ClaudeSDKClient auth: API key (pay-per-token)")
            self._auth_method = "api_key"
        else:
            logger.warning("⚠️ ClaudeSDKClient auth: NONE (no subscription, no API key)")
            self._auth_method = "none"

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Remove API key from env if we restored it."""
        if self._api_key_restored:
            os.environ.pop("ANTHROPIC_API_KEY", None)
            if not self._quiet:
                logger.debug("EnsureClaudeSDKAuth: Removed ANTHROPIC_API_KEY from env")
        return None

    async def __aenter__(self) -> "EnsureClaudeSDKAuth":
        """Async context manager entry."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        return self.__exit__(exc_type, exc_val, exc_tb)


# =============================================================================
# CLAUDE AUTHENTICATION (LEGACY - kept for compatibility)
# =============================================================================

# Legacy storage (used by old configure_claude_auth/restore_api_key)
_SAVED_API_KEY: Optional[str] = None


class SubscriptionAuthContext:
    """
    Context manager for temporarily using subscription auth without losing API key.

    Problem: The Claude Agent SDK requires ANTHROPIC_API_KEY to be absent
    when using subscription auth (~/.claude). But other components (like
    langchain ChatAnthropic) need the API key.

    Solution: This context manager temporarily removes the API key, then
    restores it when exiting the context.

    Usage::

        # API key is available here for other components
        with SubscriptionAuthContext():
            # API key is temporarily removed
            # SDK uses subscription auth
            result = await client.query(prompt)
        # API key is restored here

    Note: Can also be used as a decorator for async functions.
    """

    def __init__(self, quiet: bool = True):
        self._saved_api_key: Optional[str] = None
        self._quiet = quiet

    def __enter__(self) -> "SubscriptionAuthContext":
        """Remove API key temporarily if subscription auth is available."""
        claude_credentials = Path.home() / ".claude"

        if claude_credentials.exists():
            # Save and remove API key
            if "ANTHROPIC_API_KEY" in os.environ:
                self._saved_api_key = os.environ.pop("ANTHROPIC_API_KEY")
                if not self._quiet:
                    logger.debug("SubscriptionAuthContext: Temporarily removed ANTHROPIC_API_KEY")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Restore API key if it was saved."""
        if self._saved_api_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = self._saved_api_key
            if not self._quiet:
                logger.debug("SubscriptionAuthContext: Restored ANTHROPIC_API_KEY")

        return None  # Don't suppress exceptions

    async def __aenter__(self) -> "SubscriptionAuthContext":
        """Async context manager entry."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        return self.__exit__(exc_type, exc_val, exc_tb)


def configure_claude_auth(
    prefer_subscription: bool = True,
    quiet: bool = False
) -> AuthMethod:
    """
    Configure Claude authentication (subscription vs API key).

    WARNING: This function has a SIDE EFFECT - it removes ANTHROPIC_API_KEY
    from the environment when subscription auth is used. This can break
    other components that need the API key.

    For SDK calls, prefer using SubscriptionAuthContext instead::

        with SubscriptionAuthContext():
            # SDK uses subscription, API key is preserved outside

    Subscription auth (~/.claude) is preferred because:
    - Cost effective (free with Pro/Max subscription)
    - No API key management needed

    Args:
        prefer_subscription: If True, use ~/.claude if available (default: True)
        quiet: If True, suppress info logs (default: False)

    Returns:
        "subscription" - Using ~/.claude credentials (free with Pro/Max)
        "api_key" - Using ANTHROPIC_API_KEY (pay-per-token)
        "none" - No authentication found
    """
    global _SAVED_API_KEY
    claude_credentials = Path.home() / ".claude"

    # Priority 1: Subscription auth (free with Pro/Max)
    if prefer_subscription and claude_credentials.exists():
        # Save API key before removing (so it can be restored later)
        if "ANTHROPIC_API_KEY" in os.environ:
            _SAVED_API_KEY = os.environ.pop("ANTHROPIC_API_KEY")
            if not quiet:
                logger.debug("Saved ANTHROPIC_API_KEY for later restoration")

        if not quiet:
            logger.info("Using subscription auth (~/.claude credentials) - COST EFFECTIVE!")

        return "subscription"

    # Priority 2: API key auth (pay-per-token)
    if os.environ.get("ANTHROPIC_API_KEY"):
        if not quiet:
            logger.info("Using API key auth (ANTHROPIC_API_KEY)")

        return "api_key"

    # No auth found
    if not quiet:
        logger.warning("No Claude authentication found!")
        logger.warning("To use subscription: npx @anthropic-ai/claude-code (login once)")
        logger.warning("To use API: set ANTHROPIC_API_KEY environment variable")

    return "none"


def restore_api_key() -> bool:
    """
    Restore the saved ANTHROPIC_API_KEY to the environment.

    This undoes the side effect of configure_claude_auth() when it
    uses subscription auth.

    Note: This function is idempotent - it keeps the saved key so it can
    be called multiple times safely. Only configure_claude_auth() should
    clear the saved key (when saving a new one).

    Returns:
        True if API key was restored or already present, False if no saved key

    Usage::

        configure_claude_auth()  # May remove API key
        # ... SDK calls ...
        restore_api_key()  # Bring it back for other components
    """
    global _SAVED_API_KEY

    # Already in environment - nothing to do
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True

    if _SAVED_API_KEY is not None:
        os.environ["ANTHROPIC_API_KEY"] = _SAVED_API_KEY
        logger.debug("Restored ANTHROPIC_API_KEY to environment")
        # DON'T clear _SAVED_API_KEY - keep it for future restores
        # It will be overwritten by the next configure_claude_auth() call
        return True

    return False


def check_claude_subscription() -> bool:
    """
    Check if Claude subscription credentials exist.

    Returns:
        True if ~/.claude exists, False otherwise
    """
    return (Path.home() / ".claude").exists()


# =============================================================================
# API KEY UTILITIES
# =============================================================================

def get_api_key_for_model(model: str) -> Optional[str]:
    """
    Get the appropriate API key based on model name.

    Auto-detects the provider from the model name and returns
    the corresponding API key from environment variables.

    Args:
        model: Model name (e.g., "claude-3-5-sonnet", "gpt-4o", "anthropic/claude-sonnet-4")

    Returns:
        API key string if found, None otherwise

    Example::

        api_key = get_api_key_for_model("claude-3-5-sonnet")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
    """
    model_lower = model.lower()

    # Anthropic / Claude models
    if any(x in model_lower for x in ["anthropic", "claude"]):
        return os.environ.get("ANTHROPIC_API_KEY")

    # OpenAI / GPT models
    if any(x in model_lower for x in ["openai", "gpt", "o1", "o3"]):
        return os.environ.get("OPENAI_API_KEY")

    # Cohere models
    if "cohere" in model_lower:
        return os.environ.get("COHERE_API_KEY")

    # Azure models
    if "azure" in model_lower:
        return os.environ.get("AZURE_OPENAI_API_KEY")

    # Generic fallback
    return os.environ.get("LLM_API_KEY")


def get_provider_from_model(model: str) -> str:
    """
    Detect the provider from a model name.

    Args:
        model: Model name

    Returns:
        Provider name: "anthropic", "openai", "cohere", "azure", or "unknown"
    """
    model_lower = model.lower()

    if any(x in model_lower for x in ["anthropic", "claude"]):
        return "anthropic"
    if any(x in model_lower for x in ["openai", "gpt", "o1", "o3"]):
        return "openai"
    if "cohere" in model_lower:
        return "cohere"
    if "azure" in model_lower:
        return "azure"

    return "unknown"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AuthMethod",
    "SubscriptionAuthContext",
    "configure_claude_auth",
    "restore_api_key",
    "check_claude_subscription",
    "get_api_key_for_model",
    "get_provider_from_model",
]
