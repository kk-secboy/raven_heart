"""Error types for the provider-neutral agent core."""

from __future__ import annotations


class AgentCoreError(Exception):
    """Base class for all agent core failures."""


class ProviderError(AgentCoreError):
    """LLM provider returned an error or unusable response."""


class ActionError(AgentCoreError):
    """Model output could not be parsed or validated as an action."""


class ToolError(AgentCoreError):
    """Tool execution failed at the core protocol boundary."""


class HarnessError(AgentCoreError):
    """Run lifecycle, checkpoint, or resume state failed."""


class ResumeError(HarnessError):
    """A checkpoint could not be resumed safely."""


class PolicyError(AgentCoreError):
    """A policy decision denied or interrupted an agent operation."""

