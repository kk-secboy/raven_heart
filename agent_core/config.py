"""Configuration models for provider-neutral agents."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuntimeBudget:
    max_iterations: int = 12
    max_prompt_bytes: int = 128 * 1024
    max_timeline_bytes: int = 64 * 1024
    max_tool_result_bytes: int = 32 * 1024
    max_cost_usd: float | None = None


@dataclass(frozen=True)
class CapabilitySet:
    actions: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    memory_enabled: bool = True
    human_in_loop_enabled: bool = True


@dataclass(frozen=True)
class AgentProfile:
    name: str
    model: str = ""
    instructions: str = ""
    capabilities: CapabilitySet = field(default_factory=CapabilitySet)
    budget: RuntimeBudget = field(default_factory=RuntimeBudget)

