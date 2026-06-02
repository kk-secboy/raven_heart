"""Planner protocol placeholders for plan-and-execute agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


PlanStepStatus = Literal["pending", "running", "completed", "failed", "skipped"]


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    goal: str
    status: PlanStepStatus = "pending"
    depends_on: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Plan:
    plan_id: str
    goal: str
    steps: tuple[PlanStep, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanUpdate:
    plan_id: str
    step_id: str = ""
    status: PlanStepStatus | None = None
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class PlannerPort(Protocol):
    async def create_plan(self, goal: str, context: dict[str, Any] | None = None) -> Plan:
        """Create a plan for a goal."""

    async def update_plan(self, update: PlanUpdate) -> Plan:
        """Apply a plan update and return the latest plan."""

