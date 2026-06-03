"""Planner contracts for plan-and-execute agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from uuid import uuid4


PlanStepStatus = Literal["pending", "running", "completed", "failed", "skipped"]
TERMINAL_PLAN_STEP_STATUSES = frozenset({"completed", "failed", "skipped"})


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    goal: str
    status: PlanStepStatus = "pending"
    depends_on: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "goal": self.goal,
            "status": self.status,
            "depends_on": list(self.depends_on),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Plan:
    plan_id: str
    goal: str
    steps: tuple[PlanStep, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def step(self, step_id: str) -> PlanStep | None:
        for item in self.steps:
            if item.step_id == step_id:
                return item
        return None

    def ready_steps(self) -> tuple[PlanStep, ...]:
        completed = {item.step_id for item in self.steps if item.status == "completed"}
        return tuple(
            item
            for item in self.steps
            if item.status == "pending" and all(dep in completed for dep in item.depends_on)
        )

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.steps:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def terminal(self) -> bool:
        return bool(self.steps) and all(item.status in TERMINAL_PLAN_STEP_STATUSES for item in self.steps)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-plan/v1",
            "plan_id": self.plan_id,
            "goal": self.goal,
            "terminal": self.terminal(),
            "ready_steps": [item.step_id for item in self.ready_steps()],
            "status_counts": self.status_counts(),
            "steps": [item.manifest() for item in self.steps],
            "metadata": dict(self.metadata),
        }


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


class InMemoryPlanner(PlannerPort):
    """Small deterministic planner for SDK tests and lightweight embeddings."""

    def __init__(self, plans: tuple[Plan, ...] = ()) -> None:
        self._plans: dict[str, Plan] = {plan.plan_id: plan for plan in plans}

    async def create_plan(self, goal: str, context: dict[str, Any] | None = None) -> Plan:
        text = str(goal or "").strip()
        if not text:
            raise ValueError("plan goal is required")
        context = dict(context or {})
        step_specs = _normalize_step_specs(context.get("steps"), fallback=text)
        steps = tuple(_step_from_spec(item, index=index) for index, item in enumerate(step_specs))
        plan = Plan(
            plan_id=str(context.get("plan_id") or uuid4().hex),
            goal=text,
            steps=steps,
            metadata={key: value for key, value in context.items() if key not in {"steps", "plan_id"}},
        )
        self._plans[plan.plan_id] = plan
        return plan

    async def update_plan(self, update: PlanUpdate) -> Plan:
        current = self.get(update.plan_id)
        if not update.step_id:
            metadata = {**current.metadata, **dict(update.metadata)}
            plan = Plan(
                plan_id=current.plan_id,
                goal=current.goal,
                steps=current.steps,
                metadata={**metadata, "note": update.note} if update.note else metadata,
            )
            self._plans[plan.plan_id] = plan
            return plan
        steps = []
        found = False
        for step in current.steps:
            if step.step_id != update.step_id:
                steps.append(step)
                continue
            found = True
            status = _step_status(str(update.status)) if update.status else step.status
            steps.append(
                PlanStep(
                    step_id=step.step_id,
                    goal=step.goal,
                    status=status,
                    depends_on=step.depends_on,
                    metadata={
                        **step.metadata,
                        **dict(update.metadata),
                        **({"note": update.note} if update.note else {}),
                    },
                )
            )
        if not found:
            raise KeyError(f"unknown plan step: {update.step_id}")
        plan = Plan(
            plan_id=current.plan_id,
            goal=current.goal,
            steps=tuple(steps),
            metadata=dict(current.metadata),
        )
        self._plans[plan.plan_id] = plan
        return plan

    def get(self, plan_id: str) -> Plan:
        try:
            return self._plans[plan_id]
        except KeyError as exc:
            raise KeyError(f"unknown plan: {plan_id}") from exc

    def plans(self) -> tuple[Plan, ...]:
        return tuple(sorted(self._plans.values(), key=lambda item: item.plan_id))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-in-memory-planner/v1",
            "plans": [plan.manifest() for plan in self.plans()],
        }


def _normalize_step_specs(value: Any, *, fallback: str) -> tuple[Any, ...]:
    if value is None:
        return (fallback,)
    if isinstance(value, (str, bytes)):
        return (value,)
    try:
        items = tuple(value)
    except TypeError:
        return (value,)
    return items or (fallback,)


def _step_from_spec(spec: Any, *, index: int) -> PlanStep:
    if isinstance(spec, dict):
        goal = str(spec.get("goal") or spec.get("description") or "").strip()
        if not goal:
            raise ValueError(f"plan step {index + 1} goal is required")
        return PlanStep(
            step_id=str(spec.get("step_id") or f"step-{index + 1}"),
            goal=goal,
            status=_step_status(str(spec.get("status") or "pending")),
            depends_on=tuple(str(item) for item in spec.get("depends_on") or ()),
            metadata=dict(spec.get("metadata") or {}),
        )
    goal = str(spec).strip()
    if not goal:
        raise ValueError(f"plan step {index + 1} goal is required")
    return PlanStep(step_id=f"step-{index + 1}", goal=goal)


def _step_status(value: str) -> PlanStepStatus:
    if value not in {"pending", "running", *TERMINAL_PLAN_STEP_STATUSES}:
        raise ValueError(f"invalid plan step status: {value}")
    return value  # type: ignore[return-value]

