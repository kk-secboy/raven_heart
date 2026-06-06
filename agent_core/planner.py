"""Planner contracts for plan-and-execute agents."""

from __future__ import annotations

import base64
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from agent_core.backends import storage_backend_manifest
from agent_core.runner import AgentRunOutcome, AgentRunRequest, AgentSessionManager


PlanStepStatus = Literal["pending", "running", "completed", "failed", "skipped"]
TERMINAL_PLAN_STEP_STATUSES = frozenset({"completed", "failed", "skipped"})
PlanExecutionStatus = Literal["completed", "failed", "blocked", "max_steps"]


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


class PlannerStorePort(Protocol):
    """Persistence boundary for planner state."""

    def save(self, plan: Plan) -> None:
        """Persist or replace one plan."""

    def load(self, plan_id: str) -> Plan | None:
        """Load one plan by id."""

    def list(self) -> tuple[Plan, ...]:
        """Return all persisted plans."""

    def delete(self, plan_id: str) -> bool:
        """Delete one plan."""

    def manifest(self) -> dict[str, Any]:
        """Return prompt-safe store metadata."""


class InMemoryPlannerStore:
    def __init__(self, plans: tuple[Plan, ...] = ()) -> None:
        self._plans = {plan.plan_id: plan for plan in plans}

    def save(self, plan: Plan) -> None:
        self._plans[plan.plan_id] = plan

    def load(self, plan_id: str) -> Plan | None:
        return self._plans.get(plan_id)

    def list(self) -> tuple[Plan, ...]:
        return tuple(sorted(self._plans.values(), key=lambda item: item.plan_id))

    def delete(self, plan_id: str) -> bool:
        return self._plans.pop(plan_id, None) is not None

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-in-memory-planner-store/v1",
            "backend": storage_backend_manifest(role="planner_state", kind="in_memory"),
            "plan_count": len(self._plans),
        }


class SQLitePlannerStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def save(self, plan: Plan) -> None:
        raw = json.dumps(plan.manifest(), ensure_ascii=False, sort_keys=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO plans(plan_id, manifest_json)
                VALUES(?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    manifest_json=excluded.manifest_json
                """,
                (plan.plan_id, raw),
            )

    def load(self, plan_id: str) -> Plan | None:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT manifest_json FROM plans WHERE plan_id = ?",
                (plan_id,),
            ).fetchone()
        if row is None:
            return None
        return _plan_from_manifest(json.loads(str(row[0] or "{}")))

    def list(self) -> tuple[Plan, ...]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT manifest_json FROM plans ORDER BY plan_id ASC").fetchall()
        return tuple(_plan_from_manifest(json.loads(str(row[0] or "{}"))) for row in rows)

    def delete(self, plan_id: str) -> bool:
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute("DELETE FROM plans WHERE plan_id = ?", (plan_id,))
            return cursor.rowcount > 0

    def manifest(self) -> dict[str, Any]:
        with sqlite3.connect(self.path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM plans").fetchone()[0]
        return {
            "schema_version": "agent-core-sqlite-planner-store/v1",
            "backend": storage_backend_manifest(
                role="planner_state",
                kind="sqlite",
                location=str(self.path),
                capabilities=("save", "load", "list", "delete"),
            ),
            "path": str(self.path),
            "plan_count": int(count),
        }

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    manifest_json TEXT NOT NULL
                )
                """
            )


class MarkdownPlannerStore:
    _START = "<!-- planner-record "
    _END = " -->"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("# Planner Store\n\n", encoding="utf-8")

    def save(self, plan: Plan) -> None:
        plans = {item.plan_id: item for item in self.list()}
        plans[plan.plan_id] = plan
        self._write(tuple(sorted(plans.values(), key=lambda item: item.plan_id)))

    def load(self, plan_id: str) -> Plan | None:
        for plan in self.list():
            if plan.plan_id == plan_id:
                return plan
        return None

    def list(self) -> tuple[Plan, ...]:
        if not self.path.exists():
            return ()
        text = self.path.read_text(encoding="utf-8")
        plans = []
        for match in _PLAN_MARKDOWN_RE.finditer(text):
            plans.append(_plan_from_manifest(_decode_plan_payload(match.group(1))))
        return tuple(sorted(plans, key=lambda item: item.plan_id))

    def delete(self, plan_id: str) -> bool:
        current = self.list()
        plans = tuple(plan for plan in current if plan.plan_id != plan_id)
        existed = len(plans) != len(current)
        if existed:
            self._write(plans)
        return existed

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-markdown-planner-store/v1",
            "backend": storage_backend_manifest(
                role="planner_state",
                kind="markdown",
                location=str(self.path),
                capabilities=("save", "load", "list", "delete"),
            ),
            "path": str(self.path),
            "plan_count": len(self.list()),
        }

    def _write(self, plans: tuple[Plan, ...]) -> None:
        lines = ["# Planner Store", ""]
        for plan in plans:
            lines.extend(
                [
                    f"## Plan {plan.plan_id}",
                    "",
                    f"- terminal: {str(plan.terminal()).lower()}",
                    f"- step_count: {len(plan.steps)}",
                    f"{self._START}{_encode_plan_payload(plan.manifest())}{self._END}",
                    "",
                ]
            )
        self.path.write_text("\n".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class PlanExecutionStep:
    plan_id: str
    step_id: str
    session_name: str
    status: PlanStepStatus
    run_id: str = ""
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-plan-execution-step/v1",
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "session_name": self.session_name,
            "status": self.status,
            "run_id": self.run_id,
            "output_bytes": len(self.output.encode("utf-8")),
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PlanExecutionReport:
    plan: Plan
    status: PlanExecutionStatus
    steps: tuple[PlanExecutionStep, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-plan-execution-report/v1",
            "status": self.status,
            "plan": self.plan.manifest(),
            "steps": [step.manifest() for step in self.steps],
            "metadata": dict(self.metadata),
        }


class PlanExecutor:
    """Execute ready plan steps through `AgentSessionManager` sessions."""

    def __init__(
        self,
        *,
        planner: PlannerPort,
        manager: AgentSessionManager,
        default_session: str,
        max_steps: int = 32,
    ) -> None:
        if not default_session.strip():
            raise ValueError("default_session is required")
        self.planner = planner
        self.manager = manager
        self.default_session = default_session
        self.max_steps = max(1, max_steps)
        self.reports: list[PlanExecutionReport] = []

    async def execute(
        self,
        goal: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> PlanExecutionReport:
        plan = await self.planner.create_plan(goal, context)
        records: list[PlanExecutionStep] = []
        executed = 0
        while not plan.terminal():
            if executed >= self.max_steps:
                report = PlanExecutionReport(
                    plan=plan,
                    status="max_steps",
                    steps=tuple(records),
                    metadata={"max_steps": self.max_steps},
                )
                self.reports.append(report)
                return report
            ready = plan.ready_steps()
            if not ready:
                report = PlanExecutionReport(
                    plan=plan,
                    status="blocked",
                    steps=tuple(records),
                    metadata={"reason": "no ready pending steps"},
                )
                self.reports.append(report)
                return report
            for step in ready:
                if executed >= self.max_steps:
                    break
                plan = await self.planner.update_plan(PlanUpdate(plan.plan_id, step.step_id, status="running"))
                record = await self._execute_step(plan, step)
                records.append(record)
                executed += 1
                plan = await self.planner.update_plan(
                    PlanUpdate(
                        plan.plan_id,
                        step.step_id,
                        status=record.status,
                        note=record.error,
                        metadata={
                            "session_name": record.session_name,
                            "run_id": record.run_id,
                            **dict(record.metadata),
                        },
                    )
                )
                if record.status == "failed":
                    report = PlanExecutionReport(
                        plan=plan,
                        status="failed",
                        steps=tuple(records),
                        metadata={"failed_step_id": step.step_id},
                    )
                    self.reports.append(report)
                    return report
        report = PlanExecutionReport(plan=plan, status="completed", steps=tuple(records))
        self.reports.append(report)
        return report

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-plan-executor/v1",
            "default_session": self.default_session,
            "max_steps": self.max_steps,
            "reports": [report.manifest() for report in self.reports],
        }

    async def _execute_step(self, plan: Plan, step: PlanStep) -> PlanExecutionStep:
        session_name = str(step.metadata.get("session_name") or self.default_session)
        session = None
        timeline_cursor = None
        timeline_baseline: dict[str, Any] = {}
        try:
            session = self.manager.session(session_name)
            timeline_cursor = session.timeline.cursor()
            timeline_baseline = timeline_cursor.manifest()
            outcome = await self.manager.run(
                session_name,
                AgentRunRequest(
                    task=step.goal,
                    metadata={
                        "plan_id": plan.plan_id,
                        "plan_goal": plan.goal,
                        "step_id": step.step_id,
                        **dict(step.metadata),
                    },
                ),
            )
        except Exception as exc:
            metadata: dict[str, Any] = {}
            if session is not None and timeline_cursor is not None:
                metadata = _plan_step_timeline_metadata(
                    baseline=timeline_baseline,
                    diff=session.timeline.diff_since(timeline_cursor),
                )
            return PlanExecutionStep(
                plan_id=plan.plan_id,
                step_id=step.step_id,
                session_name=session_name,
                status="failed",
                error=str(exc),
                metadata=metadata,
            )
        record = _execution_step_from_outcome(plan, step, session_name, outcome)
        return _plan_step_with_timeline(
            record,
            baseline=timeline_baseline,
            diff=session.timeline.diff_since(timeline_cursor),
        )


class InMemoryPlanner(PlannerPort):
    """Small deterministic planner for SDK tests and lightweight embeddings."""

    def __init__(self, plans: tuple[Plan, ...] = ()) -> None:
        self._plans: dict[str, Plan] = {plan.plan_id: plan for plan in plans}

    async def create_plan(self, goal: str, context: dict[str, Any] | None = None) -> Plan:
        plan = _create_plan(goal, context)
        self._plans[plan.plan_id] = plan
        return plan

    async def update_plan(self, update: PlanUpdate) -> Plan:
        current = self.get(update.plan_id)
        plan = _apply_plan_update(current, update)
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


class PersistentPlanner(PlannerPort):
    """Planner backed by a pluggable `PlannerStorePort`."""

    def __init__(self, store: PlannerStorePort) -> None:
        self.store = store

    async def create_plan(self, goal: str, context: dict[str, Any] | None = None) -> Plan:
        plan = _create_plan(goal, context)
        self.store.save(plan)
        return plan

    async def update_plan(self, update: PlanUpdate) -> Plan:
        current = self.get(update.plan_id)
        plan = _apply_plan_update(current, update)
        self.store.save(plan)
        return plan

    def get(self, plan_id: str) -> Plan:
        plan = self.store.load(plan_id)
        if plan is None:
            raise KeyError(f"unknown plan: {plan_id}")
        return plan

    def plans(self) -> tuple[Plan, ...]:
        return self.store.list()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-persistent-planner/v1",
            "store": self.store.manifest(),
            "plans": [plan.manifest() for plan in self.plans()],
        }


def _create_plan(goal: str, context: dict[str, Any] | None = None) -> Plan:
    text = str(goal or "").strip()
    if not text:
        raise ValueError("plan goal is required")
    context = dict(context or {})
    step_specs = _normalize_step_specs(context.get("steps"), fallback=text)
    steps = tuple(_step_from_spec(item, index=index) for index, item in enumerate(step_specs))
    return Plan(
        plan_id=str(context.get("plan_id") or uuid4().hex),
        goal=text,
        steps=steps,
        metadata={key: value for key, value in context.items() if key not in {"steps", "plan_id"}},
    )


def _apply_plan_update(current: Plan, update: PlanUpdate) -> Plan:
    if not update.step_id:
        metadata = {**current.metadata, **dict(update.metadata)}
        return Plan(
            plan_id=current.plan_id,
            goal=current.goal,
            steps=current.steps,
            metadata={**metadata, "note": update.note} if update.note else metadata,
        )
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
    return Plan(
        plan_id=current.plan_id,
        goal=current.goal,
        steps=tuple(steps),
        metadata=dict(current.metadata),
    )


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


def _execution_step_from_outcome(
    plan: Plan,
    step: PlanStep,
    session_name: str,
    outcome: AgentRunOutcome,
) -> PlanExecutionStep:
    result = outcome.result
    status: PlanStepStatus = "completed" if result.status == "completed" else "failed"
    return PlanExecutionStep(
        plan_id=plan.plan_id,
        step_id=step.step_id,
        session_name=session_name,
        status=status,
        run_id=result.run_id,
        output=result.output,
        error="" if status == "completed" else result.output,
        metadata={
            "result_status": result.status,
            "iterations": result.iterations,
            "trace_run_id": outcome.trace_manifest.get("run", {}).get("run_id", ""),
        },
    )


def _plan_step_with_timeline(
    record: PlanExecutionStep,
    *,
    baseline: dict[str, Any],
    diff: Any,
) -> PlanExecutionStep:
    return PlanExecutionStep(
        plan_id=record.plan_id,
        step_id=record.step_id,
        session_name=record.session_name,
        status=record.status,
        run_id=record.run_id,
        output=record.output,
        error=record.error,
        metadata={
            **dict(record.metadata),
            **_plan_step_timeline_metadata(baseline=baseline, diff=diff),
        },
    )


def _plan_step_timeline_metadata(*, baseline: dict[str, Any], diff: Any) -> dict[str, Any]:
    manifest = diff.manifest()
    return {
        "timeline_baseline": dict(baseline),
        "timeline_diff": manifest,
        "timeline_diff_item_count": int(manifest.get("item_count") or 0),
        "timeline_diff_kinds": list(manifest.get("kinds") or ()),
    }


_PLAN_MARKDOWN_RE = re.compile(r"<!--\s*planner-record\s+([A-Za-z0-9+/=]+)\s*-->")


def _plan_from_manifest(manifest: dict[str, Any]) -> Plan:
    return Plan(
        plan_id=str(manifest.get("plan_id") or ""),
        goal=str(manifest.get("goal") or ""),
        steps=tuple(
            _plan_step_from_manifest(item)
            for item in manifest.get("steps", ())
            if isinstance(item, dict)
        ),
        metadata=dict(manifest.get("metadata") or {}),
    )


def _plan_step_from_manifest(manifest: dict[str, Any]) -> PlanStep:
    return PlanStep(
        step_id=str(manifest.get("step_id") or ""),
        goal=str(manifest.get("goal") or ""),
        status=_step_status(str(manifest.get("status") or "pending")),
        depends_on=tuple(str(item) for item in manifest.get("depends_on") or ()),
        metadata=dict(manifest.get("metadata") or {}),
    )


def _encode_plan_payload(manifest: dict[str, Any]) -> str:
    raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _decode_plan_payload(encoded: str) -> dict[str, Any]:
    raw = base64.b64decode(encoded.encode("ascii")).decode("utf-8")
    value = json.loads(raw)
    return dict(value) if isinstance(value, dict) else {}

