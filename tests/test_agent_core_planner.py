from __future__ import annotations

import pytest

from agent_core import (
    InMemoryPlanner,
    InMemoryPlannerStore,
    MarkdownPlannerStore,
    PersistentPlanner,
    PlanExecutor,
    PlanUpdate,
    SQLitePlannerStore,
)
from agent_core.config import AgentProfile
from agent_core.runner import AgentSession, AgentSessionManager
from agent_core.testing import MockLLMProvider, MockToolRuntime


@pytest.mark.asyncio
async def test_in_memory_planner_tracks_dependencies_and_manifest() -> None:
    planner = InMemoryPlanner()

    plan = await planner.create_plan(
        "ship agent sdk",
        {
            "plan_id": "plan-1",
            "request_id": "r1",
            "steps": [
                {"step_id": "audit", "goal": "Audit current core"},
                {
                    "step_id": "implement",
                    "goal": "Implement missing contracts",
                    "depends_on": ["audit"],
                    "metadata": {"area": "planner"},
                },
            ],
        },
    )

    assert plan.plan_id == "plan-1"
    assert plan.metadata == {"request_id": "r1"}
    assert [item.step_id for item in plan.ready_steps()] == ["audit"]
    assert plan.status_counts() == {"pending": 2}

    updated = await planner.update_plan(PlanUpdate("plan-1", "audit", status="completed"))

    assert [item.step_id for item in updated.ready_steps()] == ["implement"]
    assert updated.status_counts() == {"completed": 1, "pending": 1}
    assert updated.manifest()["ready_steps"] == ["implement"]
    assert updated.manifest()["steps"][1]["metadata"] == {"area": "planner"}
    assert planner.manifest()["plans"][0]["plan_id"] == "plan-1"


@pytest.mark.asyncio
async def test_in_memory_planner_updates_plan_metadata_without_step() -> None:
    planner = InMemoryPlanner()
    plan = await planner.create_plan("single goal", {"plan_id": "plan-1"})

    updated = await planner.update_plan(
        PlanUpdate(plan.plan_id, metadata={"owner": "runtime"}, note="paused")
    )

    assert updated.metadata == {"owner": "runtime", "note": "paused"}
    assert updated.steps[0].goal == "single goal"


@pytest.mark.asyncio
async def test_in_memory_planner_treats_string_steps_as_single_step() -> None:
    planner = InMemoryPlanner()

    plan = await planner.create_plan("goal", {"plan_id": "plan-1", "steps": "one explicit step"})

    assert len(plan.steps) == 1
    assert plan.steps[0].goal == "one explicit step"


@pytest.mark.asyncio
async def test_in_memory_planner_rejects_invalid_inputs() -> None:
    planner = InMemoryPlanner()

    with pytest.raises(ValueError, match="plan goal is required"):
        await planner.create_plan("")

    with pytest.raises(ValueError, match="plan step 1 goal is required"):
        await planner.create_plan("goal", {"steps": [{"step_id": "missing-goal"}]})

    plan = await planner.create_plan("goal", {"plan_id": "plan-1"})

    with pytest.raises(ValueError, match="invalid plan step status"):
        await planner.update_plan(PlanUpdate(plan.plan_id, "step-1", status="bogus"))  # type: ignore[arg-type]

    with pytest.raises(KeyError, match="unknown plan step"):
        await planner.update_plan(PlanUpdate(plan.plan_id, "missing", status="completed"))

    with pytest.raises(KeyError, match="unknown plan"):
        planner.get("missing")


@pytest.mark.asyncio
async def test_persistent_planner_uses_in_memory_store_manifest() -> None:
    planner = PersistentPlanner(InMemoryPlannerStore())

    plan = await planner.create_plan(
        "persistent goal",
        {"plan_id": "plan-1", "steps": [{"step_id": "s1", "goal": "step"}]},
    )
    updated = await planner.update_plan(PlanUpdate(plan.plan_id, "s1", status="completed"))

    assert updated.status_counts() == {"completed": 1}
    assert planner.get("plan-1").steps[0].status == "completed"
    assert planner.manifest()["store"]["schema_version"] == "agent-core-in-memory-planner-store/v1"


@pytest.mark.asyncio
async def test_sqlite_planner_store_persists_plans_across_instances(tmp_path) -> None:
    path = tmp_path / "plans.sqlite"
    first = PersistentPlanner(SQLitePlannerStore(path))

    plan = await first.create_plan("sqlite goal", {"plan_id": "plan-1", "steps": ["step"]})
    await first.update_plan(PlanUpdate(plan.plan_id, "step-1", status="completed"))

    second = PersistentPlanner(SQLitePlannerStore(path))
    restored = second.get("plan-1")

    assert restored.goal == "sqlite goal"
    assert restored.steps[0].status == "completed"
    assert second.manifest()["store"]["schema_version"] == "agent-core-sqlite-planner-store/v1"


@pytest.mark.asyncio
async def test_markdown_planner_store_persists_without_visible_goal_payload(tmp_path) -> None:
    path = tmp_path / "plans.md"
    first = PersistentPlanner(MarkdownPlannerStore(path))

    plan = await first.create_plan(
        "sensitive goal",
        {"plan_id": "plan-1", "steps": [{"step_id": "s1", "goal": "sensitive step"}]},
    )
    await first.update_plan(PlanUpdate(plan.plan_id, "s1", status="completed"))
    text = path.read_text(encoding="utf-8")
    second = PersistentPlanner(MarkdownPlannerStore(path))

    assert second.get("plan-1").steps[0].status == "completed"
    assert "sensitive goal" not in text
    assert "sensitive step" not in text
    assert "<!-- planner-record " in text


@pytest.mark.asyncio
async def test_plan_executor_runs_ready_steps_and_updates_dependencies() -> None:
    planner = InMemoryPlanner()
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="worker"),
            provider=MockLLMProvider(
                [
                    {"action": "finish", "arguments": {"output": "audit done"}},
                    {"action": "finish", "arguments": {"output": "implement done"}},
                ]
            ),
            tools=MockToolRuntime(),
        )
    )
    executor = PlanExecutor(planner=planner, manager=manager, default_session="worker")

    report = await executor.execute(
        "ship sdk",
        context={
            "plan_id": "plan-1",
            "steps": [
                {"step_id": "audit", "goal": "Audit core"},
                {"step_id": "implement", "goal": "Implement core", "depends_on": ["audit"]},
            ],
        },
    )

    assert report.status == "completed"
    assert [step.step_id for step in report.steps] == ["audit", "implement"]
    assert report.plan.status_counts() == {"completed": 2}
    assert {run.metadata["step_id"] for run in manager.runs()} == {"audit", "implement"}
    assert executor.manifest()["reports"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_plan_executor_records_step_timeline_baseline_and_diff() -> None:
    planner = InMemoryPlanner()
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="worker"),
            provider=MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}]),
            tools=MockToolRuntime(),
        )
    )

    report = await PlanExecutor(
        planner=planner,
        manager=manager,
        default_session="worker",
    ).execute(
        "ship sdk",
        context={"plan_id": "plan-1", "steps": [{"step_id": "audit", "goal": "Audit core"}]},
    )

    assert report.status == "completed"
    metadata = report.steps[0].metadata
    assert metadata["timeline_baseline"]["schema_version"] == "agent-core-timeline-cursor/v1"
    assert metadata["timeline_baseline"]["last_item_id"] == ""
    diff = metadata["timeline_diff"]
    assert diff["schema_version"] == "agent-core-timeline-diff/v1"
    assert diff["cursor"] == metadata["timeline_baseline"]
    assert diff["item_count"] >= 3
    assert {"task", "model", "final"}.issubset(set(diff["kinds"]))
    assert metadata["timeline_diff_item_count"] == diff["item_count"]
    assert "final" in metadata["timeline_diff_kinds"]
    planned_step = report.plan.step("audit")
    assert planned_step is not None
    assert planned_step.metadata["timeline_diff"]["item_count"] == diff["item_count"]


@pytest.mark.asyncio
async def test_plan_executor_uses_step_session_metadata() -> None:
    planner = InMemoryPlanner()
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="default"),
            provider=MockLLMProvider([{"action": "finish", "arguments": {"output": "default"}}]),
            tools=MockToolRuntime(),
        )
    )
    manager.register(
        AgentSession(
            profile=AgentProfile(name="specialist"),
            provider=MockLLMProvider([{"action": "finish", "arguments": {"output": "specialist"}}]),
            tools=MockToolRuntime(),
        )
    )

    report = await PlanExecutor(
        planner=planner,
        manager=manager,
        default_session="default",
    ).execute(
        "delegate",
        context={
            "steps": [
                {
                    "step_id": "special",
                    "goal": "Use specialist",
                    "metadata": {"session_name": "specialist"},
                }
            ]
        },
    )

    assert report.status == "completed"
    assert report.steps[0].session_name == "specialist"
    assert report.steps[0].output == "specialist"


@pytest.mark.asyncio
async def test_plan_executor_marks_failed_step_when_agent_run_fails() -> None:
    planner = InMemoryPlanner()
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="worker"),
            provider=MockLLMProvider([{"action": "finish", "arguments": {"output": "ok"}}]),
            tools=MockToolRuntime(),
        )
    )

    failed = await PlanExecutor(
        planner=planner,
        manager=manager,
        default_session="missing",
    ).execute("fail plan", context={"steps": ["single"], "plan_id": "p"})

    assert failed.status == "failed"
    assert failed.steps[0].status == "failed"
    assert "unknown session" in failed.steps[0].error
    assert failed.plan.status_counts() == {"failed": 1}


@pytest.mark.asyncio
async def test_plan_executor_reports_blocked_plan_without_ready_steps() -> None:
    planner = InMemoryPlanner()
    manager = AgentSessionManager()
    manager.register(
        AgentSession(
            profile=AgentProfile(name="worker"),
            provider=MockLLMProvider([]),
            tools=MockToolRuntime(),
        )
    )

    report = await PlanExecutor(
        planner=planner,
        manager=manager,
        default_session="worker",
    ).execute(
        "blocked",
        context={
            "plan_id": "blocked",
            "steps": [
                {
                    "step_id": "pending",
                    "goal": "Wait for missing dependency",
                    "depends_on": ["missing"],
                }
            ],
        },
    )

    assert report.status == "blocked"
    assert report.metadata["reason"] == "no ready pending steps"
