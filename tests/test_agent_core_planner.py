from __future__ import annotations

import pytest

from agent_core import InMemoryPlanner, PlanUpdate


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
