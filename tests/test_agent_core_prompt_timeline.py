from __future__ import annotations

import pytest

from agent_core import apply_reduction_to_timeline
from agent_core.prompt import (
    PromptBucketBudgetPolicy,
    PromptBucketBudgetRule,
    PromptBucketRole,
    PromptIR,
    PromptTrimRule,
)
from agent_core.reducer import DefaultContextReducer, ReducerRequest
from agent_core.timeline import TimelineBudget, TimelineItem, TimelineStore


def test_prompt_ir_renders_yaklang_style_buckets_in_stable_order() -> None:
    prompt = PromptIR.from_parts(
        high_static="system rules",
        frozen="known facts",
        semi_dynamic_1="skills",
        semi_dynamic_2="tools",
        timeline_open="recent observation",
        dynamic="current task",
    )

    rendered = prompt.render()
    roles = [bucket["role"] for bucket in prompt.manifest()["buckets"]]

    assert roles == [
        "high_static",
        "frozen",
        "semi_dynamic_1",
        "semi_dynamic_2",
        "timeline_open",
        "dynamic",
    ]
    assert rendered.index('role="high_static"') < rendered.index('role="frozen"')
    assert rendered.index('role="timeline_open"') < rendered.index('role="dynamic"')


def test_prompt_dynamic_change_does_not_change_stable_bucket_hashes() -> None:
    first = PromptIR.from_parts(high_static="system", frozen="facts", dynamic="task one")
    second = PromptIR.from_parts(high_static="system", frozen="facts", dynamic="task two")

    first_manifest = first.manifest()["buckets_by_role"]
    second_manifest = second.manifest()["buckets_by_role"]

    assert (
        first_manifest[PromptBucketRole.HIGH_STATIC.value]["sha256"]
        == second_manifest[PromptBucketRole.HIGH_STATIC.value]["sha256"]
    )
    assert (
        first_manifest[PromptBucketRole.FROZEN.value]["sha256"]
        == second_manifest[PromptBucketRole.FROZEN.value]["sha256"]
    )
    assert (
        first_manifest[PromptBucketRole.DYNAMIC.value]["sha256"]
        != second_manifest[PromptBucketRole.DYNAMIC.value]["sha256"]
    )


def test_prompt_ir_trims_dynamic_buckets_before_stable_prefix() -> None:
    prompt = PromptIR.from_parts(
        high_static="stable rules",
        frozen="capability " + ("c" * 120),
        semi_dynamic_1="skills " + ("s" * 120),
        semi_dynamic_2="schema " + ("x" * 120),
        timeline_open="old observation " + ("o" * 1200) + " latest observation",
        dynamic="current task",
    )

    trimmed = prompt.trim_to_budget(900)
    manifest = trimmed.manifest()
    trim = manifest["metadata"]["trim"]

    assert manifest["prompt_bytes"] <= 900
    assert trimmed.bucket(PromptBucketRole.HIGH_STATIC).content == "stable rules"
    assert trimmed.bucket(PromptBucketRole.DYNAMIC).content == "current task"
    assert trimmed.bucket(PromptBucketRole.TIMELINE_OPEN).metadata["trimmed"] is True
    assert trim["trimmed_roles"][0]["role"] == PromptBucketRole.TIMELINE_OPEN.value
    assert trim["plan"]["schema_version"] == "agent-core-prompt-trim-plan/v1"
    assert trim["steps"][0]["reason"] == "trim volatile timeline context first"
    assert trim["original_bytes"] > trim["final_bytes"]
    assert "[...trimmed...]" in trimmed.bucket(PromptBucketRole.TIMELINE_OPEN).content


def test_prompt_trim_plan_exports_semantic_rules_before_trimming() -> None:
    prompt = PromptIR.from_parts(
        high_static="stable rules",
        timeline_open="old observation " + ("o" * 500),
        dynamic="current task",
    )

    plan = prompt.trim_plan(400)
    manifest = plan.manifest()

    assert manifest["schema_version"] == "agent-core-prompt-trim-plan/v1"
    assert manifest["target_bytes"] == 400
    assert manifest["rules"][0]["role"] == PromptBucketRole.TIMELINE_OPEN.value
    assert manifest["rules"][-1]["role"] == PromptBucketRole.HIGH_STATIC.value
    assert manifest["rules"][-1]["protected"] is True
    assert manifest["metadata"]["bucket_roles"] == [
        "high_static",
        "frozen",
        "semi_dynamic_1",
        "semi_dynamic_2",
        "timeline_open",
        "dynamic",
    ]


def test_prompt_trim_rules_can_protect_semantic_buckets() -> None:
    prompt = PromptIR.from_parts(
        high_static="stable rules " + ("s" * 200),
        timeline_open="timeline " + ("t" * 700),
        dynamic="current task " + ("d" * 200),
    )
    rules = (
        PromptTrimRule(
            role=PromptBucketRole.TIMELINE_OPEN,
            order=0,
            min_keep_bytes=0,
            preserve_head_ratio=0.2,
            reason="timeline can be aggressively reduced",
        ),
        PromptTrimRule(
            role=PromptBucketRole.DYNAMIC,
            order=1,
            protected=True,
            reason="current user task is protected",
        ),
    )

    trimmed = prompt.trim_to_budget(650, rules=rules)
    trim = trimmed.manifest()["metadata"]["trim"]

    assert trimmed.bucket(PromptBucketRole.DYNAMIC).content.startswith("current task")
    assert trimmed.bucket(PromptBucketRole.DYNAMIC).metadata == {}
    assert trim["steps"][0]["role"] == PromptBucketRole.TIMELINE_OPEN.value
    assert all(step["role"] != PromptBucketRole.DYNAMIC.value for step in trim["trimmed_roles"])


def test_prompt_bucket_budget_policy_trims_declared_semantic_buckets() -> None:
    prompt = PromptIR.from_parts(
        high_static="system rules",
        frozen="tools " + ("t" * 200),
        semi_dynamic_1="memory " + ("m" * 500),
        timeline_open="timeline " + ("o" * 100),
        dynamic="current task " + ("d" * 200),
    )
    policy = PromptBucketBudgetPolicy(
        rules=(
            PromptBucketBudgetRule(
                role=PromptBucketRole.SEMI_DYNAMIC_1,
                max_bytes=160,
                min_keep_bytes=80,
                preserve_head_ratio=0.25,
                reason="memory recall must fit its semantic bucket",
            ),
            PromptBucketBudgetRule(
                role=PromptBucketRole.DYNAMIC,
                max_bytes=80,
                protected=True,
                reason="current task remains authoritative",
            ),
        )
    )

    budgeted = policy.apply(prompt)
    manifest = budgeted.manifest()
    budget = manifest["metadata"]["bucket_budget"]

    assert budget["schema_version"] == "agent-core-prompt-bucket-budget-result/v1"
    assert budget["trimmed_count"] == 1
    assert budget["protected_count"] == 1
    assert budget["decisions"][2]["role"] == PromptBucketRole.SEMI_DYNAMIC_1.value
    assert budget["decisions"][2]["status"] == "trimmed"
    assert budget["decisions"][-1]["status"] == "protected"
    assert "[...bucket budget trimmed...]" in budgeted.bucket(
        PromptBucketRole.SEMI_DYNAMIC_1
    ).content
    assert budgeted.bucket(PromptBucketRole.DYNAMIC).content.startswith("current task")
    assert budgeted.bucket(PromptBucketRole.SEMI_DYNAMIC_1).metadata[
        "bucket_budget_trimmed"
    ] is True


def test_timeline_splits_frozen_and_open_when_over_budget() -> None:
    timeline = TimelineStore()
    for index in range(8):
        timeline.add(f"observation {index} " + ("x" * 80), kind="tool")

    view = timeline.view(TimelineBudget(max_bytes=220, recent_keep_ratio=0.25))

    assert view.frozen_items
    assert view.open_items
    assert view.open_items[-1].content.startswith("observation 7")
    assert view.frozen_items[0].content.startswith("observation 0")


@pytest.mark.asyncio
async def test_default_reducer_keeps_recent_window_and_compresses_old_items() -> None:
    timeline = TimelineStore()
    for index in range(6):
        timeline.add(f"step {index} " + ("payload " * 20), kind="observation")

    reducer = DefaultContextReducer()
    result = await reducer.reduce(
        ReducerRequest(items=timeline.items, max_bytes=260, recent_keep_ratio=0.25)
    )

    assert result.compressed_head
    assert result.archive_refs
    assert result.retained_items
    assert result.retained_items[-1].content.startswith("step 5")


@pytest.mark.asyncio
async def test_default_reducer_exports_manifest_and_applies_to_timeline() -> None:
    pinned = TimelineItem("pinned fact " + ("p" * 80), kind="fact", pinned=True)
    timeline = TimelineStore([pinned])
    for index in range(4):
        timeline.add(f"old step {index} " + ("payload " * 20), kind="observation")
    latest = timeline.add("latest observation " + ("z" * 80), kind="observation")

    reducer = DefaultContextReducer()
    request = ReducerRequest(items=timeline.items, max_bytes=220, recent_keep_ratio=0.2)
    result = await reducer.reduce(request)
    view = apply_reduction_to_timeline(timeline, result)
    manifest = result.manifest()

    active_ids = {item.item_id for item in timeline.items if not item.deleted}
    assert pinned.item_id in active_ids
    assert latest.item_id in active_ids
    assert result.metadata["compressed_item_count"] > 0
    assert manifest["schema_version"] == "agent-core-reducer-result/v1"
    assert manifest["retained_item_ids"] == [item.item_id for item in result.retained_items]
    assert timeline.compressed_head
    assert timeline.archive_refs == list(result.archive_refs)
    assert view.compressed_head == timeline.compressed_head


@pytest.mark.asyncio
async def test_reducer_request_normalizes_budget_values() -> None:
    timeline = TimelineStore()
    item = timeline.add("large " + ("x" * 200), kind="observation")

    result = await DefaultContextReducer().reduce(
        ReducerRequest(items=(item,), max_bytes=0, recent_keep_ratio=5.0)
    )

    assert result.metadata["max_bytes"] == 1
    assert result.metadata["recent_keep_ratio"] == 1.0

