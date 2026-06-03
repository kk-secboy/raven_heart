from __future__ import annotations

import pytest

from agent_core import apply_reduction_to_timeline
from agent_core.prompt import PromptBucketRole, PromptIR
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
    assert trim["original_bytes"] > trim["final_bytes"]
    assert "[...trimmed...]" in trimmed.bucket(PromptBucketRole.TIMELINE_OPEN).content


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

