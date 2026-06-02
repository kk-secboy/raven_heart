from __future__ import annotations

import pytest

from agent_core.prompt import PromptBucketRole, PromptIR
from agent_core.reducer import DefaultContextReducer, ReducerRequest
from agent_core.timeline import TimelineBudget, TimelineStore


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

