from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from agent_core.actions import ParsedAction
from agent_core.context import ContextInjection
from agent_core.prompt import PromptBucketRole
from agent_core.timeline import TimelineStore
from agent_core.tools import ToolResult
from agent_core.turn_runtime import (
    INTENT_SHIFT_DRIFT,
    INTENT_SHIFT_NONE,
    INTENT_SHIFT_PIVOT,
    PERCEPTION_TRIGGER_FORCED,
    PERCEPTION_TRIGGER_POST_ACTION,
    PERCEPTION_TRIGGER_SPIN_DETECTED,
    PerceptionInput,
    PerceptionRuntimeConfig,
    PerceptionState,
    ProviderBackedPerceptionEvaluator,
    ToolResultEvent,
    YaklangStylePerceptionController,
)
from agent_core.testing import MockLLMProvider


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class _QueuedPerceptionEvaluator:
    def __init__(self, states: list[PerceptionState]) -> None:
        self.states = list(states)
        self.inputs: list[PerceptionInput] = []

    async def evaluate(self, request: PerceptionInput) -> PerceptionState:
        self.inputs.append(request)
        if not self.states:
            return PerceptionState(
                topics=("fallback",),
                keywords=("fallback",),
                summary="fallback",
                changed=True,
                confidence=0.5,
                intent_shift=INTENT_SHIFT_PIVOT,
            )
        return self.states.pop(0)


class _BlockingPerceptionEvaluator:
    def __init__(self, state: PerceptionState) -> None:
        self.state = state
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def evaluate(self, request: PerceptionInput) -> PerceptionState:
        self.started.set()
        await self.release.wait()
        return self.state


def _event(iteration: int, *, force: bool = False, loop_guard: bool = False) -> ToolResultEvent:
    timeline = TimelineStore()
    timeline.add(f"tool result iteration {iteration}", kind="tool")
    metadata = {"loop_guard": {"stalled": True}} if loop_guard else {}
    return ToolResultEvent(
        task="investigate auth callback failure",
        iteration=iteration,
        run_id="run",
        turn_id=f"turn-{iteration}",
        tool_result=ToolResult(
            call_id=f"call-{iteration}",
            tool_name="inspect_http",
            content=f"csrf callback failed on iteration {iteration}",
        ),
        action=ParsedAction(name="call_tool", arguments={"tool_name": "inspect_http"}),
        timeline_diff=timeline.diff_since(),
        force_perception=force,
        metadata=metadata,
    )


def _sync_perception_config(**overrides) -> PerceptionRuntimeConfig:
    return PerceptionRuntimeConfig(sync_triggers=True, **overrides)


@pytest.mark.asyncio
async def test_yaklang_style_perception_triggers_on_first_post_action_then_respects_interval() -> None:
    clock = _Clock()
    evaluator = _QueuedPerceptionEvaluator(
        [
            PerceptionState(
                topics=("auth callback", "csrf"),
                keywords=("auth", "csrf"),
                summary="auth callback csrf failure",
                changed=True,
                confidence=0.9,
                intent_shift=INTENT_SHIFT_PIVOT,
            )
        ]
    )
    controller = YaklangStylePerceptionController(
        evaluator=evaluator,
        now=clock.now,
        config=_sync_perception_config(),
    )

    injections = await controller.after_tool_result(_event(0))

    assert len(evaluator.inputs) == 1
    assert evaluator.inputs[0].trigger == PERCEPTION_TRIGGER_POST_ACTION
    assert len(injections) == 1
    assert isinstance(injections[0], ContextInjection)
    assert injections[0].target is PromptBucketRole.TIMELINE_OPEN
    assert "auth callback csrf failure" in injections[0].content
    observation = controller.consume_observation()
    assert observation["schema_version"] == "agent-core-perception-observation/v1"
    assert observation["updated"] is True
    assert observation["state"]["summary"] == "auth callback csrf failure"
    downstream = controller.consume_downstream_refresh()
    assert downstream["reason"] == "forced_or_intent_pivot"
    assert "memory_recall" in downstream["intents"]
    assert "capability_search" in downstream["intents"]
    assert "auth callback csrf failure" in downstream["query"]
    for iteration in range(1, 4):
        assert await controller.after_tool_result(_event(iteration)) == ()
    assert len(evaluator.inputs) == 1


@pytest.mark.asyncio
async def test_yaklang_style_perception_defaults_to_async_triggering() -> None:
    evaluator = _QueuedPerceptionEvaluator(
        [
            PerceptionState(
                topics=("async default",),
                keywords=("async",),
                summary="async default perception",
                changed=True,
                confidence=0.8,
                intent_shift=INTENT_SHIFT_PIVOT,
            )
        ]
    )
    controller = YaklangStylePerceptionController(evaluator=evaluator)

    injections = await controller.after_tool_result(_event(4))

    assert injections == ()
    assert controller.manifest()["config"]["sync_triggers"] is False
    for _ in range(10):
        await asyncio.sleep(0)
        if evaluator.inputs:
            break
    assert evaluator.inputs[0].trigger == PERCEPTION_TRIGGER_POST_ACTION
    assert controller.consume_observation()["state"]["summary"] == "async default perception"
    assert controller.consume_downstream_refresh()["reason"] == "forced_or_intent_pivot"


@pytest.mark.asyncio
async def test_yaklang_style_perception_respects_time_interval_after_update() -> None:
    clock = _Clock()
    evaluator = _QueuedPerceptionEvaluator(
        [
            PerceptionState(
                topics=("phase one",),
                keywords=("phase",),
                summary="phase one",
                changed=True,
                confidence=0.8,
                intent_shift=INTENT_SHIFT_PIVOT,
            ),
            PerceptionState(
                topics=("phase two",),
                keywords=("phase", "two"),
                summary="phase two",
                changed=True,
                confidence=0.8,
                intent_shift=INTENT_SHIFT_PIVOT,
            ),
        ]
    )
    controller = YaklangStylePerceptionController(
        evaluator=evaluator,
        now=clock.now,
        config=_sync_perception_config(),
    )

    await controller.after_tool_result(_event(4))
    assert len(evaluator.inputs) == 1
    assert await controller.after_tool_result(_event(8)) == ()
    assert len(evaluator.inputs) == 1
    assert controller.manifest()["last_trigger"]["reason"] == "time_interval_not_reached"

    clock.advance(121)
    await controller.after_tool_result(_event(8))
    assert len(evaluator.inputs) == 2


@pytest.mark.asyncio
async def test_yaklang_style_perception_forced_bypasses_interval_but_spin_uses_pivot_gate() -> None:
    clock = _Clock()
    evaluator = _QueuedPerceptionEvaluator(
        [
            PerceptionState(
                topics=("forced",),
                keywords=("forced",),
                summary="forced update",
                changed=False,
                confidence=0.7,
                intent_shift=INTENT_SHIFT_NONE,
            ),
            PerceptionState(
                topics=("same",),
                keywords=("same",),
                summary="spin but no pivot",
                changed=False,
                confidence=0.7,
                intent_shift=INTENT_SHIFT_DRIFT,
            ),
        ]
    )
    controller = YaklangStylePerceptionController(
        evaluator=evaluator,
        now=clock.now,
        config=_sync_perception_config(),
    )

    forced = await controller.after_tool_result(_event(0, force=True))
    assert len(forced) == 1
    assert evaluator.inputs[0].trigger == PERCEPTION_TRIGGER_FORCED
    assert controller.consume_downstream_refresh()["reason"] == "forced_or_intent_pivot"

    spin = await controller.after_tool_result(_event(1, force=True, loop_guard=True))
    assert len(spin) == 1
    assert evaluator.inputs[1].trigger == PERCEPTION_TRIGGER_SPIN_DETECTED
    assert evaluator.inputs[1].loop_guard["stalled"] is True
    assert controller.consume_downstream_refresh() == {}


@pytest.mark.asyncio
async def test_yaklang_style_perception_unchanged_results_backoff_without_downstream_refresh() -> None:
    clock = _Clock()
    evaluator = _QueuedPerceptionEvaluator(
        [
            PerceptionState(
                topics=("auth",),
                keywords=("auth",),
                summary="auth",
                changed=True,
                confidence=0.8,
                intent_shift=INTENT_SHIFT_PIVOT,
            ),
            PerceptionState(
                topics=("auth",),
                keywords=("auth",),
                summary="auth still",
                changed=False,
                confidence=0.8,
                intent_shift=INTENT_SHIFT_NONE,
            ),
            PerceptionState(
                topics=("auth",),
                keywords=("auth",),
                summary="auth still",
                changed=False,
                confidence=0.8,
                intent_shift=INTENT_SHIFT_NONE,
            ),
        ]
    )
    controller = YaklangStylePerceptionController(
        evaluator=evaluator,
        now=clock.now,
        config=_sync_perception_config(min_interval_seconds=0, max_interval_seconds=120),
    )

    await controller.after_tool_result(_event(4))
    assert controller.consume_downstream_refresh()
    await controller.after_tool_result(_event(8))
    assert controller.consume_downstream_refresh() == {}
    assert controller.consume_observation()["updated"] is False
    await controller.after_tool_result(_event(12))

    manifest = controller.manifest()
    assert manifest["consecutive_unchanged"] == 2
    assert manifest["current_interval_seconds"] == 0


@pytest.mark.asyncio
async def test_yaklang_style_perception_can_fire_async_without_blocking_action_hook() -> None:
    evaluator = _BlockingPerceptionEvaluator(
        PerceptionState(
            topics=("async perception",),
            keywords=("async",),
            summary="async perception finished",
            changed=True,
            confidence=0.8,
            intent_shift=INTENT_SHIFT_PIVOT,
        )
    )
    controller = YaklangStylePerceptionController(
        evaluator=evaluator,
        config=PerceptionRuntimeConfig(sync_triggers=False),
    )

    injections = await controller.after_tool_result(_event(4))

    assert injections == ()
    await evaluator.started.wait()
    assert controller.manifest()["background_task_count"] == 1
    evaluator.release.set()
    for _ in range(10):
        await asyncio.sleep(0)
        if controller.current_state() is not None:
            break
    assert controller.current_state() is not None
    assert controller.current_state().summary == "async perception finished"


@pytest.mark.asyncio
async def test_provider_backed_perception_evaluator_uses_internal_provider_request() -> None:
    provider = MockLLMProvider(
        [],
        perception_responses=[
            (
                '{"summary":"semantic phase pivot",'
                '"topics":["auth callback"],'
                '"keywords":["csrf","session"],'
                '"changed":true,"confidence":0.91,"intent_shift":"pivot"}'
            )
        ],
    )
    evaluator = ProviderBackedPerceptionEvaluator(provider, model="test-model")

    state = await evaluator.evaluate(
        PerceptionInput(
            task="inspect auth",
            iteration=3,
            trigger=PERCEPTION_TRIGGER_POST_ACTION,
            tool_result=ToolResult(
                call_id="call",
                tool_name="inspect_http",
                content="csrf session callback failed",
            ),
        )
    )

    assert state.summary == "semantic phase pivot"
    assert state.topics == ("auth callback",)
    assert state.keywords == ("csrf", "session")
    assert state.intent_shift == INTENT_SHIFT_PIVOT
    assert len(provider.perception_requests) == 1
    assert provider.perception_requests[0].metadata["agent_core_internal"] == "perception"
    assert provider.perception_requests[0].response_format is not None
    assert provider.perception_requests[0].response_format.kind == "json_schema"
