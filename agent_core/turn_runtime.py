"""Turn-level runtime contracts for Yaklang-style loop refresh."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from agent_core.actions import ParsedAction
from agent_core.context import ContextInjection
from agent_core.prompt import PromptBucketRole, PromptIR
from agent_core.providers import LLMMessage, LLMProviderPort, LLMRequest, LLMResponse, LLMResponseFormat
from agent_core.timeline import TimelineCursor, TimelineDiff
from agent_core.tools import ToolResult


PERCEPTION_TRIGGER_POST_ACTION = "post_action"
PERCEPTION_TRIGGER_VERIFICATION = "verification"
PERCEPTION_TRIGGER_FORCED = "forced"
PERCEPTION_TRIGGER_SPIN_DETECTED = "spin_detected"
PERCEPTION_TRIGGER_LOOP_SWITCH = "loop_switch"

INTENT_SHIFT_NONE = "none"
INTENT_SHIFT_DRIFT = "drift"
INTENT_SHIFT_PIVOT = "pivot"

_PERCEPTION_STOP_WORDS = frozenset(
    {
        "and",
        "are",
        "but",
        "for",
        "from",
        "into",
        "that",
        "the",
        "this",
        "with",
        "action",
        "agent",
        "call",
        "completed",
        "current",
        "error",
        "result",
        "status",
        "task",
        "tool",
    }
)

DEFAULT_PERCEPTION_DOWNSTREAM_INTENTS = (
    "memory_recall",
    "context_material_recall",
    "capability_search",
    "knowledge_search",
    "midterm_timeline_recall",
)


@dataclass(frozen=True)
class PerceptionDownstreamIntent:
    """Internal scheduling hint produced by perception."""

    intents: tuple[str, ...] = DEFAULT_PERCEPTION_DOWNSTREAM_INTENTS
    query: str = ""
    topics: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    summary: str = ""
    trigger: str = ""
    reason: str = "forced_or_intent_pivot"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-perception-downstream-intent/v1",
            "intents": list(self.intents),
            "query": self.query,
            "topics": list(self.topics),
            "keywords": list(self.keywords),
            "summary": self.summary,
            "trigger": self.trigger,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PerceptionDownstreamPlan:
    """Internal turn scheduler plan derived from perception state."""

    intents: tuple[str, ...] = ()
    query: str = ""
    topics: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    summary: str = ""
    trigger: str = ""
    enabled: bool = False
    reason: str = "no_downstream_refresh"
    metadata: dict[str, Any] = field(default_factory=dict)

    def intent_enabled(self, intent: str, *, default: bool) -> bool:
        if not self.enabled or not self.intents:
            return default
        return intent in set(self.intents)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-perception-downstream-plan/v1",
            "enabled": self.enabled,
            "intents": list(self.intents),
            "query": self.query,
            "topics": list(self.topics),
            "keywords": list(self.keywords),
            "summary": self.summary,
            "trigger": self.trigger,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


class PerceptionDownstreamScheduler:
    """Internal scheduler that turns perception refreshes into loop work."""

    def plan(self, downstream_refresh: dict[str, Any], *, fallback_query: str = "") -> PerceptionDownstreamPlan:
        if not downstream_refresh:
            return PerceptionDownstreamPlan(query=fallback_query, enabled=False)
        state = downstream_refresh.get("state")
        if not isinstance(state, dict):
            state = {}
        intent = downstream_refresh.get("intent")
        if not isinstance(intent, dict):
            intent = {}
        intents = _loose_string_tuple(downstream_refresh.get("intents")) or _loose_string_tuple(intent.get("intents"))
        if not intents:
            intents = DEFAULT_PERCEPTION_DOWNSTREAM_INTENTS
        topics = _loose_string_tuple(state.get("topics"))
        keywords = _loose_string_tuple(state.get("keywords"))
        summary = str(state.get("summary") or intent.get("summary") or "").strip()
        query = str(
            downstream_refresh.get("query")
            or intent.get("query")
            or " ".join(part for part in (summary, " ".join(topics), " ".join(keywords)) if part)
            or fallback_query
        ).strip()
        return PerceptionDownstreamPlan(
            intents=tuple(dict.fromkeys(intents)),
            query=query,
            topics=topics,
            keywords=keywords,
            summary=summary,
            trigger=str(downstream_refresh.get("trigger") or state.get("last_trigger") or ""),
            enabled=True,
            reason=str(downstream_refresh.get("reason") or "forced_or_intent_pivot"),
            metadata={"downstream_refresh": dict(downstream_refresh)},
        )


@dataclass(frozen=True)
class LoopStateFrame:
    """Internal prompt-facing frame for current loop state."""

    task: str = ""
    iteration: int = 0
    run_id: str = ""
    turn_id: str = ""
    root_task: str = ""
    parent_task: str = ""
    current_task: str = ""
    plan_context: str = ""
    current_objective: str = ""
    next_movement: str = ""
    dynamic_feedback: str = ""
    repair_context: str = ""
    approval_context: str = ""
    verification_state: str = ""
    action_history: tuple[dict[str, Any], ...] = ()
    failed_tools: tuple[str, ...] = ()
    perception_state: dict[str, Any] = field(default_factory=dict)
    memory_flush: dict[str, Any] = field(default_factory=dict)
    timeline_diff: dict[str, Any] = field(default_factory=dict)
    context_material_selection: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_prompt_content(self) -> bool:
        perception_payload = (
            self.perception_state.get("state")
            if isinstance(self.perception_state, dict)
            else {}
        )
        has_perception = isinstance(perception_payload, dict) and any(
            perception_payload.get(key) for key in ("summary", "topics", "keywords")
        )
        has_memory_flush = bool(
            self.memory_flush
            and (
                self.memory_flush.get("flushed")
                or self.memory_flush.get("flushed_signal_count")
                or self.memory_flush.get("decisions")
            )
        )
        return any(
            (
                self.root_task,
                self.parent_task,
                self.current_task,
                self.plan_context,
                self.current_objective,
                self.next_movement,
                self.dynamic_feedback,
                self.repair_context,
                self.approval_context,
                self.verification_state,
                self.failed_tools,
                has_perception,
                has_memory_flush,
            )
        )

    def render_prompt(self) -> str:
        if not self.has_prompt_content:
            return ""
        lines = ["[loop_state_frame]"]
        for key, value in (
            ("root_task", self.root_task),
            ("parent_task", self.parent_task),
            ("current_task", self.current_task),
            ("plan_context", self.plan_context),
            ("current_objective", self.current_objective),
            ("next_movement", self.next_movement),
            ("dynamic_feedback", self.dynamic_feedback),
            ("repair_context", self.repair_context),
            ("approval_context", self.approval_context),
            ("verification_state", self.verification_state),
        ):
            if value:
                lines.append(f"{key}: {value}")
        if self.failed_tools:
            lines.append("failed_tools: " + ", ".join(self.failed_tools))
        if self.action_history:
            lines.append("recent_action_decisions:")
            for record in self.action_history[-5:]:
                lines.append(
                    "- "
                    + " ".join(
                        part
                        for part in (
                            f"iteration={record.get('iteration')}",
                            f"action={record.get('action') or ''}",
                            f"tool={record.get('tool_name') or ''}",
                            f"status={record.get('status') or ''}",
                        )
                        if not part.endswith("=")
                    )
                )
        state = self.perception_state.get("state") if isinstance(self.perception_state, dict) else {}
        if isinstance(state, dict) and state:
            summary = str(state.get("summary") or "").strip()
            if summary:
                lines.append(f"perception_summary: {summary}")
        if self.timeline_diff:
            lines.append(
                "timeline_delta: "
                f"items={self.timeline_diff.get('item_count')} bytes={self.timeline_diff.get('bytes')}"
            )
        if self.context_material_selection:
            lines.append(
                "context_material_selection: "
                f"selected={self.context_material_selection.get('selected_count', self.context_material_selection.get('injection_count', 0))}"
            )
        if self.memory_flush:
            lines.append(
                "memory_flush: "
                f"pending={self.memory_flush.get('pending_signal_count', 0)} flushed={self.memory_flush.get('flushed', False)}"
            )
        return "\n".join(lines)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-loop-state-frame/v1",
            "task": self.task,
            "iteration": self.iteration,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "has_prompt_content": self.has_prompt_content,
            "root_task": self.root_task,
            "parent_task": self.parent_task,
            "current_task": self.current_task,
            "plan_context": self.plan_context,
            "current_objective": self.current_objective,
            "next_movement": self.next_movement,
            "dynamic_feedback": self.dynamic_feedback,
            "repair_context": self.repair_context,
            "approval_context": self.approval_context,
            "verification_state": self.verification_state,
            "action_history_count": len(self.action_history),
            "failed_tools": list(self.failed_tools),
            "perception_state": dict(self.perception_state),
            "memory_flush": dict(self.memory_flush),
            "timeline_diff": dict(self.timeline_diff),
            "context_material_selection": dict(self.context_material_selection),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TurnRefreshRequest:
    """Experimental request passed to the turn context refresher."""

    task: str
    iteration: int
    run_id: str
    turn_id: str
    bootstrap_prompt: PromptIR
    token_budget: int
    timeline_cursor: TimelineCursor | None = None
    compact_delta: tuple[LLMMessage, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-turn-refresh-request/v1",
            "task": self.task,
            "iteration": self.iteration,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "token_budget": self.token_budget,
            "timeline_cursor": self.timeline_cursor.manifest()
            if self.timeline_cursor is not None
            else {},
            "compact_delta_count": len(self.compact_delta),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TurnRefreshResult:
    """Experimental result returned by the turn context refresher."""

    prompt: PromptIR
    timeline_cursor: TimelineCursor
    timeline_diff: TimelineDiff | None = None
    memory_hits: tuple[Any, ...] = ()
    context_injections: tuple[ContextInjection, ...] = ()
    perception_state: dict[str, Any] = field(default_factory=dict)
    capability_selection: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-turn-refresh-result/v1",
            "prompt": self.prompt.manifest(),
            "timeline_cursor": self.timeline_cursor.manifest(),
            "timeline_diff": self.timeline_diff.manifest() if self.timeline_diff is not None else {},
            "memory_hit_count": len(self.memory_hits),
            "context_injections": [
                injection.manifest() for injection in self.context_injections
            ],
            "perception_state": dict(self.perception_state),
            "capability_selection": dict(self.capability_selection),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ModelResponseEvent:
    """Internal model-response event for turn lifecycle callbacks."""

    task: str
    iteration: int
    run_id: str
    turn_id: str
    response: LLMResponse
    timeline_diff: TimelineDiff | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResultEvent:
    """Internal tool/action event for turn lifecycle callbacks."""

    task: str
    iteration: int
    run_id: str
    turn_id: str
    tool_result: ToolResult
    action: ParsedAction | None = None
    timeline_diff: TimelineDiff | None = None
    force_perception: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnCompletedEvent:
    """Internal turn completion event for memory flush and observations."""

    task: str
    iteration: int
    run_id: str
    turn_id: str
    status: str
    timeline_diff: TimelineDiff | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerceptionRuntimeConfig:
    """Internal Yaklang-style perception scheduling knobs."""

    iteration_trigger_interval: int = 4
    min_interval_seconds: float = 120.0
    max_interval_seconds: float = 300.0
    max_context_bytes: int = 4096
    max_recent_actions: int = 5
    sync_triggers: bool = False

    def normalized(self) -> "PerceptionRuntimeConfig":
        min_interval = max(0.0, float(self.min_interval_seconds))
        max_interval = max(min_interval, float(self.max_interval_seconds))
        return PerceptionRuntimeConfig(
            iteration_trigger_interval=max(0, int(self.iteration_trigger_interval)),
            min_interval_seconds=min_interval,
            max_interval_seconds=max_interval,
            max_context_bytes=max(512, int(self.max_context_bytes)),
            max_recent_actions=max(1, int(self.max_recent_actions)),
            sync_triggers=bool(self.sync_triggers),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-perception-runtime-config/v1",
            "iteration_trigger_interval": self.iteration_trigger_interval,
            "min_interval_seconds": self.min_interval_seconds,
            "max_interval_seconds": self.max_interval_seconds,
            "max_context_bytes": self.max_context_bytes,
            "max_recent_actions": self.max_recent_actions,
            "sync_triggers": self.sync_triggers,
        }


@dataclass(frozen=True)
class PerceptionState:
    """Prompt-safe semantic state for the current loop phase."""

    topics: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    summary: str = ""
    confidence: float = 0.0
    changed: bool = False
    epoch: int = 0
    last_trigger: str = ""
    last_update_at: datetime | None = None
    prev_topics_hash: str = ""
    intent_shift: str = ""

    def is_intent_pivot(self) -> bool:
        shift = self.intent_shift.strip().lower()
        if shift == INTENT_SHIFT_PIVOT:
            return True
        if shift in {INTENT_SHIFT_DRIFT, INTENT_SHIFT_NONE}:
            return False
        return self.changed

    def should_refresh_downstream(self, *, updated: bool) -> bool:
        if not updated:
            return False
        if self.last_trigger == PERCEPTION_TRIGGER_FORCED:
            return True
        return self.is_intent_pivot()

    def format_for_context(self, *, now: datetime | None = None) -> str:
        if not self.summary and not self.topics and not self.keywords:
            return ""
        now = now or datetime.now(UTC)
        if self.last_update_at is None:
            age = "unknown"
        else:
            seconds = max(0, int((now - self.last_update_at).total_seconds()))
            age = f"{seconds}s"
        lines = [f"## Current Perception (epoch {self.epoch}, {age} ago)"]
        if self.summary:
            lines.append(f"Summary: {self.summary}")
        if self.topics:
            lines.append("Topics: " + ", ".join(self.topics))
        if self.keywords:
            lines.append("Keywords: " + ", ".join(self.keywords))
        if self.intent_shift:
            lines.append(f"Intent Shift: {self.intent_shift}")
        return "\n".join(lines)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-perception-state/v1",
            "topics": list(self.topics),
            "keywords": list(self.keywords),
            "summary": self.summary,
            "confidence": self.confidence,
            "changed": self.changed,
            "epoch": self.epoch,
            "last_trigger": self.last_trigger,
            "last_update_at": self.last_update_at.isoformat()
            if self.last_update_at is not None
            else "",
            "prev_topics_hash": self.prev_topics_hash,
            "intent_shift": self.intent_shift,
            "intent_pivot": self.is_intent_pivot(),
        }


@dataclass(frozen=True)
class PerceptionInput:
    task: str
    iteration: int
    trigger: str
    tool_result: ToolResult
    action: ParsedAction | None = None
    timeline_diff: TimelineDiff | None = None
    previous_state: PerceptionState | None = None
    recent_actions: tuple[dict[str, Any], ...] = ()
    loop_guard: dict[str, Any] = field(default_factory=dict)
    verification_result: dict[str, Any] = field(default_factory=dict)
    task_metadata: dict[str, Any] = field(default_factory=dict)
    context_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, *, max_bytes: int = 4096) -> str:
        lines = [
            f"User Goal: {self.task}",
            f"Iteration: {self.iteration + 1}",
            f"Trigger: {self.trigger}",
            f"Tool: {self.tool_result.tool_name}",
            f"Tool Status: {self.tool_result.status}",
        ]
        if self.action is not None:
            lines.append(f"Action: {self.action.name}")
        content = self.tool_result.content or self.tool_result.error
        if content:
            lines.append("Tool Output:\n" + content)
        if self.timeline_diff is not None and self.timeline_diff.changed:
            lines.append("Recent Timeline Changes:\n" + self.timeline_diff.render())
        if self.recent_actions:
            lines.append(
                "Recent Actions:\n"
                + "\n".join(
                    json.dumps(item, ensure_ascii=False, sort_keys=True)
                    for item in self.recent_actions
                )
            )
        if self.loop_guard:
            lines.append(
                "Loop Guard / Spin Metadata:\n"
                + json.dumps(self.loop_guard, ensure_ascii=False, sort_keys=True)
            )
        if self.verification_result:
            lines.append(
                "Verification Result:\n"
                + json.dumps(self.verification_result, ensure_ascii=False, sort_keys=True)
            )
        if self.task_metadata:
            lines.append(
                "Current Plan/Task Metadata:\n"
                + json.dumps(self.task_metadata, ensure_ascii=False, sort_keys=True)
            )
        if self.context_summary:
            lines.append("Context Material Summary:\n" + self.context_summary)
        if self.previous_state is not None:
            lines.append(
                "Previous Perception:\n"
                + self.previous_state.format_for_context(now=self.previous_state.last_update_at)
            )
        rendered = "\n\n".join(lines)
        raw = rendered.encode("utf-8")
        if len(raw) <= max_bytes:
            return rendered
        return raw[:max_bytes].decode("utf-8", errors="ignore").rstrip()


class PerceptionEvaluatorPort(Protocol):
    """Internal extension point for semantic perception evaluation."""

    async def evaluate(self, request: PerceptionInput) -> PerceptionState:
        """Return one perception assessment."""


class DeterministicPerceptionEvaluator:
    """Dependency-free semantic fallback for SDK-only runtimes."""

    async def evaluate(self, request: PerceptionInput) -> PerceptionState:
        text = request.render(max_bytes=8192)
        keywords = _top_terms(text, limit=8)
        topics = _topic_terms(text, keywords)
        summary = _one_line_summary(request, topics, keywords)
        previous = request.previous_state
        previous_hash = _hash_topics(previous.topics) if previous is not None else ""
        current_hash = _hash_topics(topics)
        changed = previous is None or current_hash != previous_hash
        intent_shift = INTENT_SHIFT_PIVOT if changed else INTENT_SHIFT_NONE
        return PerceptionState(
            topics=topics,
            keywords=keywords,
            summary=summary,
            confidence=0.65 if keywords else 0.35,
            changed=changed,
            last_trigger=request.trigger,
            prev_topics_hash=previous_hash,
            intent_shift=intent_shift,
        )


class ProviderBackedPerceptionEvaluator:
    """Yaklang-style AI semantic perception backed by the session provider."""

    def __init__(
        self,
        provider: LLMProviderPort,
        *,
        model: str = "",
        fallback: PerceptionEvaluatorPort | None = None,
        strict_response_format: bool = True,
    ) -> None:
        self.provider = provider
        self.model = model
        self.fallback = fallback or DeterministicPerceptionEvaluator()
        self.strict_response_format = bool(strict_response_format)
        self.last_manifest: dict[str, Any] = {}

    async def evaluate(self, request: PerceptionInput) -> PerceptionState:
        prompt = _build_perception_prompt(request)
        llm_request = LLMRequest(
            messages=[
                LLMMessage(
                    role="system",
                    content=(
                        "You are the SDK semantic perception layer. "
                        "Return only JSON matching the requested schema."
                    ),
                ),
                LLMMessage(role="user", content=prompt),
            ],
            model=self.model,
            response_format=(
                LLMResponseFormat(
                    kind="json_schema",
                    name="agent_core_perception",
                    description="Semantic perception state for the current agent loop.",
                    schema=_perception_response_schema(),
                    strict=True,
                )
                if self.strict_response_format
                else None
            ),
            metadata={
                "agent_core_internal": "perception",
                "agent_core_perception": True,
                "trigger": request.trigger,
                "iteration": request.iteration,
                "strict_response_format": self.strict_response_format,
            },
        )
        try:
            response = await self.provider.complete(llm_request)
            payload = _perception_payload_from_response(response)
            state = _perception_state_from_payload(payload)
            self.last_manifest = {
                "schema_version": "agent-core-provider-backed-perception-evaluator/v1",
                "provider_called": True,
                "fallback_used": False,
                "request": llm_request.manifest(),
                "response": response.manifest(),
            }
            return state
        except Exception as exc:
            fallback_state = await self.fallback.evaluate(request)
            self.last_manifest = {
                "schema_version": "agent-core-provider-backed-perception-evaluator/v1",
                "provider_called": True,
                "fallback_used": True,
                "error": str(exc),
            }
            return fallback_state


class TurnContextRefresherPort(Protocol):
    """Stable contract for turn-level prompt refresh."""

    async def before_model_call(self, request: TurnRefreshRequest) -> TurnRefreshResult:
        """Return the prompt view for the next model call."""

    async def after_model_response(self, event: ModelResponseEvent) -> None:
        """Observe model output after it has been recorded."""

    async def after_tool_result(self, event: ToolResultEvent) -> None:
        """Observe tool/action output after it has been recorded."""

    async def after_turn(self, event: TurnCompletedEvent) -> None:
        """Flush end-of-turn state."""


class PerceptionControllerPort(Protocol):
    """Stable contract for loop-level semantic perception."""

    async def after_tool_result(self, event: ToolResultEvent) -> tuple[ContextInjection, ...]:
        """Return context injections generated from the tool/action result."""

    def manifest(self) -> dict[str, Any]:
        """Return prompt-safe perception state."""


class CapabilityRefreshPort(Protocol):
    """Stable contract for loop-level capability selection refresh."""

    async def refresh(self, request: TurnRefreshRequest) -> dict[str, Any]:
        """Return prompt-safe capability selection metadata."""


class NullTurnContextRefresher:
    """Fallback refresher that keeps existing behavior when no session wiring exists."""

    async def before_model_call(self, request: TurnRefreshRequest) -> TurnRefreshResult:
        cursor = request.timeline_cursor or TimelineCursor()
        return TurnRefreshResult(prompt=request.bootstrap_prompt, timeline_cursor=cursor)

    async def after_model_response(self, event: ModelResponseEvent) -> None:
        return None

    async def after_tool_result(self, event: ToolResultEvent) -> None:
        return None

    async def after_turn(self, event: TurnCompletedEvent) -> None:
        return None


class NullPerceptionController:
    async def after_tool_result(self, event: ToolResultEvent) -> tuple[ContextInjection, ...]:
        return ()

    def manifest(self) -> dict[str, Any]:
        return {"enabled": False}


class YaklangStylePerceptionController:
    """SDK-owned perception lifecycle with Yaklang-style timing and gates."""

    def __init__(
        self,
        *,
        evaluator: PerceptionEvaluatorPort | None = None,
        config: PerceptionRuntimeConfig | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = (config or PerceptionRuntimeConfig()).normalized()
        self.evaluator = evaluator or DeterministicPerceptionEvaluator()
        self._now = now or (lambda: datetime.now(UTC))
        self._state: PerceptionState | None = None
        self._epoch = 0
        self._current_interval = timedelta(seconds=self.config.min_interval_seconds)
        self._consecutive_unchanged = 0
        self._running = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[tuple[ContextInjection, ...]]] = set()
        self._recent_actions: list[dict[str, Any]] = []
        self._last_trigger_manifest: dict[str, Any] = {
            "triggered": False,
            "reason": "not_started",
        }
        self._downstream_refresh: dict[str, Any] = {}
        self._pending_observation: dict[str, Any] = {}
        self._last_task_signature = ""

    async def after_tool_result(self, event: ToolResultEvent) -> tuple[ContextInjection, ...]:
        self._remember_action(event)
        trigger = self._trigger_for_event(event)
        if not self._should_trigger(event, trigger):
            return ()
        if not self.config.sync_triggers:
            task = asyncio.create_task(self._evaluate_event(event, trigger))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            self._last_trigger_manifest = {
                "triggered": True,
                "trigger": trigger,
                "iteration": event.iteration,
                "async": True,
            }
            return self._current_injection()
        return await self._evaluate_event(event, trigger)

    async def _evaluate_event(
        self,
        event: ToolResultEvent,
        trigger: str,
    ) -> tuple[ContextInjection, ...]:
        if self._running.locked():
            self._last_trigger_manifest = {
                "triggered": False,
                "reason": "already_running",
                "trigger": trigger,
                "iteration": event.iteration,
            }
            return self._current_injection()
        async with self._running:
            perception_input = PerceptionInput(
                task=event.task,
                iteration=event.iteration,
                trigger=trigger,
                tool_result=event.tool_result,
                action=event.action,
                timeline_diff=event.timeline_diff,
                previous_state=self._state,
                recent_actions=tuple(self._recent_actions[-self.config.max_recent_actions :]),
                loop_guard=_dict_metadata(event.metadata.get("loop_guard")),
                verification_result=_dict_metadata(event.metadata.get("verification_result")),
                task_metadata=_dict_metadata(event.metadata.get("task_metadata")),
                context_summary=str(event.metadata.get("context_summary") or ""),
                metadata=dict(event.metadata),
            )
            candidate = await self.evaluator.evaluate(perception_input)
            self._last_task_signature = _task_signature(event.task)
            updated = self._apply_result(candidate, trigger=trigger, force=event.force_perception)
            downstream = self._state.should_refresh_downstream(updated=updated) if self._state else False
            if downstream and self._state is not None:
                intent = _downstream_intent_from_state(
                    self._state,
                    trigger=trigger,
                    metadata={
                        "run_id": event.run_id,
                        "turn_id": event.turn_id,
                        "iteration": event.iteration,
                    },
                )
                self._downstream_refresh = {
                    "schema_version": "agent-core-perception-downstream-refresh/v1",
                    "trigger": trigger,
                    "state": self._state.manifest(),
                    "reason": "forced_or_intent_pivot",
                    "intent": intent.manifest(),
                    "intents": list(intent.intents),
                    "query": intent.query,
                }
            if self._state is not None:
                self._pending_observation = {
                    "schema_version": "agent-core-perception-observation/v1",
                    "triggered": True,
                    "trigger": trigger,
                    "iteration": event.iteration,
                    "updated": updated,
                    "downstream_refresh": downstream,
                    "state": self._state.manifest(),
                    "tool_name": event.tool_result.tool_name,
                    "tool_status": event.tool_result.status,
                }
            self._last_trigger_manifest = {
                "triggered": True,
                "trigger": trigger,
                "iteration": event.iteration,
                "updated": updated,
                "downstream_refresh": downstream,
            }
            return self._current_injection()

    def manifest(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "schema_version": "agent-core-yaklang-style-perception-controller/v1",
            "config": self.config.manifest(),
            "state": self._state.manifest() if self._state is not None else {},
            "epoch": self._epoch,
            "current_interval_seconds": self._current_interval.total_seconds(),
            "consecutive_unchanged": self._consecutive_unchanged,
            "recent_action_count": len(self._recent_actions),
            "background_task_count": len(self._background_tasks),
            "last_trigger": dict(self._last_trigger_manifest),
            "pending_downstream_refresh": bool(self._downstream_refresh),
        }

    def consume_downstream_refresh(self) -> dict[str, Any]:
        refresh = dict(self._downstream_refresh)
        self._downstream_refresh = {}
        return refresh

    def consume_observation(self) -> dict[str, Any]:
        observation = dict(self._pending_observation)
        self._pending_observation = {}
        return observation

    def context_injections(self) -> tuple[ContextInjection, ...]:
        return self._current_injection()

    def current_state(self) -> PerceptionState | None:
        return self._state

    def _remember_action(self, event: ToolResultEvent) -> None:
        action_name = event.action.name if event.action is not None else ""
        self._recent_actions.append(
            {
                "iteration": event.iteration,
                "action": action_name,
                "tool_name": event.tool_result.tool_name,
                "status": event.tool_result.status,
                "ok": event.tool_result.ok,
            }
        )
        self._recent_actions = self._recent_actions[-self.config.max_recent_actions :]

    def _trigger_for_event(self, event: ToolResultEvent) -> str:
        if not event.force_perception:
            return PERCEPTION_TRIGGER_POST_ACTION
        if event.metadata.get("trigger"):
            return str(event.metadata["trigger"])
        if event.metadata.get("loop_guard"):
            return PERCEPTION_TRIGGER_SPIN_DETECTED
        return PERCEPTION_TRIGGER_FORCED

    def _should_trigger(self, event: ToolResultEvent, trigger: str) -> bool:
        forced = event.force_perception or trigger in {
            PERCEPTION_TRIGGER_FORCED,
            PERCEPTION_TRIGGER_SPIN_DETECTED,
            PERCEPTION_TRIGGER_LOOP_SWITCH,
        }
        if not forced:
            task_signature = _task_signature(event.task)
            if self._last_task_signature and task_signature != self._last_task_signature:
                return True
            if self._state is None:
                return True
            interval = self.config.iteration_trigger_interval
            if interval <= 0 or event.iteration <= 0 or event.iteration % interval != 0:
                self._last_trigger_manifest = {
                    "triggered": False,
                    "reason": "iteration_interval_not_reached",
                    "iteration": event.iteration,
                    "iteration_trigger_interval": interval,
                }
                return False
            if self._state is not None and self._now() - self._state.last_update_at < self._current_interval:  # type: ignore[operator]
                self._last_trigger_manifest = {
                    "triggered": False,
                    "reason": "time_interval_not_reached",
                    "iteration": event.iteration,
                    "current_interval_seconds": self._current_interval.total_seconds(),
                }
                return False
        return True

    def _apply_result(self, candidate: PerceptionState, *, trigger: str, force: bool) -> bool:
        now = self._now()
        self._epoch += 1
        candidate = replace(
            candidate,
            epoch=self._epoch,
            last_trigger=trigger,
            last_update_at=now,
            topics=tuple(candidate.topics),
            keywords=tuple(candidate.keywords),
            intent_shift=str(candidate.intent_shift or "").strip().lower(),
        )
        updated = self._should_update(candidate, force=force)
        if updated:
            self._state = replace(candidate, prev_topics_hash=_hash_topics(candidate.topics))
            self._consecutive_unchanged = 0
            self._current_interval = timedelta(seconds=self.config.min_interval_seconds)
            return True
        if self._state is not None:
            self._state = replace(
                self._state,
                epoch=candidate.epoch,
                last_trigger=trigger,
                last_update_at=now,
            )
        else:
            self._state = candidate
        self._consecutive_unchanged += 1
        if self._consecutive_unchanged >= 2:
            seconds = min(
                self.config.max_interval_seconds,
                max(self.config.min_interval_seconds, self._current_interval.total_seconds() * 2),
            )
            self._current_interval = timedelta(seconds=seconds)
        return False

    def _should_update(self, candidate: PerceptionState, *, force: bool) -> bool:
        if self._state is None:
            return True
        if force or candidate.last_trigger in {
            PERCEPTION_TRIGGER_FORCED,
            PERCEPTION_TRIGGER_SPIN_DETECTED,
            PERCEPTION_TRIGGER_LOOP_SWITCH,
        }:
            return True
        if not candidate.changed:
            return False
        return _hash_topics(candidate.topics) != _hash_topics(self._state.topics)

    def _current_injection(self) -> tuple[ContextInjection, ...]:
        if self._state is None:
            return ()
        content = self._state.format_for_context(now=self._now())
        if not content:
            return ()
        raw = content.encode("utf-8")
        if len(raw) > self.config.max_context_bytes:
            content = raw[: self.config.max_context_bytes].decode("utf-8", errors="ignore").rstrip()
        return (
            ContextInjection(
                name="perception_awareness",
                content=content,
                target=PromptBucketRole.TIMELINE_OPEN,
                source="perception",
                priority=90,
                metadata={
                    "perception": self._state.manifest(),
                    "controller": "yaklang_style",
                },
            ),
        )


class NullCapabilityRefresh:
    async def refresh(self, request: TurnRefreshRequest) -> dict[str, Any]:
        return {"enabled": False}


def _loose_string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    try:
        return tuple(str(item).strip() for item in value or () if str(item).strip())
    except TypeError:
        return ()


def _top_terms(text: str, *, limit: int) -> tuple[str, ...]:
    counts: dict[str, int] = {}
    for token in re.findall(r"[a-zA-Z0-9_\u4e00-\u9fff]{2,}", str(text or "").lower()):
        if token in _PERCEPTION_STOP_WORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts, key=lambda item: (-counts[item], item))
    return tuple(ranked[:limit])


def _topic_terms(text: str, keywords: tuple[str, ...]) -> tuple[str, ...]:
    if keywords:
        return tuple(keywords[: min(5, len(keywords))])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8] if text else "empty"
    return (f"context-{digest}",)


def _task_signature(task: str) -> str:
    terms = _top_terms(task, limit=8)
    if terms:
        return hashlib.sha256("|".join(terms).encode("utf-8")).hexdigest()[:16]
    normalized = " ".join(str(task or "").lower().split())
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _one_line_summary(
    request: PerceptionInput,
    topics: tuple[str, ...],
    keywords: tuple[str, ...],
) -> str:
    subject = ", ".join(topics[:3] or keywords[:3])
    if not subject:
        subject = request.tool_result.tool_name or "current task"
    status = "failed" if not request.tool_result.ok else "completed"
    return f"{request.tool_result.tool_name or 'tool'} {status}; focus: {subject}".strip()


def _hash_topics(topics: tuple[str, ...]) -> str:
    normalized = "|".join(sorted(str(topic).strip().lower() for topic in topics if str(topic).strip()))
    if not normalized:
        return ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _downstream_intent_from_state(
    state: PerceptionState,
    *,
    trigger: str,
    metadata: dict[str, Any] | None = None,
) -> PerceptionDownstreamIntent:
    query = " ".join(
        item
        for item in (
            state.summary,
            " ".join(state.topics),
            " ".join(state.keywords),
        )
        if str(item).strip()
    ).strip()
    return PerceptionDownstreamIntent(
        query=query,
        topics=state.topics,
        keywords=state.keywords,
        summary=state.summary,
        trigger=trigger,
        metadata=dict(metadata or {}),
    )


def _dict_metadata(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _build_perception_prompt(request: PerceptionInput) -> str:
    return (
        "Assess the current agent loop state.\n"
        "Return JSON with: summary, topics, keywords, changed, confidence, intent_shift.\n"
        "intent_shift must be one of: none, drift, pivot.\n\n"
        + request.render(max_bytes=30000)
    )


def _perception_response_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["summary", "topics", "keywords", "changed", "confidence"],
        "properties": {
            "summary": {"type": "string"},
            "topics": {"type": "array", "items": {"type": "string"}},
            "keywords": {"type": "array", "items": {"type": "string"}},
            "changed": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "intent_shift": {
                "type": "string",
                "enum": [INTENT_SHIFT_NONE, INTENT_SHIFT_DRIFT, INTENT_SHIFT_PIVOT],
            },
        },
        "additionalProperties": False,
    }


def _perception_payload_from_response(response: LLMResponse) -> dict[str, Any]:
    if isinstance(response.action, dict) and response.action:
        if "arguments" in response.action and isinstance(response.action["arguments"], dict):
            return dict(response.action["arguments"])
        return dict(response.action)
    text = response.content.strip()
    if not text:
        raise ValueError("perception response is empty")
    return _extract_json_object(text)


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("perception response must be a JSON object")
    return payload


def _perception_state_from_payload(payload: dict[str, Any]) -> PerceptionState:
    return PerceptionState(
        topics=_string_tuple(payload.get("topics"), limit=5),
        keywords=_string_tuple(payload.get("keywords"), limit=8),
        summary=str(payload.get("summary") or "").strip(),
        changed=bool(payload.get("changed")),
        confidence=max(0.0, min(1.0, float(payload.get("confidence") or 0.0))),
        intent_shift=str(payload.get("intent_shift") or "").strip().lower(),
    )


def _string_tuple(value: Any, *, limit: int) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value or ())
        except TypeError:
            items = []
    normalized = []
    for item in items:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return tuple(normalized[:limit])
