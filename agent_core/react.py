"""Provider-neutral ReAct executor."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, TypeVar

from agent_core.actions import (
    ActionRegistry,
    ActionSpec,
    ActionVerifierPort,
    NullActionVerifier,
    ParsedAction,
)
from agent_core.approvals import (
    ApprovalGrant,
    ApprovalRecord,
    ApprovalResumeContext,
    ApprovalStorePort,
    NullApprovalStore,
)
from agent_core.artifacts import ArtifactStorePort
from agent_core.capabilities import CapabilityCatalog, CapabilityQuery
from agent_core.config import RuntimeBudget
from agent_core.errors import ActionError, ProviderError
from agent_core.events import AgentEvent, EventSinkPort, NullEventSink
from agent_core.harness import AgentHarness, CancelToken, RunInterrupt, RunState, TurnState
from agent_core.loop_guard import LoopGuard, LoopGuardConfig
from agent_core.memory import MemoryPort, MemoryQuery, NullMemory
from agent_core.policy import (
    AllowAllPolicy,
    ApprovalRequest,
    NullPolicyDecisionStore,
    PolicyDecision,
    PolicyDecisionStorePort,
    PolicyPort,
    PolicySubjectKind,
)
from agent_core.prompt import PromptBucket, PromptBucketRole, PromptIR, default_cache_hint
from agent_core.providers import (
    LLMMessage,
    LLMProviderPort,
    LLMRequest,
    LLMResponse,
    LLMResponseFormat,
    LLMStreamAccumulator,
    LLMToolCall,
    LLMToolChoice,
    LLMToolContract,
)
from agent_core.skills import SkillsContext
from agent_core.structured import (
    JsonStructuredOutputValidator,
    StructuredOutputSpec,
    StructuredOutputValidatorPort,
    structured_output_feedback,
)
from agent_core.mcp import MCPCenter
from agent_core.timeline import TimelineStore
from agent_core.tools import (
    NullToolReplay,
    ToolInvocation,
    ToolExecutionCenter,
    ToolRegistry,
    ToolReplayPort,
    ToolResult,
    ToolRuntimePort,
    ToolRetryPolicy,
    tool_manifest_item,
)
from agent_core.turn_runtime import (
    ModelResponseEvent,
    NullTurnContextRefresher,
    ToolResultEvent,
    TurnCompletedEvent,
    TurnContextRefresherPort,
    TurnRefreshRequest,
)

T = TypeVar("T")

TIMELINE_SOURCE_REACT = "react_executor"
TIMELINE_KIND_TASK = "task"
TIMELINE_KIND_MODEL = "model"
TIMELINE_KIND_TOOL = "tool"
TIMELINE_KIND_POLICY = "policy"
TIMELINE_KIND_APPROVAL = "approval"
TIMELINE_KIND_FINAL = "final"
TIMELINE_KIND_VERIFICATION = "verification"
TIMELINE_KIND_LOOP = "loop"
TIMELINE_KIND_ERROR = "error"


@dataclass(frozen=True)
class ReActConfig:
    model: str = ""
    max_iterations: int = 12
    finish_action: str = "finish"
    tool_action: str = "call_tool"
    search_tools_action: str = "search_tools"
    search_skills_action: str = "search_skills"
    search_capabilities_action: str = "search_capabilities"
    load_capability_action: str = "load_capability"
    query_mcp_servers_action: str = "query_mcp_servers"
    query_mcp_tools_action: str = "query_mcp_tools"
    load_skill_action: str = "load_skill"
    load_skill_resource_action: str = "load_skill_resource"
    change_skill_view_offset_action: str = "change_skill_view_offset"
    stream: bool = False
    budget: RuntimeBudget = field(default_factory=RuntimeBudget)
    loop_guard: LoopGuardConfig = field(default_factory=LoopGuardConfig)
    structured_output: StructuredOutputSpec | None = None
    timeout_seconds: float | None = None
    tool_retry_policy: ToolRetryPolicy = field(default_factory=ToolRetryPolicy)
    native_tool_calls: bool = False
    loop_delta_max_bytes: int = 16 * 1024
    provider_cache_mode: str = "strip"
    provider_cache_min_segment_bytes: int = 1024


@dataclass(frozen=True)
class ReActResult:
    run_id: str
    status: str
    output: str = ""
    iterations: int = 0
    final_action: ParsedAction | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ReActExecutor:
    def __init__(
        self,
        *,
        provider: LLMProviderPort,
        tool_runtime: ToolRuntimePort,
        action_registry: ActionRegistry,
        harness: AgentHarness,
        event_sink: EventSinkPort | None = None,
        policy: PolicyPort | None = None,
        policy_decision_store: PolicyDecisionStorePort | None = None,
        approval_store: ApprovalStorePort | None = None,
        approval_resume: ApprovalResumeContext | None = None,
        memory: MemoryPort | None = None,
        skills: SkillsContext | None = None,
        mcp: MCPCenter | None = None,
        capability_catalog: CapabilityCatalog | None = None,
        timeline: TimelineStore | None = None,
        tool_replay: ToolReplayPort | None = None,
        action_verifier: ActionVerifierPort | None = None,
        structured_output_validator: StructuredOutputValidatorPort | None = None,
        loop_guard: LoopGuard | None = None,
        artifact_store: ArtifactStorePort | None = None,
        cancel_token: CancelToken | None = None,
        turn_refresher: TurnContextRefresherPort | None = None,
        config: ReActConfig | None = None,
    ) -> None:
        self.provider = provider
        self.tool_runtime = tool_runtime
        self.action_registry = action_registry
        self.harness = harness
        self.event_sink = event_sink or NullEventSink()
        self.policy = policy or AllowAllPolicy()
        self.policy_decision_store = policy_decision_store or NullPolicyDecisionStore()
        self.approval_store = approval_store or NullApprovalStore()
        self.approval_resume = approval_resume or ApprovalResumeContext()
        self.memory = memory or NullMemory()
        self.skills = skills
        self.mcp = mcp
        self.capability_catalog = capability_catalog
        self.timeline = timeline
        self.tool_replay = tool_replay or NullToolReplay()
        self.action_verifier = action_verifier or NullActionVerifier()
        self.structured_output_validator = structured_output_validator or JsonStructuredOutputValidator()
        self.cancel_token = cancel_token or CancelToken()
        self._uses_null_refresher = turn_refresher is None
        self.turn_refresher = turn_refresher or NullTurnContextRefresher()
        self.config = config or ReActConfig()
        self.loop_guard = loop_guard or LoopGuard(self.config.loop_guard)
        self.artifact_store = artifact_store
        self.last_prompt_manifest: dict[str, Any] = {}
        self._action_history: list[dict[str, Any]] = []
        self._loaded_capabilities: dict[str, dict[str, Any]] = {}
        self._event_sequence = 0
        self._ensure_builtin_actions()
        self._ensure_tool_actions()

    async def run(self, task: str, prompt: PromptIR) -> ReActResult:
        run = await self.harness.start_run(task)
        await self._emit("run_started", run, payload={"task": task})
        await self._record_timeline(
            task,
            kind=TIMELINE_KIND_TASK,
            run=run,
            status="started",
        )
        deadline = _deadline_after(self.config.timeout_seconds)
        compact_delta: list[LLMMessage] = []
        timeline_cursor = self.timeline.diff_since().next_cursor if self.timeline is not None else None
        result_output = ""
        final_action: ParsedAction | None = None
        structured_output_repairs = 0

        for index in range(self.config.max_iterations):
            if _deadline_expired(deadline):
                self.cancel_token.timeout(
                    "run timeout",
                    timeout_seconds=self.config.timeout_seconds,
                    metadata={"iteration": index, "phase": "before_turn"},
                )
                return await self._finish_interrupted(
                    run,
                    status="timeout",
                    event_type="run_timeout",
                    output=self.cancel_token.reason or "run timeout",
                    iterations=index,
                )
            if self.cancel_token.cancelled:
                return await self._finish_interrupted(
                    run,
                    status="cancelled",
                    event_type="run_cancelled",
                    output=self.cancel_token.reason or "run cancelled",
                    iterations=index,
                )
            turn = await self.harness.start_turn(run, index)
            await self._emit("turn_started", run, turn_id=turn.turn_id, payload={"index": index})
            refresh = await self.turn_refresher.before_model_call(
                TurnRefreshRequest(
                    task=task,
                    iteration=index,
                    run_id=run.run_id,
                    turn_id=turn.turn_id,
                    bootstrap_prompt=prompt,
                    token_budget=self.config.budget.max_prompt_bytes,
                    timeline_cursor=timeline_cursor,
                    compact_delta=tuple(compact_delta),
                    metadata={
                        "model": self.config.model,
                        "action_history": tuple(self._action_history[-12:]),
                    },
                )
            )
            current_prompt = refresh.prompt
            timeline_cursor = refresh.timeline_cursor
            prompt_manifest = current_prompt.manifest()
            self.last_prompt_manifest = prompt_manifest
            await self.harness.record_prompt(turn, prompt_manifest)
            await self._emit(
                "prompt_ready",
                run,
                turn_id=turn.turn_id,
                payload={"manifest": prompt_manifest, "turn_refresh": refresh.manifest()},
            )
            legacy_memory_messages: list[LLMMessage] = []
            if self._uses_null_refresher:
                memory_message = await self._memory_message(task)
                if memory_message is not None:
                    legacy_memory_messages.append(memory_message)

            provider_messages = self._provider_messages(
                current_prompt,
                compact_delta,
                extra_messages=legacy_memory_messages,
            )
            native_tools = self._native_tool_contracts(
                task=task,
                provider_messages=provider_messages,
            )
            if native_tools:
                provider_messages = self._provider_messages(
                    current_prompt,
                    compact_delta,
                    extra_messages=legacy_memory_messages,
                    reserved_bytes=_native_tool_contracts_bytes(native_tools),
                )
            await self._emit(
                "prompt_profile",
                run,
                turn_id=turn.turn_id,
                payload=self._prompt_profile_payload(
                    prompt_manifest,
                    refresh_manifest=refresh.manifest(),
                    compact_delta=compact_delta,
                    provider_messages=provider_messages,
                ),
            )

            try:
                response = await self._await_with_deadline(
                    self._call_provider(
                        LLMRequest(
                            messages=provider_messages,
                            model=self.config.model,
                            tools=native_tools,
                            tool_choice=self._native_tool_choice(native_tools),
                            response_format=_structured_response_format(
                                self.config.structured_output
                            ),
                            metadata=_provider_request_metadata(self.config),
                        ),
                        run=run,
                        turn_id=turn.turn_id,
                    ),
                    deadline,
                )
            except asyncio.TimeoutError:
                self.cancel_token.timeout(
                    "provider call timed out",
                    timeout_seconds=self.config.timeout_seconds,
                    metadata={"iteration": index, "phase": "provider"},
                )
                return await self._finish_interrupted(
                    run,
                    status="timeout",
                    event_type="run_timeout",
                    output=self.cancel_token.reason or "provider call timed out",
                    iterations=index + 1,
                    turn=turn,
                    checkpoint_state={"phase": "provider"},
                )
            if self.cancel_token.cancelled:
                return await self._finish_interrupted(
                    run,
                    status="cancelled",
                    event_type="run_cancelled",
                    output=self.cancel_token.reason or "run cancelled",
                    iterations=index + 1,
                    turn=turn,
                )
            await self.harness.record_model_event(
                turn,
                {
                    "content": response.content,
                    "action": response.action,
                    "tool_calls": [tool_call.manifest() for tool_call in response.tool_calls],
                    "finish_reason": response.finish_reason,
                    "usage": response.usage.__dict__,
                    "metadata": dict(response.metadata),
                },
            )
            await self._record_timeline(
                response.content or json.dumps(response.action or {}, ensure_ascii=False),
                kind=TIMELINE_KIND_MODEL,
                run=run,
                turn_id=turn.turn_id,
                iteration=index,
                status=response.finish_reason,
                tool_call_count=len(response.tool_calls),
            )
            model_diff = self.timeline.diff_since(timeline_cursor) if self.timeline is not None else None
            await self.turn_refresher.after_model_response(
                ModelResponseEvent(
                    task=task,
                    iteration=index,
                    run_id=run.run_id,
                    turn_id=turn.turn_id,
                    response=response,
                    timeline_diff=model_diff,
                )
            )
            if model_diff is not None:
                timeline_cursor = model_diff.next_cursor
            await self._emit(
                "model_response",
                run,
                turn_id=turn.turn_id,
                payload={"finish_reason": response.finish_reason},
            )

            if response.tool_calls:
                terminal_tool_call = next(
                    (
                        tool_call
                        for tool_call in response.tool_calls
                        if tool_call.tool_name == self.config.finish_action
                    ),
                    None,
                )
                if terminal_tool_call is not None:
                    action = ParsedAction(
                        name=self.config.finish_action,
                        arguments=dict(terminal_tool_call.arguments),
                        raw=terminal_tool_call.transport_payload(),
                    )
                    self._append_action_history(
                        iteration=index,
                        action=action,
                        status="finished",
                        ok=True,
                    )
                    result_output = _terminal_action_output(action, fallback=response.content)
                    if not result_output.strip():
                        feedback = self._feedback(
                            "finish_error",
                            "finish requires a non-empty output string. Return a final answer in the output field.",
                        )
                        compact_delta.append(feedback)
                        compact_delta = self._compact_loop_delta(compact_delta)
                        await self.harness.checkpoint(
                            turn,
                            {
                                "status": "finish_error",
                                "error": "missing_finish_output",
                                "iteration": index,
                                "provider_tool_call": terminal_tool_call.manifest(),
                            },
                        )
                        await self._record_timeline(
                            "finish tool call missing non-empty output",
                            kind=TIMELINE_KIND_ERROR,
                            run=run,
                            turn_id=turn.turn_id,
                            iteration=index,
                            status="finish_error",
                            provider_tool_call=terminal_tool_call.manifest(),
                        )
                        finish_error_diff = (
                            self.timeline.diff_since(timeline_cursor)
                            if self.timeline is not None
                            else None
                        )
                        if finish_error_diff is not None:
                            timeline_cursor = finish_error_diff.next_cursor
                        await self.turn_refresher.after_turn(
                            TurnCompletedEvent(
                                task=task,
                                iteration=index,
                                run_id=run.run_id,
                                turn_id=turn.turn_id,
                                status="action_error",
                                timeline_diff=finish_error_diff,
                            )
                        )
                        continue
                    final_action = action
                    await self.harness.checkpoint(
                        turn,
                        {
                            "status": "finished",
                            "output": result_output,
                            "iteration": index,
                            "provider_tool_call": terminal_tool_call.manifest(),
                        },
                    )
                    await self.harness.finish_run(run, "completed", {"output": result_output})
                    await self._record_timeline(
                        result_output,
                        kind=TIMELINE_KIND_FINAL,
                        run=run,
                        turn_id=turn.turn_id,
                        iteration=index,
                        action=action.name,
                        status="completed",
                        provider_tool_call=terminal_tool_call.manifest(),
                    )
                    terminal_diff = (
                        self.timeline.diff_since(timeline_cursor) if self.timeline is not None else None
                    )
                    if terminal_diff is not None:
                        timeline_cursor = terminal_diff.next_cursor
                    await self.turn_refresher.after_turn(
                        TurnCompletedEvent(
                            task=task,
                            iteration=index,
                            run_id=run.run_id,
                            turn_id=turn.turn_id,
                            status="finished",
                            timeline_diff=terminal_diff,
                        )
                    )
                    await self._emit(
                        "run_finished",
                        run,
                        turn_id=turn.turn_id,
                        payload={"status": "completed"},
                    )
                    return ReActResult(
                        run_id=run.run_id,
                        status="completed",
                        output=result_output,
                        iterations=index + 1,
                        final_action=action,
                    )
                try:
                    provider_tool_messages, provider_tool_results = await self._await_with_deadline(
                        self._execute_provider_tool_calls(
                            run,
                            turn,
                            response=response,
                            iteration=index,
                        ),
                        deadline,
                    )
                except asyncio.TimeoutError:
                    self.cancel_token.timeout(
                        "provider tool call timed out",
                        timeout_seconds=self.config.timeout_seconds,
                        metadata={"iteration": index, "phase": "provider_tool_call"},
                    )
                    return await self._finish_interrupted(
                        run,
                        status="timeout",
                        event_type="run_timeout",
                        output=self.cancel_token.reason or "provider tool call timed out",
                        iterations=index + 1,
                        turn=turn,
                        checkpoint_state={"phase": "provider_tool_call"},
                    )
                compact_delta.extend(provider_tool_messages)
                compact_delta = self._compact_loop_delta(compact_delta)
                provider_tool_diff = (
                    self.timeline.diff_since(timeline_cursor) if self.timeline is not None else None
                )
                provider_turn_cursor = (
                    provider_tool_diff.cursor
                    if provider_tool_diff is not None
                    else timeline_cursor
                )
                if provider_tool_diff is not None:
                    timeline_cursor = provider_tool_diff.next_cursor
                for provider_tool_result in provider_tool_results:
                    await self.turn_refresher.after_tool_result(
                        ToolResultEvent(
                            task=task,
                            iteration=index,
                            run_id=run.run_id,
                            turn_id=turn.turn_id,
                            tool_result=provider_tool_result,
                            timeline_diff=provider_tool_diff,
                        )
                    )
                await asyncio.sleep(0)
                if self.timeline is not None:
                    provider_tool_diff = self.timeline.diff_since(provider_turn_cursor)
                    timeline_cursor = provider_tool_diff.next_cursor
                await self.turn_refresher.after_turn(
                    TurnCompletedEvent(
                        task=task,
                        iteration=index,
                        run_id=run.run_id,
                        turn_id=turn.turn_id,
                        status="provider_tool_finished",
                        timeline_diff=provider_tool_diff,
                    )
                )
                continue

            try:
                if (
                    self.config.native_tool_calls
                    and response.action is None
                    and response.content.strip()
                    and response.finish_reason in {"stop", "length", ""}
                ):
                    action = ParsedAction(
                        name=self.config.finish_action,
                        arguments={"output": response.content},
                        raw=response.content,
                    )
                else:
                    action = self._parse_response_action(response.action, response.content)
                await self._emit(
                    "action_parsed",
                    run,
                    turn_id=turn.turn_id,
                    payload={"action": action.name},
                )
            except (ActionError, ProviderError) as exc:
                feedback = self._feedback("action_error", str(exc))
                compact_delta.extend([LLMMessage(role="assistant", content=response.content), feedback])
                compact_delta = self._compact_loop_delta(compact_delta)
                await self.harness.checkpoint(
                    turn,
                    {"status": "action_error", "error": str(exc), "iteration": index},
                )
                await self._record_timeline(
                    str(exc),
                    kind=TIMELINE_KIND_ERROR,
                    run=run,
                    turn_id=turn.turn_id,
                    iteration=index,
                    status="action_error",
                )
                action_error_diff = (
                    self.timeline.diff_since(timeline_cursor) if self.timeline is not None else None
                )
                if action_error_diff is not None:
                    timeline_cursor = action_error_diff.next_cursor
                await self.turn_refresher.after_turn(
                    TurnCompletedEvent(
                        task=task,
                        iteration=index,
                        run_id=run.run_id,
                        turn_id=turn.turn_id,
                        status="action_error",
                        timeline_diff=action_error_diff,
                    )
                )
                continue

            decision = await self.policy.check_action(action)
            approval_grant = None
            if not decision.allowed:
                approval_grant = self.approval_resume.approved_for(decision.approval)
            await self._record_policy_decision(
                decision,
                run,
                turn.turn_id,
                subject_kind="action",
                subject_name=action.name,
                metadata={
                    "argument_keys": sorted(str(key) for key in action.arguments),
                    "approval_resumed": approval_grant.manifest() if approval_grant else None,
                },
            )
            if not decision.allowed:
                if decision.status == "approval_required" and approval_grant is not None:
                    await self._emit_approval_resumed(run, turn.turn_id, approval_grant)
                else:
                    result_output = decision.reason or f"action denied: {action.name}"
                    status = "approval_required" if decision.status == "approval_required" else "denied"
                    approval_record = await self._submit_approval(
                        decision.approval,
                        run,
                        turn.turn_id,
                        metadata={"kind": "action", "action": action.name},
                    )
                    await self.harness.checkpoint(
                        turn,
                        {
                            "status": status,
                            "reason": result_output,
                            "approval": _approval_request_manifest(decision.approval),
                            "approval_record": approval_record.manifest() if approval_record else None,
                            "iteration": index,
                        },
                    )
                    await self.harness.finish_run(
                        run,
                        status,
                        {
                            "output": result_output,
                            "approval": _approval_request_manifest(decision.approval),
                            "approval_record": approval_record.manifest() if approval_record else None,
                        },
                    )
                    await self._emit(
                        "run_finished",
                        run,
                        turn_id=turn.turn_id,
                        payload={"status": status},
                    )
                    await self._record_timeline(
                        result_output,
                        kind=TIMELINE_KIND_APPROVAL,
                        run=run,
                        turn_id=turn.turn_id,
                        iteration=index,
                        action=action.name,
                        status=status,
                        approval=(
                            approval_record.manifest() if approval_record is not None else {}
                        ),
                    )
                    return ReActResult(
                        run_id=run.run_id,
                        status=status,
                        output=result_output,
                        iterations=index + 1,
                        final_action=action,
                        metadata=_approval_metadata(decision.approval, approval_record),
                    )

            verification = await self.action_verifier.verify(action)
            if not verification.ok:
                self._append_action_history(
                    iteration=index,
                    action=action,
                    status="verification_failed",
                    ok=False,
                    metadata={"verification_result": _verification_manifest(verification)},
                )
                feedback = self._feedback(
                    "action_verifier_error",
                    verification.message or f"action rejected by verifier: {action.name}",
                )
                compact_delta.extend([LLMMessage(role="assistant", content=response.content), feedback])
                compact_delta = self._compact_loop_delta(compact_delta)
                await self.harness.checkpoint(
                    turn,
                    {
                        "status": "action_verifier_error",
                        "action": action.name,
                        "error": verification.message,
                        "metadata": verification.metadata,
                        "iteration": index,
                    },
                )
                await self._record_timeline(
                    verification.message,
                    kind=TIMELINE_KIND_VERIFICATION,
                    run=run,
                    turn_id=turn.turn_id,
                    iteration=index,
                    action=action.name,
                    status="failed",
                    verification=_verification_manifest(verification),
                )
                await self._emit(
                    "error",
                    run,
                    turn_id=turn.turn_id,
                    payload={
                        "kind": "action_verifier_error",
                        "action": action.name,
                        "message": verification.message,
                    },
                )
                verifier_diff = (
                    self.timeline.diff_since(timeline_cursor) if self.timeline is not None else None
                )
                if verifier_diff is not None:
                    timeline_cursor = verifier_diff.next_cursor
                await self.turn_refresher.after_turn(
                    TurnCompletedEvent(
                        task=task,
                        iteration=index,
                        run_id=run.run_id,
                        turn_id=turn.turn_id,
                        status="action_verifier_error",
                        timeline_diff=verifier_diff,
                    )
                )
                continue

            if action.name == self.config.finish_action:
                self._append_action_history(
                    iteration=index,
                    action=action,
                    status="finished",
                    ok=True,
                )
                result_output = _terminal_action_output(action)
                final_action = action
                structured_output_result = None
                if self.config.structured_output is not None:
                    structured_output_result = await self.structured_output_validator.validate(
                        result_output,
                        self.config.structured_output,
                    )
                    if not structured_output_result.ok:
                        if structured_output_repairs < self.config.structured_output.max_repairs:
                            structured_output_repairs += 1
                            feedback = self._feedback(
                                "structured_output_error",
                                structured_output_feedback(
                                    structured_output_result,
                                    self.config.structured_output,
                                ),
                            )
                            compact_delta.extend(
                                [
                                    LLMMessage(
                                        role="assistant",
                                        content=response.content or json.dumps(action.raw),
                                    ),
                                    feedback,
                                ]
                            )
                            compact_delta = self._compact_loop_delta(compact_delta)
                            await self.harness.checkpoint(
                                turn,
                                {
                                    "status": "structured_output_error",
                                    "error": structured_output_result.error,
                                    "repair_attempt": structured_output_repairs,
                                    "iteration": index,
                                },
                            )
                            await self._record_timeline(
                                structured_output_result.error,
                                kind=TIMELINE_KIND_ERROR,
                                run=run,
                                turn_id=turn.turn_id,
                                iteration=index,
                                action=action.name,
                                status="structured_output_error",
                                repair_attempt=structured_output_repairs,
                            )
                            await self._emit(
                                "error",
                                run,
                                turn_id=turn.turn_id,
                                payload={
                                    "kind": "structured_output_error",
                                    "message": structured_output_result.error,
                                    "repair_attempt": structured_output_repairs,
                                },
                            )
                            repair_diff = (
                                self.timeline.diff_since(timeline_cursor)
                                if self.timeline is not None
                                else None
                            )
                            if repair_diff is not None:
                                timeline_cursor = repair_diff.next_cursor
                            await self.turn_refresher.after_turn(
                                TurnCompletedEvent(
                                    task=task,
                                    iteration=index,
                                    run_id=run.run_id,
                                    turn_id=turn.turn_id,
                                    status="structured_output_error",
                                    timeline_diff=repair_diff,
                                )
                            )
                            continue
                        await self.harness.checkpoint(
                            turn,
                            {
                                "status": "structured_output_failed",
                                "output": result_output,
                                "error": structured_output_result.error,
                                "iteration": index,
                            },
                        )
                        await self.harness.finish_run(
                            run,
                            "output_validation_failed",
                            {
                                "output": result_output,
                                "structured_output": structured_output_result.manifest(),
                            },
                        )
                        await self._record_timeline(
                            structured_output_result.error,
                            kind=TIMELINE_KIND_FINAL,
                            run=run,
                            turn_id=turn.turn_id,
                            iteration=index,
                            action=action.name,
                            status="output_validation_failed",
                            structured_output=structured_output_result.manifest(),
                        )
                        terminal_diff = (
                            self.timeline.diff_since(timeline_cursor)
                            if self.timeline is not None
                            else None
                        )
                        if terminal_diff is not None:
                            timeline_cursor = terminal_diff.next_cursor
                        await self.turn_refresher.after_turn(
                            TurnCompletedEvent(
                                task=task,
                                iteration=index,
                                run_id=run.run_id,
                                turn_id=turn.turn_id,
                                status="output_validation_failed",
                                timeline_diff=terminal_diff,
                            )
                        )
                        await self._emit(
                            "run_finished",
                            run,
                            turn_id=turn.turn_id,
                            payload={"status": "output_validation_failed"},
                        )
                        return ReActResult(
                            run_id=run.run_id,
                            status="output_validation_failed",
                            output=result_output,
                            iterations=index + 1,
                            final_action=action,
                            metadata={"structured_output": structured_output_result.manifest()},
                        )
                finish_metadata = {}
                if structured_output_result is not None:
                    finish_metadata["structured_output"] = structured_output_result.manifest()
                await self.harness.checkpoint(
                    turn,
                    {
                        "status": "finished",
                        "output": result_output,
                        "iteration": index,
                        **finish_metadata,
                    },
                )
                await self.harness.finish_run(
                    run,
                    "completed",
                    {"output": result_output, **finish_metadata},
                )
                await self._record_timeline(
                    result_output,
                    kind=TIMELINE_KIND_FINAL,
                    run=run,
                    turn_id=turn.turn_id,
                    iteration=index,
                    action=action.name,
                    status="completed",
                    **finish_metadata,
                )
                terminal_diff = (
                    self.timeline.diff_since(timeline_cursor) if self.timeline is not None else None
                )
                if terminal_diff is not None:
                    timeline_cursor = terminal_diff.next_cursor
                await self.turn_refresher.after_turn(
                    TurnCompletedEvent(
                        task=task,
                        iteration=index,
                        run_id=run.run_id,
                        turn_id=turn.turn_id,
                        status="finished",
                        timeline_diff=terminal_diff,
                    )
                )
                await self._emit(
                    "run_finished",
                    run,
                    turn_id=turn.turn_id,
                    payload={"status": "completed"},
                )
                return ReActResult(
                    run_id=run.run_id,
                    status="completed",
                    output=result_output,
                    iterations=index + 1,
                    final_action=action,
                    metadata=finish_metadata,
                )

            try:
                tool_result = await self._await_with_deadline(
                    self._execute_action(run, turn.turn_id, action),
                    deadline,
                )
            except asyncio.TimeoutError:
                self.cancel_token.timeout(
                    "tool call timed out",
                    timeout_seconds=self.config.timeout_seconds,
                    metadata={"iteration": index, "phase": "tool", "action": action.name},
                )
                return await self._finish_interrupted(
                    run,
                    status="timeout",
                    event_type="run_timeout",
                    output=self.cancel_token.reason or "tool call timed out",
                    iterations=index + 1,
                    turn=turn,
                    checkpoint_state={"phase": "tool", "action": action.name},
                    final_action=action,
                )
            prompt_tool_result = await self._prompt_safe_tool_result(tool_result)
            await self.harness.record_tool_call(
                turn,
                {
                    "call_id": prompt_tool_result.call_id,
                    "tool_name": prompt_tool_result.tool_name,
                    "status": prompt_tool_result.status,
                    "error": prompt_tool_result.error,
                    "metadata": prompt_tool_result.metadata,
                },
            )
            await self._record_timeline(
                prompt_tool_result.content if prompt_tool_result.ok else prompt_tool_result.error,
                kind=TIMELINE_KIND_TOOL,
                run=run,
                turn_id=turn.turn_id,
                iteration=index,
                action=action.name,
                tool_name=prompt_tool_result.tool_name,
                status=prompt_tool_result.status,
                call_id=prompt_tool_result.call_id,
            )
            tool_diff = self.timeline.diff_since(timeline_cursor) if self.timeline is not None else None
            turn_start_cursor = tool_diff.cursor if tool_diff is not None else timeline_cursor
            if tool_diff is not None:
                timeline_cursor = tool_diff.next_cursor
            self._append_action_history(
                iteration=index,
                action=action,
                status=prompt_tool_result.status,
                ok=prompt_tool_result.ok,
                tool_result=prompt_tool_result,
            )
            await self.turn_refresher.after_tool_result(
                ToolResultEvent(
                    task=task,
                    iteration=index,
                    run_id=run.run_id,
                    turn_id=turn.turn_id,
                    tool_result=prompt_tool_result,
                    action=action,
                    timeline_diff=tool_diff,
                    metadata={
                        "action_history": tuple(self._action_history[-12:]),
                        "task_metadata": self.config.budget.manifest()
                        if hasattr(self.config.budget, "manifest")
                        else {},
                    },
                )
            )
            await asyncio.sleep(0)
            if self.timeline is not None:
                tool_diff = self.timeline.diff_since(turn_start_cursor)
                timeline_cursor = tool_diff.next_cursor
            await self.harness.checkpoint(
                turn,
                {
                    "status": "tool_finished",
                    "tool_name": prompt_tool_result.tool_name,
                    "tool_status": prompt_tool_result.status,
                    "iteration": index,
                },
            )
            loop_decision = self.loop_guard.observe(
                action,
                tool_result=prompt_tool_result,
                iteration=index,
            )
            if loop_decision.stalled:
                result_output = loop_decision.message
                await self.harness.checkpoint(
                    turn,
                    {
                        "status": "loop_stalled",
                        "message": result_output,
                        "metadata": loop_decision.metadata,
                        "iteration": index,
                    },
                )
                await self.harness.finish_run(run, "stalled", {"output": result_output})
                await self._record_timeline(
                    result_output,
                    kind=TIMELINE_KIND_LOOP,
                    run=run,
                    turn_id=turn.turn_id,
                    iteration=index,
                    action=action.name,
                    status="loop_stalled",
                    loop_guard=loop_decision.metadata,
                )
                stalled_diff = (
                    self.timeline.diff_since(timeline_cursor) if self.timeline is not None else None
                )
                if stalled_diff is not None:
                    timeline_cursor = stalled_diff.next_cursor
                await self.turn_refresher.after_tool_result(
                    ToolResultEvent(
                        task=task,
                        iteration=index,
                        run_id=run.run_id,
                        turn_id=turn.turn_id,
                        tool_result=prompt_tool_result,
                        action=action,
                        timeline_diff=stalled_diff,
                        force_perception=True,
                        metadata={
                            "loop_guard": loop_decision.metadata,
                            "action_history": tuple(self._action_history[-12:]),
                        },
                    )
                )
                await asyncio.sleep(0)
                if self.timeline is not None:
                    stalled_diff = self.timeline.diff_since(turn_start_cursor)
                    timeline_cursor = stalled_diff.next_cursor
                await self.turn_refresher.after_turn(
                    TurnCompletedEvent(
                        task=task,
                        iteration=index,
                        run_id=run.run_id,
                        turn_id=turn.turn_id,
                        status="loop_stalled",
                        timeline_diff=stalled_diff,
                    )
                )
                await self._emit(
                    "run_finished",
                    run,
                    turn_id=turn.turn_id,
                    payload={"status": "stalled", "loop_guard": loop_decision.metadata},
                )
                return ReActResult(
                    run_id=run.run_id,
                    status="stalled",
                    output=result_output,
                    iterations=index + 1,
                    final_action=action,
                    metadata={"loop_guard": loop_decision.metadata},
                )
            compact_delta.extend(
                [
                    LLMMessage(role="assistant", content=response.content or json.dumps(action.raw)),
                    LLMMessage(
                        role="user",
                        name=prompt_tool_result.tool_name,
                        content=prompt_tool_result.content
                        if prompt_tool_result.ok
                        else prompt_tool_result.error,
                        metadata={
                            "tool_name": prompt_tool_result.tool_name,
                            "tool_status": prompt_tool_result.status,
                            "json_action_tool_result": True,
                        },
                    ),
                ]
            )
            compact_delta = self._compact_loop_delta(compact_delta)
            turn_diff = tool_diff
            if loop_decision.warning:
                feedback = self._feedback("loop_warning", loop_decision.message)
                compact_delta.append(feedback)
                compact_delta = self._compact_loop_delta(compact_delta)
                await self.harness.checkpoint(
                    turn,
                    {
                        "status": "loop_warning",
                        "message": loop_decision.message,
                        "metadata": loop_decision.metadata,
                        "iteration": index,
                    },
                )
                await self._record_timeline(
                    loop_decision.message,
                    kind=TIMELINE_KIND_LOOP,
                    run=run,
                    turn_id=turn.turn_id,
                    iteration=index,
                    action=action.name,
                    status="loop_warning",
                    loop_guard=loop_decision.metadata,
                )
                warning_diff = (
                    self.timeline.diff_since(timeline_cursor) if self.timeline is not None else None
                )
                if warning_diff is not None:
                    timeline_cursor = warning_diff.next_cursor
                    turn_diff = warning_diff
                await self.turn_refresher.after_tool_result(
                    ToolResultEvent(
                        task=task,
                        iteration=index,
                        run_id=run.run_id,
                        turn_id=turn.turn_id,
                    tool_result=prompt_tool_result,
                    action=action,
                    timeline_diff=warning_diff,
                    force_perception=True,
                    metadata={
                        "loop_guard": loop_decision.metadata,
                        "action_history": tuple(self._action_history[-12:]),
                    },
                )
                )
                await asyncio.sleep(0)
                if self.timeline is not None:
                    turn_diff = self.timeline.diff_since(turn_start_cursor)
                    timeline_cursor = turn_diff.next_cursor
                await self._emit(
                    "loop_warning",
                    run,
                    turn_id=turn.turn_id,
                    payload=loop_decision.metadata,
                )
            await self.turn_refresher.after_turn(
                TurnCompletedEvent(
                    task=task,
                    iteration=index,
                    run_id=run.run_id,
                    turn_id=turn.turn_id,
                    status="tool_finished",
                    timeline_diff=turn_diff,
                )
            )

        result_output = "max iterations reached"
        await self.harness.finish_run(run, "max_iterations", {"output": result_output})
        await self._record_timeline(
            result_output,
            kind=TIMELINE_KIND_LOOP,
            run=run,
            status="max_iterations",
            iteration=self.config.max_iterations,
            action=final_action.name if final_action is not None else "",
        )
        await self._emit("run_finished", run, payload={"status": "max_iterations"})
        return ReActResult(
            run_id=run.run_id,
            status="max_iterations",
            output=result_output,
            iterations=self.config.max_iterations,
            final_action=final_action,
        )

    async def _finish_interrupted(
        self,
        run: RunState,
        *,
        status: str,
        event_type: str,
        output: str,
        iterations: int,
        turn: TurnState | None = None,
        checkpoint_state: dict[str, Any] | None = None,
        final_action: ParsedAction | None = None,
    ) -> ReActResult:
        interrupt = self.cancel_token.interrupt
        if interrupt is None:
            interrupt = RunInterrupt(
                kind="timeout" if status == "timeout" else "cancelled",
                reason=output,
            )
        interrupt_manifest = interrupt.manifest()
        if turn is not None:
            await self.harness.checkpoint(
                turn,
                {
                    "status": status,
                    "reason": output,
                    "iteration": max(0, iterations - 1),
                    "interrupt": interrupt_manifest,
                    **dict(checkpoint_state or {}),
                },
            )
        await self.harness.finish_run(
            run,
            status,
            {"output": output, "interrupt": interrupt_manifest},
        )
        await self._record_timeline(
            output,
            kind=TIMELINE_KIND_LOOP,
            run=run,
            turn_id=turn.turn_id if turn is not None else "",
            iteration=max(0, iterations - 1),
            action=final_action.name if final_action is not None else "",
            status=status,
            interrupt=interrupt_manifest,
            checkpoint_state=dict(checkpoint_state or {}),
        )
        await self._emit(
            event_type,
            run,
            turn_id=turn.turn_id if turn is not None else "",
            payload={"reason": output, "status": status, "interrupt": interrupt_manifest},
        )
        return ReActResult(
            run_id=run.run_id,
            status=status,
            output=output,
            iterations=iterations,
            final_action=final_action,
            metadata={"interrupt": interrupt_manifest},
        )

    async def _await_with_deadline(self, awaitable: Awaitable[T], deadline: float | None) -> T:
        if deadline is None:
            return await awaitable
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            close = getattr(awaitable, "close", None)
            if callable(close):
                close()
            raise asyncio.TimeoutError
        return await asyncio.wait_for(awaitable, timeout=remaining)

    def _provider_messages(
        self,
        prompt: PromptIR,
        compact_delta: list[LLMMessage],
        *,
        extra_messages: list[LLMMessage] | tuple[LLMMessage, ...] = (),
        reserved_bytes: int = 0,
    ) -> list[LLMMessage]:
        target_bytes = max(1, self._provider_request_budget_bytes() - max(0, int(reserved_bytes)))
        tail_messages = self._provider_tail_messages(
            compact_delta,
            extra_messages=extra_messages,
            target_bytes=target_bytes,
        )
        tail_bytes = sum(_message_bytes(message) for message in tail_messages)
        prompt_overhead = _messages_bytes(_prompt_segment_messages(PromptIR()))
        prompt_target = max(1, target_bytes - tail_bytes - prompt_overhead)
        prompt_messages = self._prompt_messages_with_provider_budget(
            prompt,
            prompt_target,
            target_bytes=target_bytes,
            tail_bytes=tail_bytes,
        )
        provider_messages = [*prompt_messages, *tail_messages]
        if _messages_bytes(provider_messages) > target_bytes:
            prompt_messages = self._fit_prompt_messages_to_budget(
                prompt_messages,
                max(1, target_bytes - tail_bytes),
            )
            provider_messages = [*prompt_messages, *tail_messages]
        if _messages_bytes(provider_messages) > target_bytes:
            tail_messages = self._fit_tail_messages_to_budget(
                tail_messages,
                max(0, target_bytes - _messages_bytes(prompt_messages)),
            )
            provider_messages = [*prompt_messages, *tail_messages]
        if _messages_bytes(provider_messages) > target_bytes:
            prompt_messages = self._prompt_messages_with_provider_budget(
                prompt,
                max(1, target_bytes - _messages_bytes(tail_messages) - prompt_overhead),
                target_bytes=target_bytes,
                tail_bytes=_messages_bytes(tail_messages),
            )
            provider_messages = [*prompt_messages, *tail_messages]
        if _messages_bytes(provider_messages) > target_bytes:
            provider_messages = self._fit_prompt_messages_to_budget(provider_messages, target_bytes)
        return [
            *provider_messages,
        ]

    def _prompt_messages_with_provider_budget(
        self,
        prompt: PromptIR,
        prompt_target_bytes: int,
        *,
        target_bytes: int,
        tail_bytes: int,
    ) -> list[LLMMessage]:
        target = max(1, int(prompt_target_bytes))
        for _ in range(8):
            provider_prompt = prompt.trim_to_budget(
                target,
                protect_context_injections=True,
            )
            messages = _prompt_segment_messages(provider_prompt)
            total = _messages_bytes(messages) + max(0, int(tail_bytes))
            if total <= max(1, int(target_bytes)) or target <= 1:
                return messages
            target = max(1, target - (total - int(target_bytes)) - 16)
        return messages

    def _fit_prompt_messages_to_budget(
        self,
        messages: list[LLMMessage],
        target_bytes: int,
    ) -> list[LLMMessage]:
        target = max(1, int(target_bytes))
        fitted = list(messages)
        if _messages_bytes(fitted) <= target:
            return fitted
        trim_order = sorted(
            range(len(fitted)),
            key=lambda index: _provider_prompt_trim_priority(fitted[index]),
        )
        for index in trim_order:
            total = _messages_bytes(fitted)
            if total <= target:
                break
            message = fitted[index]
            if not message.metadata.get("agent_core_prompt"):
                continue
            other_bytes = total - _message_bytes(message)
            available = max(0, target - other_bytes)
            fitted[index] = _trim_message_to_budget(message, available)
        while len(fitted) > 1 and _messages_bytes(fitted) > target:
            removable = sorted(
                range(len(fitted)),
                key=lambda index: _provider_prompt_trim_priority(fitted[index]),
            )[0]
            fitted.pop(removable)
        if fitted and _messages_bytes(fitted) > target:
            fitted[0] = _trim_message_to_budget(fitted[0], target)
        return fitted

    def _provider_request_budget_bytes(self) -> int:
        return max(1, int(self.config.budget.max_prompt_bytes))

    def _provider_tail_messages(
        self,
        compact_delta: list[LLMMessage],
        *,
        extra_messages: list[LLMMessage] | tuple[LLMMessage, ...] = (),
        target_bytes: int,
    ) -> list[LLMMessage]:
        candidates = [*self._compact_loop_delta(compact_delta), *extra_messages]
        budget = min(
            max(0, int(self.config.loop_delta_max_bytes)),
            max(0, int(target_bytes) - _message_bytes(LLMMessage(role="user", content=""))),
        )
        if budget <= 0:
            return []
        selected: list[LLMMessage] = []
        used = 0
        for message in reversed(candidates):
            size = _message_bytes(message)
            if selected and used + size > budget:
                break
            if not selected and size > budget:
                continue
            selected.append(message)
            used += size
        return list(reversed(selected))

    def _fit_tail_messages_to_budget(
        self,
        messages: list[LLMMessage],
        target_bytes: int,
    ) -> list[LLMMessage]:
        target = max(0, int(target_bytes))
        if not messages or target <= 0:
            return []
        if _messages_bytes(messages) <= target:
            return messages
        fitted = list(messages)
        for index in range(len(fitted) - 1, -1, -1):
            if _messages_bytes(fitted) <= target:
                break
            message = fitted[index]
            if message.role not in {"tool", "user"}:
                continue
            other_bytes = _messages_bytes(fitted) - _message_bytes(message)
            fitted[index] = _trim_message_to_budget(message, max(0, target - other_bytes))
        while len(fitted) > 1 and _messages_bytes(fitted) > target:
            removable = next(
                (
                    index
                    for index, message in enumerate(fitted)
                    if message.role not in {"assistant", "tool"}
                ),
                -1,
            )
            if removable < 0:
                break
            fitted.pop(removable)
        if _messages_bytes(fitted) > target:
            return []
        return fitted

    def _prompt_profile_payload(
        self,
        prompt_manifest: dict[str, Any],
        *,
        refresh_manifest: dict[str, Any],
        compact_delta: list[LLMMessage],
        provider_messages: list[LLMMessage],
    ) -> dict[str, Any]:
        bucket_manifests = list(prompt_manifest.get("buckets") or ())
        metadata = dict(prompt_manifest.get("metadata") or {})
        prompt_bytes = int(prompt_manifest.get("prompt_bytes") or 0)
        delta_messages = [
            message
            for message in provider_messages
            if not bool(message.metadata.get("agent_core_prompt"))
        ]
        delta_bytes = sum(_message_bytes(message) for message in delta_messages)
        provider_bytes = sum(_message_bytes(message) for message in provider_messages)
        provider_prompt_bytes = 0
        if provider_messages:
            provider_prompt_bytes = sum(
                len(message.content.encode("utf-8"))
                for message in provider_messages
                if bool(message.metadata.get("agent_core_prompt"))
            )
        provider_budget = self._provider_request_budget_bytes()
        injection_decisions = (
            metadata.get("context_injections")
            or metadata.get("context_material_selection", {}).get("selections")
            or ()
        )
        return {
            "schema_version": "agent-core-prompt-profile/v1",
            "prompt_bytes": prompt_bytes,
            "provider_prompt_bytes": provider_prompt_bytes,
            "provider_message_count": len(provider_messages),
            "provider_message_bytes": provider_bytes,
            "provider_request_budget_bytes": provider_budget,
            "provider_request_over_budget": provider_bytes > provider_budget,
            "provider_cache": _provider_cache_profile(
                provider_messages,
                config=self.config,
            ),
            "compact_delta_count": len(delta_messages),
            "compact_delta_bytes": delta_bytes,
            "fresh_prompt_ratio": round(
                prompt_bytes / max(1, prompt_bytes + delta_bytes),
                6,
            ),
            "buckets": [
                {
                    "role": item.get("role"),
                    "bytes": item.get("bytes"),
                    "estimated_tokens": item.get("estimated_tokens"),
                    "cache_hint": item.get("cache_hint"),
                    "sha256": item.get("sha256"),
                    "included": item.get("included"),
                }
                for item in bucket_manifests
                if isinstance(item, dict)
            ],
            "section_observations": list(prompt_manifest.get("section_observations") or ()),
            "context_injection_count": len(injection_decisions)
            if isinstance(injection_decisions, list | tuple)
            else 0,
            "turn_refresh": refresh_manifest,
            "prompt_budget": dict(metadata.get("prompt_budget") or {}),
            "semantic_trim": dict(metadata.get("semantic_trim") or {}),
            "memory_recall": dict(metadata.get("memory_recall") or {}),
            "capability_selection": dict(metadata.get("capability_selection") or {}),
            "knowledge_recall": dict(metadata.get("knowledge_recall") or {}),
            "midterm_timeline_recall": dict(metadata.get("midterm_timeline_recall") or {}),
            "perception": dict(metadata.get("perception") or {}),
            "loop_state": dict(metadata.get("loop_state") or {}),
        }

    def _compact_loop_delta(self, messages: list[LLMMessage]) -> list[LLMMessage]:
        budget = max(0, int(self.config.loop_delta_max_bytes))
        if budget <= 0:
            return []
        selected: list[LLMMessage] = []
        used = 0
        for message in reversed(messages):
            size = _message_bytes(message)
            if selected and used + size > budget:
                break
            selected.append(message)
            used += size
        return list(reversed(selected))

    def _append_action_history(
        self,
        *,
        iteration: int,
        action: ParsedAction,
        status: str,
        ok: bool,
        tool_result: ToolResult | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        tool_name = ""
        if tool_result is not None:
            tool_name = tool_result.tool_name
        if not tool_name:
            raw_tool = action.arguments.get("tool_name") or action.arguments.get("name")
            tool_name = str(raw_tool or "")
        action_metadata = dict(metadata or {})
        if tool_result is not None and tool_result.metadata:
            action_metadata["tool_result_metadata"] = dict(tool_result.metadata)
        self._action_history.append(
            {
                "iteration": iteration,
                "action": action.name,
                "tool_name": tool_name,
                "status": status,
                "ok": bool(ok),
                "argument_keys": sorted(str(key) for key in action.arguments),
                "metadata": action_metadata,
            }
        )
        self._action_history = self._action_history[-32:]

    async def _call_provider(
        self,
        request: LLMRequest,
        *,
        run: RunState,
        turn_id: str,
    ) -> LLMResponse:
        request = replace(
            request,
            messages=tuple(request.messages),
            metadata={
                **request.metadata,
                "run_id": run.run_id,
                "turn_id": turn_id,
            },
        )
        if not self.config.stream:
            return await self.provider.complete(request)

        accumulator = LLMStreamAccumulator()
        async for event in self.provider.stream(request):
            accumulator.add(event)
            await self._emit(
                "model_stream",
                run,
                turn_id=turn_id,
                payload={
                    "type": event.type,
                    "delta_bytes": len(event.delta.encode("utf-8")),
                    "has_action": event.action is not None,
                    "error": event.error,
                },
            )
        return accumulator.response()

    async def _memory_message(self, task: str) -> LLMMessage | None:
        hits = await self.memory.search(MemoryQuery(query=task))
        if not hits:
            return None
        rendered = "\n".join(
            f"- {hit.source or 'memory'} ({hit.score:.3f}): {hit.content}" for hit in hits
        )
        return LLMMessage(role="user", name="memory", content=f"[memory]\n{rendered}")

    def _ensure_builtin_actions(self) -> None:
        if not self.action_registry.get(self.config.finish_action):
            self.action_registry.register(
                ActionSpec(
                    name=self.config.finish_action,
                    description="Finish",
                    parameters_schema={
                        "type": "object",
                        "properties": {"output": {"type": "string"}},
                    },
                    terminal=True,
                )
            )
        if not self.action_registry.get(self.config.tool_action):
            self.action_registry.register(
                ActionSpec(
                    name=self.config.tool_action,
                    description="Invoke tool",
                    parameters_schema={
                        "type": "object",
                        "required": ["tool_name"],
                        "properties": {
                            "tool_name": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                    },
                )
            )
        self._register_core_action(
            self.config.search_tools_action,
            "Search tools",
            {"required": ["query"], "properties": {"query": {"type": "string"}}},
        )
        self._register_core_action(
            self.config.search_skills_action,
            "Search skills",
            {"required": ["query"], "properties": {"query": {"type": "string"}}},
        )
        self._register_core_action(
            self.config.search_capabilities_action,
            "Search capabilities",
            {"required": ["query"], "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}},
        )
        self._register_core_action(
            self.config.load_capability_action,
            "Load capability",
            {
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string"},
                    "refresh": {"type": "boolean"},
                },
            },
        )
        self._register_core_action(
            self.config.query_mcp_servers_action,
            "List MCP servers",
            {"properties": {"include_disabled": {"type": "boolean"}, "refresh": {"type": "boolean"}}},
        )
        self._register_core_action(
            self.config.query_mcp_tools_action,
            "Search MCP tools",
            {
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "refresh": {"type": "boolean"},
                },
            },
        )
        self._register_core_action(
            self.config.load_skill_action,
            "Load skill",
            {"required": ["name"], "properties": {"name": {"type": "string"}}},
        )
        self._register_core_action(
            self.config.load_skill_resource_action,
            "Load skill resource",
            {
                "required": ["ref"],
                "properties": {"ref": {"type": "string"}, "offset": {"type": "integer"}},
            },
        )
        self._register_core_action(
            self.config.change_skill_view_offset_action,
            "Change skill view",
            {
                "required": ["view_id", "offset"],
                "properties": {"view_id": {"type": "string"}, "offset": {"type": "integer"}},
            },
        )

    def _register_core_action(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
    ) -> None:
        if self.action_registry.get(name):
            return
        self.action_registry.register(
            ActionSpec(
                name=name,
                description=description,
                parameters_schema={"type": "object", **schema},
            )
        )

    def _ensure_tool_actions(self) -> None:
        for spec in self.tool_runtime.specs():
            names = (spec.name, *spec.aliases)
            for name in names:
                if not name or self.action_registry.get(name):
                    continue
                self.action_registry.register(
                    ActionSpec(
                        name=name,
                        description=spec.description,
                        parameters_schema=spec.parameters_schema,
                    )
                )

    def _parse_response_action(self, action: dict[str, Any] | None, content: str) -> ParsedAction:
        if action is not None:
            return self.action_registry.parse(action)
        text = str(content or "").strip()
        if not text:
            raise ProviderError("provider returned empty content and no action")
        return self.action_registry.parse(text)

    def _native_tool_contracts(
        self,
        *,
        task: str = "",
        provider_messages: list[LLMMessage] | tuple[LLMMessage, ...] = (),
    ) -> tuple[LLMToolContract, ...]:
        if not self.config.native_tool_calls:
            return ()
        contracts = tuple(
            LLMToolContract.from_tool_spec(spec)
            for spec in self.tool_runtime.specs()
            if spec.enabled
        )
        if not contracts:
            return ()
        max_prompt = self._provider_request_budget_bytes()
        yaklang_tool_budget = min(16000, max(256, max_prompt // 4))
        budget = min(yaklang_tool_budget, max(0, max_prompt - _message_bytes(LLMMessage(role="user", content=""))))
        query = " ".join(
            part
            for part in (
                task,
                " ".join(_message_text(message) for message in tuple(provider_messages)[-4:]),
            )
            if part
        )
        ranked = _rank_native_tool_contracts(
            contracts,
            query=query,
            recent_tools=tuple(self._recent_tool_names()),
            failed_tools=tuple(self._failed_tool_names()),
            whitelist=tuple(self._loaded_capability_tool_names()),
        )
        return _stabilize_native_tool_contract_order(
            _select_native_tool_contracts_by_budget(ranked, budget, min_count=20)
        )

    def _native_tool_choice(self, contracts: tuple[LLMToolContract, ...] | None = None) -> LLMToolChoice | None:
        if not self.config.native_tool_calls:
            return None
        contracts = contracts if contracts is not None else self._native_tool_contracts()
        if not contracts:
            return None
        return LLMToolChoice(mode="auto")

    def _recent_tool_names(self) -> list[str]:
        names: list[str] = []
        for item in self._action_history[-8:]:
            tool_name = str(item.get("tool_name") or item.get("action") or "")
            if tool_name:
                names.append(tool_name)
        return names

    def _failed_tool_names(self) -> list[str]:
        names: list[str] = []
        for item in self._action_history[-12:]:
            if item.get("ok") is False:
                tool_name = str(item.get("tool_name") or item.get("action") or "")
                if tool_name:
                    names.append(tool_name)
        return names

    def _loaded_capability_tool_names(self) -> list[str]:
        names: list[str] = []
        for manifest in self._loaded_capabilities.values():
            kind = str(manifest.get("kind") or "")
            name = str(manifest.get("name") or "")
            if kind in {"tool", "mcp_tool"} and name:
                names.append(name)
        return names

    async def _execute_provider_tool_calls(
        self,
        run: RunState,
        turn: TurnState,
        *,
        response: LLMResponse,
        iteration: int,
    ) -> tuple[list[LLMMessage], list[ToolResult]]:
        messages = [
            LLMMessage(
                role="assistant",
                content=response.content,
                metadata={
                    "provider_tool_calls": [
                        tool_call.manifest() for tool_call in response.tool_calls
                    ],
                    "provider_tool_calls_payload": [
                        tool_call.transport_payload() for tool_call in response.tool_calls
                    ],
                },
            )
        ]
        tool_results: list[ToolResult] = []
        for tool_call in response.tool_calls:
            parsed = ParsedAction(
                name=tool_call.tool_name,
                arguments=dict(tool_call.arguments),
                raw=tool_call.transport_payload(),
            )
            tool_result = await self._execute_core_action(parsed)
            if tool_result is None:
                invocation = ToolInvocation(
                    tool_name=tool_call.tool_name,
                    arguments=dict(tool_call.arguments),
                    call_id=tool_call.call_id,
                    metadata={"provider_tool_call": tool_call.manifest()},
                )
                tool_result = await self._execute_tool_invocation(run, turn.turn_id, invocation)
            prompt_tool_result = await self._prompt_safe_tool_result(tool_result)
            tool_results.append(prompt_tool_result)
            await self.harness.record_tool_call(
                turn,
                {
                    "call_id": prompt_tool_result.call_id,
                    "tool_name": prompt_tool_result.tool_name,
                    "status": prompt_tool_result.status,
                    "error": prompt_tool_result.error,
                    "metadata": {
                        **prompt_tool_result.metadata,
                        "provider_tool_call": tool_call.manifest(),
                    },
                },
            )
            await self._record_timeline(
                prompt_tool_result.content if prompt_tool_result.ok else prompt_tool_result.error,
                kind=TIMELINE_KIND_TOOL,
                run=run,
                turn_id=turn.turn_id,
                iteration=iteration,
                action="provider_tool_call",
                tool_name=prompt_tool_result.tool_name,
                status=prompt_tool_result.status,
                call_id=prompt_tool_result.call_id,
                provider_tool_call=tool_call.manifest(),
            )
            await self.harness.checkpoint(
                turn,
                {
                    "status": "provider_tool_finished",
                    "tool_name": prompt_tool_result.tool_name,
                    "tool_status": prompt_tool_result.status,
                    "provider_tool_call": tool_call.manifest(),
                    "iteration": iteration,
                },
            )
            messages.append(
                LLMMessage(
                    role="tool",
                    name=prompt_tool_result.tool_name,
                    content=prompt_tool_result.content
                    if prompt_tool_result.ok
                    else prompt_tool_result.error,
                    metadata={
                        "provider_tool_call_id": tool_call.call_id,
                        "provider_tool_name": tool_call.tool_name,
                        "tool_status": prompt_tool_result.status,
                    },
                )
            )
        return messages, tool_results

    async def _execute_action(
        self,
        run: RunState,
        turn_id: str,
        action: ParsedAction,
    ) -> ToolResult:
        if action.name != self.config.tool_action:
            core_result = await self._execute_core_action(action)
            if core_result is not None:
                await self._emit(
                    "tool_finished",
                    run,
                    turn_id=turn_id,
                    payload={
                        "call_id": core_result.call_id,
                        "tool_name": core_result.tool_name,
                        "status": core_result.status,
                        "core_action": True,
                    },
                )
                return core_result
            direct_invocation = self._direct_tool_invocation(action)
            if direct_invocation is not None:
                return await self._execute_tool_invocation(run, turn_id, direct_invocation)
            return ToolResult(
                call_id="",
                tool_name=action.name,
                status="failed",
                error=f"unsupported non-terminal action: {action.name}",
            )
        tool_name = str(action.arguments.get("tool_name") or "")
        tool_args = action.arguments.get("arguments") or {}
        invocation = ToolInvocation(tool_name=tool_name, arguments=tool_args)
        return await self._execute_tool_invocation(run, turn_id, invocation)

    def _direct_tool_invocation(self, action: ParsedAction) -> ToolInvocation | None:
        if isinstance(self.tool_runtime, ToolRegistry) and self.tool_runtime.get(action.name):
            return ToolInvocation(tool_name=action.name, arguments=dict(action.arguments))
        for spec in self.tool_runtime.specs():
            if action.name == spec.name or action.name in spec.aliases:
                return ToolInvocation(tool_name=action.name, arguments=dict(action.arguments))
        return None

    async def _execute_tool_invocation(
        self,
        run: RunState,
        turn_id: str,
        invocation: ToolInvocation,
    ) -> ToolResult:
        invocation = self._with_tool_spec_metadata(invocation)
        decision = await self.policy.check_tool(invocation)
        approval_grant = None
        if not decision.allowed:
            approval_grant = self.approval_resume.approved_for(decision.approval)
        await self._record_policy_decision(
            decision,
            run,
            turn_id,
            subject_kind="tool",
            subject_name=invocation.tool_name,
            metadata={
                "call_id": invocation.call_id,
                "argument_keys": sorted(str(key) for key in invocation.arguments),
                "tool_spec": invocation.metadata.get("tool_spec"),
                "approval_resumed": approval_grant.manifest() if approval_grant else None,
            },
        )
        if not decision.allowed:
            if decision.status == "approval_required" and approval_grant is not None:
                await self._emit_approval_resumed(run, turn_id, approval_grant)
            else:
                status = "approval_required" if decision.status == "approval_required" else "denied"
                approval_record = await self._submit_approval(
                    decision.approval,
                    run,
                    turn_id,
                    metadata={"kind": "tool", "tool_name": invocation.tool_name},
                )
                return ToolResult(
                    call_id=invocation.call_id,
                    tool_name=invocation.tool_name,
                    status=status,
                    error=decision.reason or f"tool denied: {invocation.tool_name}",
                    metadata=_approval_metadata(decision.approval, approval_record),
                )
        replayed = await self.tool_replay.get(invocation)
        if replayed is not None:
            await self._emit(
                "tool_finished",
                run,
                turn_id=turn_id,
                payload={
                    "call_id": replayed.call_id,
                    "tool_name": replayed.tool_name,
                    "status": replayed.status,
                    "replayed": True,
                },
            )
            return ToolResult(
                call_id=replayed.call_id,
                tool_name=replayed.tool_name,
                status=replayed.status,
                content=replayed.content,
                data=dict(replayed.data),
                error=replayed.error,
                metadata={**replayed.metadata, "replayed": True},
            )
        await self._emit(
            "tool_started",
            run,
            turn_id=turn_id,
            payload={"call_id": invocation.call_id, "tool_name": invocation.tool_name},
        )
        execution_runtime = self.tool_runtime
        if not isinstance(execution_runtime, ToolExecutionCenter):
            execution_runtime = ToolExecutionCenter(
                execution_runtime,
                retry_policy=self.config.tool_retry_policy,
                metadata={"source": "react"},
            )
        result = await execution_runtime.invoke(invocation)
        if result.ok:
            await self.tool_replay.put(invocation, result)
        execution = result.metadata.get("tool_execution")
        await self._emit(
            "tool_finished",
            run,
            turn_id=turn_id,
            payload={
                "call_id": result.call_id,
                "tool_name": result.tool_name,
                "status": result.status,
                "execution": execution if isinstance(execution, dict) else None,
            },
        )
        return result

    async def _prompt_safe_tool_result(self, tool_result: ToolResult) -> ToolResult:
        budget = self.config.budget.max_tool_result_bytes
        if self.artifact_store is None or budget <= 0 or tool_result.content_bytes <= budget:
            return tool_result.compact(budget)
        artifact = await self.artifact_store.put_text(
            tool_result.content,
            metadata={
                "kind": "tool_result",
                "call_id": tool_result.call_id,
                "tool_name": tool_result.tool_name,
                "status": tool_result.status,
            },
        )
        compacted = tool_result.compact(budget)
        metadata = {
            **compacted.metadata,
            "artifact": artifact.manifest(),
        }
        notice = f"\n\n[artifact_ref uri={artifact.uri} sha256={artifact.sha256}]"
        return ToolResult(
            call_id=compacted.call_id,
            tool_name=compacted.tool_name,
            status=compacted.status,
            content=compacted.content + notice if compacted.content else compacted.content,
            data=dict(compacted.data),
            error=compacted.error,
            metadata=metadata,
        )

    def _with_tool_spec_metadata(self, invocation: ToolInvocation) -> ToolInvocation:
        spec = self._tool_spec_for_name(invocation.tool_name)
        if spec is None:
            return invocation
        return ToolInvocation(
            tool_name=invocation.tool_name,
            arguments=dict(invocation.arguments),
            call_id=invocation.call_id,
            metadata={
                **invocation.metadata,
                "tool_spec": tool_manifest_item(spec),
            },
        )

    def _tool_spec_for_name(self, tool_name: str):
        for spec in self.tool_runtime.specs():
            if tool_name == spec.name or tool_name in spec.aliases:
                return spec
        return None

    async def _execute_core_action(self, action: ParsedAction) -> ToolResult | None:
        name = action.name
        args = action.arguments
        if name == self.config.search_tools_action:
            search = getattr(self.tool_runtime, "search", None)
            if not callable(search):
                return self._core_action_result(name, "tool search is unavailable", status="failed")
            query = str(args.get("query") or "")
            results = search(query)
            if not results:
                return self._core_action_result(name, "no matching tools")
            lines = [f"- {spec.name}: {spec.description}".rstrip() for spec in results]
            return self._core_action_result(name, "\n".join(lines))

        if name == self.config.search_skills_action:
            if self.skills is None:
                return self._core_action_result(name, "skill search is unavailable", status="failed")
            query = str(args.get("query") or "")
            results = self.skills.search(query)
            if not results:
                return self._core_action_result(name, "no matching skills")
            return self._core_action_result(name, "\n".join(skill.brief() for skill in results))

        if name == self.config.search_capabilities_action:
            query = str(args.get("query") or "")
            limit = int(args.get("limit") or 12)
            catalog = self._current_capability_catalog()
            result = catalog.discover(CapabilityQuery(query=query, limit=limit))
            return self._core_action_result(
                name,
                result.render_prompt(),
                metadata={"capability_discovery": result.manifest()},
            )

        if name == self.config.load_capability_action:
            cap_name = str(args.get("name") or args.get("capability") or "").strip()
            cap_kind = str(args.get("kind") or "").strip()
            if not cap_name:
                return self._core_action_result(name, "capability name is required", status="failed")
            loaded = await self._load_capability(cap_name, kind=cap_kind, refresh=bool(args.get("refresh")))
            return self._core_action_result(
                name,
                loaded["content"],
                status=str(loaded.get("status") or "completed"),
                metadata={"load_capability": loaded},
            )

        if name == self.config.query_mcp_servers_action:
            if self.mcp is None:
                return self._core_action_result(name, "MCP is unavailable", status="failed")
            if args.get("refresh"):
                await self.mcp.refresh_status(fail_fast=False)
            include_disabled = bool(args.get("include_disabled"))
            lines = []
            for server in self.mcp.servers(include_disabled=include_disabled):
                state = self.mcp.state(server.name)
                status_value = state.status if state is not None else "registered"
                counts = ""
                if state is not None:
                    counts = f" tools={state.tool_count} resources={state.resource_count} prompts={state.prompt_count}"
                lines.append(f"- {server.name} transport={server.transport} status={status_value}{counts}")
            return self._core_action_result(
                name,
                "\n".join(lines) if lines else "no MCP servers",
                metadata={"mcp": self.mcp.manifest()},
            )

        if name == self.config.query_mcp_tools_action:
            if self.mcp is None:
                return self._core_action_result(name, "MCP is unavailable", status="failed")
            if args.get("refresh"):
                await self.mcp.refresh_status(fail_fast=False)
            query = str(args.get("query") or "")
            limit = int(args.get("limit") or 8)
            results = self.mcp.search(query, limit=limit) if query else self.mcp.specs()[: max(0, limit)]
            if not results:
                server_states = ", ".join(
                    f"{state.server_name}:{state.status}" for state in self.mcp.states()
                )
                content = "no matching MCP tools"
                if server_states:
                    content += f"\nserver_states: {server_states}"
                return self._core_action_result(name, content)
            lines = [
                f"- {spec.name}: {spec.description} tags={','.join(spec.tags)}".rstrip()
                for spec in results
            ]
            return self._core_action_result(
                name,
                "\n".join(lines),
                metadata={"matches": [tool_manifest_item(spec) for spec in results]},
            )

        if name == self.config.load_skill_action:
            if self.skills is None:
                return self._core_action_result(name, "skill loading is unavailable", status="failed")
            skill_name = str(args.get("name") or args.get("skill_name") or "")
            try:
                skill = self.skills.load(skill_name)
            except KeyError:
                return self._core_action_result(name, f"skill not found: {skill_name}", status="failed")
            return self._core_action_result(
                name,
                f"loaded skill: {skill.name}\n\n{self.skills.registry.render_prompt((skill,))}",
            )

        if name == self.config.load_skill_resource_action:
            if self.skills is None:
                return self._core_action_result(name, "skill resource loading is unavailable", status="failed")
            ref = str(args.get("ref") or args.get("resource") or "")
            offset = int(args.get("offset") or 1)
            try:
                view = self.skills.load_resource(ref, offset=offset)
            except (KeyError, OSError, ValueError) as exc:
                return self._core_action_result(name, str(exc), status="failed")
            rendered, truncated = view.render()
            return self._core_action_result(
                name,
                rendered,
                metadata={"view_id": view.view_id, "truncated": truncated},
            )

        if name == self.config.change_skill_view_offset_action:
            if self.skills is None:
                return self._core_action_result(name, "skill view offset is unavailable", status="failed")
            view_id = str(args.get("view_id") or "")
            offset = int(args.get("offset") or 1)
            try:
                view = self.skills.change_view_offset(view_id, offset)
            except KeyError:
                return self._core_action_result(name, f"skill view not found: {view_id}", status="failed")
            rendered, truncated = view.render()
            return self._core_action_result(
                name,
                rendered,
                metadata={"view_id": view.view_id, "truncated": truncated},
            )
        return None

    def _current_capability_catalog(self) -> CapabilityCatalog:
        return self.capability_catalog or CapabilityCatalog(
            actions=self.action_registry,
            tools=self.tool_runtime,
            skills=self.skills,
            mcp=self.mcp,
        )

    async def _load_capability(
        self,
        capability_name: str,
        *,
        kind: str = "",
        refresh: bool = False,
    ) -> dict[str, Any]:
        catalog = self._current_capability_catalog()
        query = CapabilityQuery(query=capability_name, limit=8)
        matches = catalog.discover(query).matches
        exact = [
            match
            for match in matches
            if match.name == capability_name and (not kind or match.kind == kind)
        ]
        match = exact[0] if exact else (matches[0] if matches else None)
        if match is None:
            return {
                "status": "failed",
                "content": f"capability not found: {capability_name}",
                "name": capability_name,
                "kind": kind,
            }
        if match.kind == "skill":
            if self.skills is None:
                return {"status": "failed", "content": "skill loading is unavailable", "match": match.manifest()}
            try:
                skill = self.skills.load(match.name)
            except KeyError:
                return {"status": "failed", "content": f"skill not found: {match.name}", "match": match.manifest()}
            self._remember_loaded_capability(match)
            return {
                "status": "completed",
                "content": f"loaded skill: {skill.name}\n\n{self.skills.registry.render_prompt((skill,))}",
                "match": match.manifest(),
            }
        if match.kind == "mcp_server" and self.mcp is not None:
            if refresh:
                await self.mcp.refresh_inventory(match.name, fail_fast=False)
            state = self.mcp.state(match.name)
            self._remember_loaded_capability(match)
            return {
                "status": "completed",
                "content": f"MCP server available: {match.name} status={state.status if state else 'registered'}",
                "match": match.manifest(),
                "state": state.manifest() if state is not None else {},
            }
        self._remember_loaded_capability(match)
        return {
            "status": "completed",
            "content": f"capability available: {match.kind}:{match.name}\n{match.description}",
            "match": match.manifest(),
            "load_hint": "Use call_tool for tools, load_skill for skills, or query_mcp_tools for MCP tools.",
        }

    def _remember_loaded_capability(self, match: Any) -> None:
        manifest = match.manifest() if hasattr(match, "manifest") else {}
        name = str(manifest.get("name") or "")
        if not name:
            return
        self._loaded_capabilities[name] = manifest
        if len(self._loaded_capabilities) > 24:
            for key in list(self._loaded_capabilities)[:-24]:
                self._loaded_capabilities.pop(key, None)

    @staticmethod
    def _core_action_result(
        action_name: str,
        content: str,
        *,
        status: str = "completed",
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        return ToolResult(
            call_id=f"core-action:{action_name}",
            tool_name=action_name,
            status=status,
            content=content,
            error="" if status == "completed" else content,
            metadata={"core_action": True, **(metadata or {})},
        )

    async def _record_timeline(
        self,
        content: str,
        *,
        kind: str,
        run: RunState | None = None,
        turn_id: str = "",
        iteration: int | None = None,
        action: str = "",
        status: str = "",
        **metadata: Any,
    ) -> None:
        if self.timeline is None:
            return
        item_metadata = _timeline_metadata(
            kind=kind,
            run=run,
            turn_id=turn_id,
            iteration=iteration,
            action=action,
            status=status,
            metadata=metadata,
        )
        item = self.timeline.add(content, kind=kind, **item_metadata)
        await self.event_sink.emit(
            AgentEvent(
                type="timeline_updated",
                run_id=run.run_id if run is not None else "",
                turn_id=turn_id,
                sequence=self._next_event_sequence(),
                payload={
                    "item_id": item.item_id,
                    "kind": item.kind,
                    "bytes": item.bytes,
                    "metadata": item.metadata,
                },
            )
        )

    async def _submit_approval(
        self,
        approval: ApprovalRequest | None,
        run: RunState,
        turn_id: str,
        *,
        metadata: dict[str, Any],
    ) -> ApprovalRecord | None:
        if approval is None:
            return None
        record = await self.approval_store.submit(
            approval,
            run_id=run.run_id,
            turn_id=turn_id,
            metadata=metadata,
        )
        await self._emit(
            "approval_requested",
            run,
            turn_id=turn_id,
            payload=record.manifest(),
        )
        await self._record_timeline(
            _render_approval_timeline(record),
            kind=TIMELINE_KIND_APPROVAL,
            run=run,
            turn_id=turn_id,
            status=record.status,
            approval=record.manifest(),
        )
        return record

    async def _record_policy_decision(
        self,
        decision: PolicyDecision,
        run: RunState,
        turn_id: str,
        *,
        subject_kind: PolicySubjectKind,
        subject_name: str,
        metadata: dict[str, Any],
    ) -> None:
        record = await self.policy_decision_store.submit(
            decision,
            subject_kind=subject_kind,
            subject_name=subject_name,
            run_id=run.run_id,
            turn_id=turn_id,
            metadata=metadata,
        )
        await self._emit(
            "policy_decision",
            run,
            turn_id=turn_id,
            payload={
                "decision_id": record.decision_id,
                "subject": record.subject,
                "status": decision.status,
            },
        )
        await self._record_timeline(
            _render_policy_decision_timeline(record.subject, decision),
            kind=TIMELINE_KIND_POLICY,
            run=run,
            turn_id=turn_id,
            action=subject_name,
            status=decision.status,
            decision_id=record.decision_id,
            subject=record.subject,
            decision=decision.manifest(),
        )

    async def _emit_approval_resumed(
        self,
        run: RunState,
        turn_id: str,
        grant: ApprovalGrant,
    ) -> None:
        await self._emit(
            "approval_resumed",
            run,
            turn_id=turn_id,
            payload={"grant": grant.manifest()},
        )
        await self._record_timeline(
            f"approval resumed: {grant.subject}",
            kind=TIMELINE_KIND_APPROVAL,
            run=run,
            turn_id=turn_id,
            status="resumed",
            approval=grant.manifest(),
        )

    @staticmethod
    def _feedback(kind: str, message: str) -> LLMMessage:
        return LLMMessage(
            role="user",
            content=json.dumps({"feedback": kind, "message": message}, ensure_ascii=False),
        )

    async def _emit(
        self,
        event_type: str,
        run: RunState,
        *,
        turn_id: str = "",
        payload: dict[str, Any] | None = None,
    ) -> None:
        await self.event_sink.emit(
            AgentEvent(
                type=event_type,
                run_id=run.run_id,
                turn_id=turn_id,
                sequence=self._next_event_sequence(),
                payload=payload or {},
            )
        )

    def _next_event_sequence(self) -> int:
        self._event_sequence += 1
        return self._event_sequence


def _provider_request_metadata(config: ReActConfig) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    budget = config.budget
    if budget.max_cost_usd is not None:
        metadata["max_cost_usd"] = budget.max_cost_usd
    cache_mode = str(config.provider_cache_mode or "strip").strip().lower()
    if cache_mode not in {"", "strip", "off", "none", "disabled"}:
        metadata["provider_cache_policy"] = {
            "mode": cache_mode,
            "min_segment_bytes": max(0, int(config.provider_cache_min_segment_bytes)),
            "cache_control": {"type": "ephemeral"},
        }
    return metadata


def _provider_cache_profile(
    messages: list[LLMMessage] | tuple[LLMMessage, ...],
    *,
    config: ReActConfig,
) -> dict[str, Any]:
    mode = str(config.provider_cache_mode or "strip").strip().lower() or "strip"
    cacheable = []
    for message in messages:
        hint = message.metadata.get("cache_hint")
        if not isinstance(hint, dict) or hint.get("cacheable") is not True:
            continue
        cacheable.append(
            {
                "segment": str(message.metadata.get("agent_core_prompt_segment") or ""),
                "role": message.role,
                "bytes": len(message.content.encode("utf-8")),
                "policy": str(hint.get("policy") or ""),
            }
        )
    return {
        "schema_version": "agent-core-provider-cache-profile/v1",
        "mode": mode,
        "explicit_cache_control": mode not in {"strip", "off", "none", "disabled"},
        "min_segment_bytes": max(0, int(config.provider_cache_min_segment_bytes)),
        "cacheable_segment_count": len(cacheable),
        "cacheable_bytes": sum(int(item["bytes"]) for item in cacheable),
        "segments": cacheable,
    }


def _terminal_action_output(action: ParsedAction, *, fallback: str = "") -> str:
    for key in ("output", "final", "answer", "summary", "result", "message", "finding", "terminal"):
        value = action.arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if value is not None and not isinstance(value, bool | dict | list | tuple | set):
            text = str(value).strip()
            if text:
                return text
    return str(fallback or "")


def _timeline_metadata(
    *,
    kind: str,
    run: RunState | None,
    turn_id: str,
    iteration: int | None,
    action: str,
    status: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": "agent-core-timeline-event/v1",
        "source": TIMELINE_SOURCE_REACT,
        "event_kind": kind,
        **dict(metadata),
    }
    if run is not None:
        result["run_id"] = run.run_id
    if turn_id:
        result["turn_id"] = turn_id
    if iteration is not None:
        result["iteration"] = int(iteration)
    if action:
        result["action"] = action
    if status:
        result["status"] = status
    return result


def _render_policy_decision_timeline(subject: str, decision: PolicyDecision) -> str:
    parts = [f"policy decision: {subject}", f"status={decision.status}"]
    if decision.reason:
        parts.append(f"reason={decision.reason}")
    if decision.approval is not None:
        parts.append(f"approval_subject={decision.approval.subject}")
    return " ".join(parts)


def _render_approval_timeline(record: ApprovalRecord) -> str:
    request = record.request
    parts = [
        f"approval requested: {request.subject}",
        f"status={record.status}",
    ]
    if request.reason:
        parts.append(f"reason={request.reason}")
    return " ".join(parts)


def _message_bytes(message: LLMMessage) -> int:
    payload: dict[str, Any] = {
        "role": message.role,
    }
    if message.name:
        payload["name"] = message.name
    if message.role == "assistant":
        tool_calls = _provider_tool_calls_payload(message)
        if tool_calls:
            payload["tool_calls"] = tool_calls
    if message.role == "tool":
        tool_call_id = str(message.metadata.get("provider_tool_call_id") or message.name or "")
        if tool_call_id:
            payload["tool_call_id"] = tool_call_id
    payload["content"] = _provider_message_content_payload(message)
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def _messages_bytes(messages: list[LLMMessage] | tuple[LLMMessage, ...]) -> int:
    return sum(_message_bytes(message) for message in messages)


def _rank_native_tool_contracts(
    contracts: tuple[LLMToolContract, ...],
    *,
    query: str,
    recent_tools: tuple[str, ...] = (),
    failed_tools: tuple[str, ...] = (),
    whitelist: tuple[str, ...] = (),
) -> tuple[LLMToolContract, ...]:
    terms = _tokenize_for_budget(query)
    recent = {item.casefold() for item in recent_tools if item}
    failed = {item.casefold() for item in failed_tools if item}
    preferred = {item.casefold() for item in whitelist if item}
    scored: list[tuple[float, str, LLMToolContract]] = []
    for index, contract in enumerate(contracts):
        name_key = contract.name.casefold()
        haystack = " ".join(
            (
                contract.name,
                contract.description,
                json.dumps(contract.parameters_schema, ensure_ascii=False, sort_keys=True),
            )
        ).casefold()
        score = 1.0 / (index + 1)
        for term in terms:
            if term in haystack:
                score += 1.0
            if term == name_key:
                score += 1.0
        if name_key in recent:
            score += 0.5
        if name_key in failed:
            score -= 0.5
        if name_key in preferred:
            score += 4.0
        scored.append((score, contract.name, contract))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(contract for _, _, contract in scored)


def _stabilize_native_tool_contract_order(
    contracts: tuple[LLMToolContract, ...],
) -> tuple[LLMToolContract, ...]:
    priority = {
        name: index
        for index, name in enumerate(
            (
                "exec",
                "bash",
                "cmd",
                "read_file",
                "write_file",
                "modify_file",
                "grep",
                "find_file",
                "web_search",
            )
        )
    }
    return tuple(
        sorted(
            contracts,
            key=lambda contract: (
                priority.get(contract.name, len(priority)),
                contract.name,
            ),
        )
    )


def _select_native_tool_contracts_by_budget(
    contracts: tuple[LLMToolContract, ...],
    budget_bytes: int,
    *,
    min_count: int,
) -> tuple[LLMToolContract, ...]:
    if not contracts:
        return ()
    budget = max(0, int(budget_bytes))
    if len(contracts) <= min_count and _native_tool_contracts_bytes(contracts) <= budget:
        return contracts
    selected: list[LLMToolContract] = []
    used = 0
    for contract in contracts:
        size = _native_tool_contract_bytes(contract)
        if selected and budget > 0 and used + size > budget:
            break
        if not selected and budget > 0 and size > budget:
            selected.append(contract)
            break
        selected.append(contract)
        used += size
    return tuple(selected)


def _native_tool_contract_bytes(contract: LLMToolContract) -> int:
    payload = {
        "type": "function",
        "function": {
            "name": contract.name,
            "description": contract.description,
            "parameters": contract.parameters_schema,
            "strict": contract.strict,
        },
    }
    return len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def _native_tool_contracts_bytes(contracts: tuple[LLMToolContract, ...]) -> int:
    return sum(_native_tool_contract_bytes(contract) for contract in contracts)


def _tokenize_for_budget(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    current: list[str] = []
    for char in text.casefold():
        if char.isalnum() or char in {"_", "-"}:
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    seen: set[str] = set()
    unique: list[str] = []
    for token in tokens:
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        unique.append(token)
        if len(unique) >= 64:
            break
    return tuple(unique)


def _message_text(message: LLMMessage) -> str:
    parts = [message.content]
    for part in message.content_parts:
        text = getattr(part, "text", "")
        if text:
            parts.append(str(text))
    return " ".join(part for part in parts if part)


def _provider_message_content_payload(message: LLMMessage) -> Any:
    if not message.content_parts:
        return message.content
    parts: list[dict[str, Any]] = []
    if message.content:
        parts.append({"type": "text", "text": message.content})
    for part in message.content_parts:
        if part.kind in {"text", "json"}:
            parts.append({"type": "text", "text": part.text})
        elif part.kind == "image":
            image_url: dict[str, Any] = {"url": part.uri or part.text}
            detail = str(part.metadata.get("detail") or "")
            if detail:
                image_url["detail"] = detail
            parts.append({"type": "image_url", "image_url": image_url})
        else:
            parts.append({"type": "text", "text": part.text or part.uri})
    return parts


def _provider_tool_calls_payload(message: LLMMessage) -> list[dict[str, Any]]:
    raw_calls = message.metadata.get("provider_tool_calls_payload") or ()
    calls: list[dict[str, Any]] = []
    try:
        iterator = iter(raw_calls)
    except TypeError:
        return calls
    for raw in iterator:
        if not isinstance(raw, dict):
            continue
        call_id = str(raw.get("call_id") or raw.get("id") or "").strip()
        name = str(raw.get("tool_name") or raw.get("name") or "").strip()
        arguments = raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {}
        if not call_id or not name:
            continue
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False, sort_keys=True),
                },
            }
        )
    return calls


def _provider_prompt_trim_priority(message: LLMMessage) -> int:
    segment = str(message.metadata.get("agent_core_prompt_segment") or "")
    priorities = {
        "timeline_open_dynamic": 0,
        "semi_dynamic_1": 1,
        "semi_dynamic": 1,
        "semi_dynamic_2": 2,
        "frozen": 3,
        "high_static": 4,
        "empty": 4,
    }
    return priorities.get(segment, 0)


def _trim_message_to_budget(message: LLMMessage, target_bytes: int) -> LLMMessage:
    target = max(0, int(target_bytes))
    if _message_bytes(message) <= target:
        return message
    empty = replace(
        message,
        content="",
        content_parts=(),
        metadata={**message.metadata, "provider_budget_trimmed": True},
    )
    overhead = _message_bytes(empty)
    available = max(0, target - overhead)
    content = _trim_text_payload_to_bytes(message.content, available)
    trimmed = replace(
        message,
        content=content,
        content_parts=(),
        metadata={**message.metadata, "provider_budget_trimmed": True},
    )
    while _message_bytes(trimmed) > target and trimmed.content:
        excess = _message_bytes(trimmed) - target
        next_budget = max(0, len(trimmed.content.encode("utf-8")) - excess - 8)
        trimmed = replace(trimmed, content=_trim_text_payload_to_bytes(trimmed.content, next_budget))
    return trimmed


def _trim_text_payload_to_bytes(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    payload = text.encode("utf-8")
    if len(payload) <= max_bytes:
        return text
    marker = "\n[...provider request budget trimmed...]"
    marker_bytes = marker.encode("utf-8")
    if max_bytes <= len(marker_bytes):
        return payload[:max_bytes].decode("utf-8", errors="ignore")
    keep = max_bytes - len(marker_bytes)
    head_budget = max(1, keep // 2)
    tail_budget = max(1, keep - head_budget)
    head = payload[:head_budget].decode("utf-8", errors="ignore").rstrip()
    tail = payload[-tail_budget:].decode("utf-8", errors="ignore").lstrip()
    return f"{head}{marker}{tail}".strip()


def _prompt_segment_messages(prompt: PromptIR) -> list[LLMMessage]:
    ordered = {bucket.role: bucket for bucket in prompt.ordered_buckets()}
    segments: list[LLMMessage] = []

    high_static = ordered.get(PromptBucketRole.HIGH_STATIC, PromptBucket(PromptBucketRole.HIGH_STATIC))
    if high_static.content:
        segments.append(
            _prompt_segment_message(
                "system",
                "high_static",
                (high_static,),
            )
        )

    frozen = ordered.get(PromptBucketRole.FROZEN, PromptBucket(PromptBucketRole.FROZEN))
    if frozen.content:
        segments.append(_prompt_segment_message("user", "frozen", (frozen,)))

    semi_dynamic_1 = ordered.get(
        PromptBucketRole.SEMI_DYNAMIC_1,
        PromptBucket(PromptBucketRole.SEMI_DYNAMIC_1),
    )
    semi_dynamic_2 = ordered.get(
        PromptBucketRole.SEMI_DYNAMIC_2,
        PromptBucket(PromptBucketRole.SEMI_DYNAMIC_2),
    )
    if semi_dynamic_1.content or semi_dynamic_2.content:
        segments.append(_prompt_segment_message("user", "semi_dynamic_1", (semi_dynamic_1,)))
        segments.append(_prompt_segment_message("user", "semi_dynamic_2", (semi_dynamic_2,)))

    open_buckets = tuple(
        bucket
        for role in (PromptBucketRole.TIMELINE_OPEN, PromptBucketRole.DYNAMIC)
        if (bucket := ordered.get(role, PromptBucket(role))).content
    )
    if open_buckets:
        segments.append(_prompt_segment_message("user", "timeline_open_dynamic", open_buckets))

    if not segments:
        segments.append(
            LLMMessage(
                role="user",
                content="",
                metadata={
                    "agent_core_prompt": True,
                    "agent_core_prompt_segment": "empty",
                    "cache_hint": {"cacheable": False, "policy": "empty_prompt"},
                },
            )
        )
    return segments


def _prompt_segment_message(
    role: str,
    segment: str,
    buckets: tuple[PromptBucket, ...],
) -> LLMMessage:
    content = "\n\n".join(rendered for bucket in buckets if (rendered := bucket.render()))
    hints = [bucket.cache_hint or default_cache_hint(bucket.role) for bucket in buckets]
    cacheable = all(hint.cacheable for hint in hints)
    policies = [hint.policy for hint in hints]
    return LLMMessage(
        role=role,  # type: ignore[arg-type]
        content=content,
        metadata={
            "agent_core_prompt": True,
            "agent_core_prompt_segment": segment,
            "bucket_roles": [bucket.role.value for bucket in buckets],
            "cache_hint": {
                "cacheable": cacheable,
                "policy": "+".join(policy for policy in policies if policy),
            },
        },
    )


def _verification_manifest(verification: Any) -> dict[str, Any]:
    manifest = getattr(verification, "manifest", None)
    if callable(manifest):
        value = manifest()
        return dict(value) if isinstance(value, dict) else {}
    return {
        "ok": bool(getattr(verification, "ok", False)),
        "message": str(getattr(verification, "message", "") or ""),
        "metadata": dict(getattr(verification, "metadata", {}) or {}),
    }


def _structured_response_format(spec: StructuredOutputSpec | None) -> LLMResponseFormat | None:
    if spec is None:
        return None
    return LLMResponseFormat(
        kind="json_schema",
        name=spec.name,
        description=spec.description,
        schema=spec.schema,
        strict=True,
        metadata={"source": "structured_output"},
    )


def _deadline_after(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        return None
    if timeout_seconds <= 0:
        return time.monotonic()
    return time.monotonic() + timeout_seconds


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _approval_request_manifest(approval: ApprovalRequest | None) -> dict[str, Any] | None:
    if approval is None:
        return None
    return approval.manifest()


def _approval_metadata(
    approval: ApprovalRequest | None,
    record: ApprovalRecord | None,
) -> dict[str, Any]:
    if approval is None:
        return {}
    metadata: dict[str, Any] = {"approval": _approval_request_manifest(approval)}
    if record is not None:
        metadata["approval_record"] = record.manifest()
    return metadata

