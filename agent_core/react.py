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
from agent_core.prompt import PromptIR
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

T = TypeVar("T")


@dataclass(frozen=True)
class ReActConfig:
    model: str = ""
    max_iterations: int = 12
    finish_action: str = "finish"
    tool_action: str = "call_tool"
    search_tools_action: str = "search_tools"
    search_skills_action: str = "search_skills"
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
        timeline: TimelineStore | None = None,
        tool_replay: ToolReplayPort | None = None,
        action_verifier: ActionVerifierPort | None = None,
        structured_output_validator: StructuredOutputValidatorPort | None = None,
        loop_guard: LoopGuard | None = None,
        artifact_store: ArtifactStorePort | None = None,
        cancel_token: CancelToken | None = None,
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
        self.timeline = timeline
        self.tool_replay = tool_replay or NullToolReplay()
        self.action_verifier = action_verifier or NullActionVerifier()
        self.structured_output_validator = structured_output_validator or JsonStructuredOutputValidator()
        self.cancel_token = cancel_token or CancelToken()
        self.config = config or ReActConfig()
        self.loop_guard = loop_guard or LoopGuard(self.config.loop_guard)
        self.artifact_store = artifact_store
        self._event_sequence = 0
        self._ensure_builtin_actions()
        self._ensure_tool_actions()

    async def run(self, task: str, prompt: PromptIR) -> ReActResult:
        run = await self.harness.start_run(task)
        await self._emit("run_started", run, payload={"task": task})
        await self._record_timeline(task, kind="task")
        deadline = _deadline_after(self.config.timeout_seconds)
        messages = [LLMMessage(role="user", content=prompt.render())]
        memory_message = await self._memory_message(task)
        if memory_message is not None:
            messages.append(memory_message)
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
            await self.harness.record_prompt(turn, prompt.manifest())
            await self._emit(
                "prompt_ready",
                run,
                turn_id=turn.turn_id,
                payload={"manifest": prompt.manifest()},
            )

            try:
                response = await self._await_with_deadline(
                    self._call_provider(
                        LLMRequest(
                            messages=list(messages),
                            model=self.config.model,
                            tools=self._native_tool_contracts(),
                            tool_choice=self._native_tool_choice(),
                            response_format=_structured_response_format(
                                self.config.structured_output
                            ),
                            metadata=_provider_request_metadata(self.config.budget),
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
                kind="model",
            )
            await self._emit(
                "model_response",
                run,
                turn_id=turn.turn_id,
                payload={"finish_reason": response.finish_reason},
            )

            if response.tool_calls:
                try:
                    provider_tool_messages = await self._await_with_deadline(
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
                messages.extend(provider_tool_messages)
                continue

            try:
                action = self._parse_response_action(response.action, response.content)
                await self._emit(
                    "action_parsed",
                    run,
                    turn_id=turn.turn_id,
                    payload={"action": action.name},
                )
            except ActionError as exc:
                feedback = self._feedback("action_error", str(exc))
                messages.extend([LLMMessage(role="assistant", content=response.content), feedback])
                await self.harness.checkpoint(
                    turn,
                    {"status": "action_error", "error": str(exc), "iteration": index},
                )
                await self._record_timeline(str(exc), kind="action_error")
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
                feedback = self._feedback(
                    "action_verifier_error",
                    verification.message or f"action rejected by verifier: {action.name}",
                )
                messages.extend([LLMMessage(role="assistant", content=response.content), feedback])
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
                await self._record_timeline(verification.message, kind="action_verifier_error")
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
                continue

            if action.name == self.config.finish_action:
                result_output = str(action.arguments.get("output") or action.arguments.get("answer") or "")
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
                            messages.extend(
                                [
                                    LLMMessage(
                                        role="assistant",
                                        content=response.content or json.dumps(action.raw),
                                    ),
                                    feedback,
                                ]
                            )
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
                                kind="structured_output_error",
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
                kind="tool",
                tool_name=prompt_tool_result.tool_name,
                status=prompt_tool_result.status,
            )
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
                await self._record_timeline(result_output, kind="loop_stalled")
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
            messages.extend(
                [
                    LLMMessage(role="assistant", content=response.content or json.dumps(action.raw)),
                    LLMMessage(
                        role="tool",
                        name=prompt_tool_result.tool_name,
                        content=prompt_tool_result.content
                        if prompt_tool_result.ok
                        else prompt_tool_result.error,
                    ),
                ]
            )
            if loop_decision.warning:
                feedback = self._feedback("loop_warning", loop_decision.message)
                messages.append(feedback)
                await self.harness.checkpoint(
                    turn,
                    {
                        "status": "loop_warning",
                        "message": loop_decision.message,
                        "metadata": loop_decision.metadata,
                        "iteration": index,
                    },
                )
                await self._record_timeline(loop_decision.message, kind="loop_warning")
                await self._emit(
                    "loop_warning",
                    run,
                    turn_id=turn.turn_id,
                    payload=loop_decision.metadata,
                )

        result_output = "max iterations reached"
        await self.harness.finish_run(run, "max_iterations", {"output": result_output})
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
                    description="Finish the run",
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
                    description="Invoke one tool",
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
            "Search visible tools by query",
            {"required": ["query"], "properties": {"query": {"type": "string"}}},
        )
        self._register_core_action(
            self.config.search_skills_action,
            "Search available skills by query",
            {"required": ["query"], "properties": {"query": {"type": "string"}}},
        )
        self._register_core_action(
            self.config.load_skill_action,
            "Load one skill into the active skills context",
            {"required": ["name"], "properties": {"name": {"type": "string"}}},
        )
        self._register_core_action(
            self.config.load_skill_resource_action,
            "Load a skill resource view window",
            {
                "required": ["ref"],
                "properties": {"ref": {"type": "string"}, "offset": {"type": "integer"}},
            },
        )
        self._register_core_action(
            self.config.change_skill_view_offset_action,
            "Change the offset of a loaded skill resource view window",
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

    def _native_tool_contracts(self) -> tuple[LLMToolContract, ...]:
        if not self.config.native_tool_calls:
            return ()
        return tuple(
            LLMToolContract.from_tool_spec(spec)
            for spec in self.tool_runtime.specs()
            if spec.enabled
        )

    def _native_tool_choice(self) -> LLMToolChoice | None:
        if not self.config.native_tool_calls:
            return None
        contracts = self._native_tool_contracts()
        if not contracts:
            return None
        return LLMToolChoice(mode="auto")

    async def _execute_provider_tool_calls(
        self,
        run: RunState,
        turn: TurnState,
        *,
        response: LLMResponse,
        iteration: int,
    ) -> list[LLMMessage]:
        messages = [
            LLMMessage(
                role="assistant",
                content=response.content,
                metadata={
                    "provider_tool_calls": [
                        tool_call.manifest() for tool_call in response.tool_calls
                    ],
                },
            )
        ]
        for tool_call in response.tool_calls:
            invocation = ToolInvocation(
                tool_name=tool_call.tool_name,
                arguments=dict(tool_call.arguments),
                call_id=tool_call.call_id,
                metadata={"provider_tool_call": tool_call.manifest()},
            )
            tool_result = await self._execute_tool_invocation(run, turn.turn_id, invocation)
            prompt_tool_result = await self._prompt_safe_tool_result(tool_result)
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
                kind="tool",
                tool_name=prompt_tool_result.tool_name,
                status=prompt_tool_result.status,
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
        return messages

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

    async def _record_timeline(self, content: str, *, kind: str, **metadata: Any) -> None:
        if self.timeline is None:
            return
        item = self.timeline.add(content, kind=kind, **metadata)
        await self.event_sink.emit(
            AgentEvent(
                type="timeline_updated",
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


def _provider_request_metadata(budget: RuntimeBudget) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if budget.max_cost_usd is not None:
        metadata["max_cost_usd"] = budget.max_cost_usd
    return metadata


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

