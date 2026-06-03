"""Provider-neutral ReAct executor."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agent_core.actions import (
    ActionRegistry,
    ActionSpec,
    ActionVerifierPort,
    NullActionVerifier,
    ParsedAction,
)
from agent_core.approvals import ApprovalRecord, ApprovalStorePort, NullApprovalStore
from agent_core.artifacts import ArtifactStorePort
from agent_core.config import RuntimeBudget
from agent_core.errors import ActionError, ProviderError
from agent_core.events import AgentEvent, EventSinkPort, NullEventSink
from agent_core.harness import AgentHarness, CancelToken, RunState
from agent_core.loop_guard import LoopGuard, LoopGuardConfig
from agent_core.memory import MemoryPort, MemoryQuery, NullMemory
from agent_core.policy import AllowAllPolicy, ApprovalRequest, PolicyPort
from agent_core.prompt import PromptIR
from agent_core.providers import LLMMessage, LLMProviderPort, LLMRequest, LLMResponse, UsageInfo
from agent_core.skills import SkillsContext
from agent_core.timeline import TimelineStore
from agent_core.tools import (
    NullToolReplay,
    ToolInvocation,
    ToolRegistry,
    ToolReplayPort,
    ToolResult,
    ToolRuntimePort,
    tool_manifest_item,
)


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
        approval_store: ApprovalStorePort | None = None,
        memory: MemoryPort | None = None,
        skills: SkillsContext | None = None,
        timeline: TimelineStore | None = None,
        tool_replay: ToolReplayPort | None = None,
        action_verifier: ActionVerifierPort | None = None,
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
        self.approval_store = approval_store or NullApprovalStore()
        self.memory = memory or NullMemory()
        self.skills = skills
        self.timeline = timeline
        self.tool_replay = tool_replay or NullToolReplay()
        self.action_verifier = action_verifier or NullActionVerifier()
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
        messages = [LLMMessage(role="user", content=prompt.render())]
        memory_message = await self._memory_message(task)
        if memory_message is not None:
            messages.append(memory_message)
        result_output = ""
        final_action: ParsedAction | None = None

        for index in range(self.config.max_iterations):
            if self.cancel_token.cancelled:
                result_output = self.cancel_token.reason or "run cancelled"
                await self.harness.finish_run(run, "cancelled", {"output": result_output})
                await self._emit("run_cancelled", run, payload={"reason": result_output})
                return ReActResult(
                    run_id=run.run_id,
                    status="cancelled",
                    output=result_output,
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

            response = await self._call_provider(
                LLMRequest(
                    messages=list(messages),
                    model=self.config.model,
                    metadata=_provider_request_metadata(self.config.budget),
                ),
                run=run,
                turn_id=turn.turn_id,
            )
            if self.cancel_token.cancelled:
                result_output = self.cancel_token.reason or "run cancelled"
                await self.harness.checkpoint(
                    turn,
                    {"status": "cancelled", "reason": result_output, "iteration": index},
                )
                await self.harness.finish_run(run, "cancelled", {"output": result_output})
                await self._emit(
                    "run_cancelled",
                    run,
                    turn_id=turn.turn_id,
                    payload={"reason": result_output},
                )
                return ReActResult(
                    run_id=run.run_id,
                    status="cancelled",
                    output=result_output,
                    iterations=index + 1,
                )
            await self.harness.record_model_event(
                turn,
                {
                    "content": response.content,
                    "action": response.action,
                    "finish_reason": response.finish_reason,
                    "usage": response.usage.__dict__,
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
            if not decision.allowed:
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
                await self.harness.checkpoint(
                    turn,
                    {"status": "finished", "output": result_output, "iteration": index},
                )
                await self.harness.finish_run(run, "completed", {"output": result_output})
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

            tool_result = await self._execute_action(run, turn.turn_id, action)
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

    async def _call_provider(
        self,
        request: LLMRequest,
        *,
        run: RunState,
        turn_id: str,
    ) -> LLMResponse:
        if not self.config.stream:
            return await self.provider.complete(request)

        content_parts: list[str] = []
        action: dict[str, Any] | None = None
        usage = UsageInfo()
        finish_reason = ""
        metadata: dict[str, Any] = {"streamed": True}
        async for event in self.provider.stream(request):
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
            if event.delta:
                content_parts.append(event.delta)
            if event.action is not None:
                action = event.action
            if event.usage is not None:
                usage = event.usage
            if event.type == "error":
                finish_reason = "error"
                metadata["error"] = event.error
            elif event.type == "message_end":
                finish_reason = "stop"
        return LLMResponse(
            content="".join(content_parts),
            action=action,
            usage=usage,
            finish_reason=finish_reason,
            metadata=metadata,
        )

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
        if not decision.allowed:
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
        result = await self.tool_runtime.invoke(invocation)
        if result.ok:
            await self.tool_replay.put(invocation, result)
        await self._emit(
            "tool_finished",
            run,
            turn_id=turn_id,
            payload={
                "call_id": result.call_id,
                "tool_name": result.tool_name,
                "status": result.status,
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

