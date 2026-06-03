"""Composable runner entrypoint for the provider-neutral agent core."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from agent_core.actions import ActionRegistry, ActionVerifierPort
from agent_core.approvals import ApprovalResumeContext, ApprovalStorePort, NullApprovalStore
from agent_core.artifacts import ArtifactStorePort
from agent_core.capabilities import CapabilityCatalog
from agent_core.config import AgentProfile, RuntimeBudget
from agent_core.context import AgentContextPack, AgentPromptBuilder, ContextInjection
from agent_core.events import EventSinkPort
from agent_core.harness import AgentHarness, CancelToken, InMemoryAgentJournal, ResumeToken
from agent_core.loop_guard import LoopGuard
from agent_core.memory import MemoryHit, MemoryPort, MemoryQuery, NullMemory
from agent_core.mcp import MCPCenter
from agent_core.policy import PolicyPort
from agent_core.providers import LLMProviderPort
from agent_core.prompt import PromptBucketRole
from agent_core.react import ReActConfig, ReActExecutor, ReActResult
from agent_core.reducer import ContextReducerPort, ReducerRequest, apply_reduction_to_timeline
from agent_core.skills import SkillsContext
from agent_core.timeline import TimelineBudget, TimelineStore
from agent_core.tools import NullToolReplay, ToolReplayPort, ToolRuntimePort
from agent_core.trace import AgentJournalReplay, AgentRunTraceBundle, NullRunTraceStore, RunTraceStorePort


@dataclass
class AgentSession:
    """All core-level components needed to run one agent profile."""

    profile: AgentProfile
    provider: LLMProviderPort
    tools: ToolRuntimePort
    actions: ActionRegistry = field(default_factory=ActionRegistry)
    harness: AgentHarness = field(default_factory=InMemoryAgentJournal)
    skills: SkillsContext | None = None
    mcp: MCPCenter | None = None
    memory: MemoryPort = field(default_factory=NullMemory)
    timeline: TimelineStore = field(default_factory=TimelineStore)
    context_reducer: ContextReducerPort | None = None
    event_sink: EventSinkPort | None = None
    policy: PolicyPort | None = None
    approval_store: ApprovalStorePort = field(default_factory=NullApprovalStore)
    tool_replay: ToolReplayPort = field(default_factory=NullToolReplay)
    trace_store: RunTraceStorePort = field(default_factory=NullRunTraceStore)
    action_verifier: ActionVerifierPort | None = None
    loop_guard: LoopGuard | None = None
    artifact_store: ArtifactStorePort | None = None
    cancel_token: CancelToken = field(default_factory=CancelToken)
    metadata: dict[str, Any] = field(default_factory=dict)

    def reset_cancel_token(self) -> None:
        self.cancel_token = CancelToken()

    def capability_catalog(self) -> CapabilityCatalog:
        return CapabilityCatalog(
            actions=self.actions,
            tools=self.tools,
            skills=self.skills,
            mcp=self.mcp,
            metadata={"profile": self.profile.name, **self.metadata},
        )

    def manifest(self) -> dict[str, Any]:
        provider_manifest = getattr(self.provider, "manifest", None)
        memory_manifest = getattr(self.memory, "manifest", None)
        approval_manifest = getattr(self.approval_store, "manifest", None)
        trace_store_manifest = getattr(self.trace_store, "manifest", None)
        return {
            "schema_version": "agent-core-session/v1",
            "profile": {
                "name": self.profile.name,
                "model": self.profile.model,
                "instructions": self.profile.instructions,
                "budget": _budget_manifest(self.profile.budget),
                "capabilities": {
                    "actions": list(self.profile.capabilities.actions),
                    "tools": list(self.profile.capabilities.tools),
                    "skills": list(self.profile.capabilities.skills),
                    "memory_enabled": self.profile.capabilities.memory_enabled,
                    "human_in_loop_enabled": self.profile.capabilities.human_in_loop_enabled,
                },
            },
            "provider": provider_manifest() if callable(provider_manifest) else {},
            "capabilities": self.capability_catalog().manifest(),
            "memory": memory_manifest() if callable(memory_manifest) else {},
            "approvals": approval_manifest() if callable(approval_manifest) else {},
            "trace_store": trace_store_manifest() if callable(trace_store_manifest) else {},
            "context_reducer": _context_reducer_manifest(self.context_reducer),
            "timeline_items": len(self.timeline.items),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentRunRequest:
    task: str
    context: AgentContextPack | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    refresh: bool = False
    resume_token: ResumeToken | None = None
    approval_resume: ApprovalResumeContext | None = None


@dataclass(frozen=True)
class AgentRunOutcome:
    result: ReActResult
    session_manifest: dict[str, Any]
    prompt_manifest: dict[str, Any]
    resume_manifest: dict[str, Any] = field(default_factory=dict)
    timeline_reduction_manifest: dict[str, Any] = field(default_factory=dict)
    trace_manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ManagedAgentRun:
    run_key: str
    session_name: str
    task: str
    status: str = "queued"
    result_run_id: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "run_key": self.run_key,
            "session_name": self.session_name,
            "task": self.task,
            "status": self.status,
            "result_run_id": self.result_run_id,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class AgentRunner:
    """SDK-like runner that wires core centers into ReAct execution."""

    def __init__(self, session: AgentSession) -> None:
        self.session = session

    async def refresh(self) -> None:
        refresh_tools = getattr(self.session.tools, "refresh", None)
        if callable(refresh_tools):
            await refresh_tools()
        if self.session.mcp is not None:
            await self.session.mcp.refresh(fail_fast=False)
            try:
                await self.session.mcp.refresh_resources()
            except Exception as exc:
                self.session.metadata["mcp_resource_refresh_error"] = str(exc)
            try:
                await self.session.mcp.refresh_prompts()
            except Exception as exc:
                self.session.metadata["mcp_prompt_refresh_error"] = str(exc)

    async def run(self, request: AgentRunRequest | str) -> AgentRunOutcome:
        run_request = request if isinstance(request, AgentRunRequest) else AgentRunRequest(task=str(request))
        if run_request.refresh:
            await self.refresh()
        resume_manifest = await self._resume_manifest(run_request.resume_token)
        memory_injections = await self._memory_injections(run_request)
        timeline_reduction_manifest = await self._reduce_timeline_if_needed(run_request)
        context = self._context_for(
            run_request,
            resume_manifest=resume_manifest,
            injections=memory_injections,
            timeline_reduction_manifest=timeline_reduction_manifest,
        )
        prompt = self._prompt_builder().build(context).trim_to_budget(
            self.session.profile.budget.max_prompt_bytes
        )
        executor = self._executor(run_request.approval_resume)
        result = await executor.run(run_request.task, prompt)
        session_manifest = self.session.manifest()
        trace_manifest = await self._trace_manifest(
            result,
            session_manifest=session_manifest,
            prompt_manifest=prompt.manifest(),
            resume_manifest=resume_manifest,
            timeline_reduction_manifest=timeline_reduction_manifest,
            request=run_request,
        )
        await self.session.trace_store.save(trace_manifest)
        return AgentRunOutcome(
            result=result,
            session_manifest=session_manifest,
            prompt_manifest=prompt.manifest(),
            resume_manifest=resume_manifest,
            timeline_reduction_manifest=timeline_reduction_manifest,
            trace_manifest=trace_manifest,
        )

    def _prompt_builder(self) -> AgentPromptBuilder:
        return AgentPromptBuilder(
            tools=self.session.tools,
            skills=self.session.skills,
            timeline=self.session.timeline,
            timeline_budget=self._timeline_budget(),
            capabilities=self.session.capability_catalog(),
        )

    def _timeline_budget(self) -> TimelineBudget:
        return TimelineBudget(max_bytes=self.session.profile.budget.max_timeline_bytes)

    async def _resume_manifest(self, token: ResumeToken | None) -> dict[str, Any]:
        if token is None:
            return {}
        checkpoint = await self.session.harness.resume(token)
        return {
            "schema_version": "agent-core-resume/v1",
            "run_id": checkpoint.run_id,
            "turn_id": checkpoint.turn_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "sequence": checkpoint.sequence,
            "created_at": checkpoint.created_at,
            "state": dict(checkpoint.state),
            "token_metadata": dict(token.metadata),
        }

    def _context_for(
        self,
        request: AgentRunRequest,
        *,
        resume_manifest: dict[str, Any] | None = None,
        injections: tuple[ContextInjection, ...] = (),
        timeline_reduction_manifest: dict[str, Any] | None = None,
    ) -> AgentContextPack:
        base = request.context or AgentContextPack()
        system = base.system or self.session.profile.instructions
        dynamic_task = base.dynamic_task or request.task
        resume_manifest = resume_manifest or {}
        timeline_reduction_manifest = timeline_reduction_manifest or {}
        context_injections = (*base.injections, *injections)
        if resume_manifest:
            context_injections = (
                *context_injections,
                ContextInjection(
                    name="resume_checkpoint",
                    content=_resume_context_block(resume_manifest),
                    target=PromptBucketRole.TIMELINE_OPEN,
                    source="harness",
                    priority=100,
                    metadata={
                        "run_id": resume_manifest.get("run_id"),
                        "checkpoint_id": resume_manifest.get("checkpoint_id"),
                        "sequence": resume_manifest.get("sequence"),
                    },
                ),
            )
        approval_resume_manifest = _approval_resume_manifest(request.approval_resume)
        if approval_resume_manifest:
            context_injections = (
                *context_injections,
                ContextInjection(
                    name="approval_resume",
                    content=_approval_resume_context_block(approval_resume_manifest),
                    target=PromptBucketRole.TIMELINE_OPEN,
                    source="approval",
                    priority=95,
                    metadata={
                        "grant_count": approval_resume_manifest.get("grant_count"),
                        "approved_count": approval_resume_manifest.get("approved_count"),
                    },
                ),
            )
        metadata = {
            **base.metadata,
            **request.metadata,
            "profile": self.session.profile.name,
        }
        if resume_manifest:
            metadata["resume"] = resume_manifest
        if approval_resume_manifest:
            metadata["approval_resume"] = approval_resume_manifest
        if timeline_reduction_manifest:
            metadata["timeline_reduction"] = timeline_reduction_manifest
        return AgentContextPack(
            system=system,
            task_instruction=base.task_instruction,
            schema=base.schema,
            output_example=base.output_example,
            recent_tools_cache=base.recent_tools_cache,
            user_history=base.user_history,
            workspace=base.workspace,
            current_time=base.current_time,
            dynamic_task=dynamic_task,
            injections=context_injections,
            metadata=metadata,
        )

    async def _reduce_timeline_if_needed(self, request: AgentRunRequest) -> dict[str, Any]:
        reducer = self.session.context_reducer
        if reducer is None:
            return {}
        budget = self._timeline_budget()
        total_bytes = self.session.timeline.total_bytes()
        if total_bytes <= budget.max_bytes:
            return {}
        reducer_request = ReducerRequest(
            items=self.session.timeline.items,
            max_bytes=budget.max_bytes,
            recent_keep_ratio=budget.recent_keep_ratio,
            metadata={
                "profile": self.session.profile.name,
                "request_metadata": dict(request.metadata),
            },
        )
        result = await reducer.reduce(reducer_request)
        view = apply_reduction_to_timeline(self.session.timeline, result)
        manifest = result.manifest()
        manifest["request"] = reducer_request.manifest()
        manifest["view"] = {
            "open_item_count": len(view.open_items),
            "compressed_head_bytes": len(view.compressed_head.encode("utf-8")),
            "archive_refs": list(view.archive_refs),
        }
        return manifest

    async def _memory_injections(self, request: AgentRunRequest) -> tuple[ContextInjection, ...]:
        if not self.session.profile.capabilities.memory_enabled:
            return ()
        context = request.context or AgentContextPack()
        query_text = context.dynamic_task or request.task
        hits = await self.session.memory.search(MemoryQuery(query=query_text, limit=5))
        if not hits:
            return ()
        return (
            ContextInjection(
                name="memory_recall",
                content=_memory_context_block(hits),
                target=PromptBucketRole.SEMI_DYNAMIC_1,
                source="memory",
                priority=80,
                metadata={
                    "query": query_text,
                    "hit_count": len(hits),
                    "sources": [hit.source for hit in hits if hit.source],
                },
            ),
        )

    def _executor(self, approval_resume: ApprovalResumeContext | None = None) -> ReActExecutor:
        budget = self.session.profile.budget
        return ReActExecutor(
            provider=self.session.provider,
            tool_runtime=self.session.tools,
            action_registry=self.session.actions,
            harness=self.session.harness,
            event_sink=self.session.event_sink,
            policy=self.session.policy,
            approval_store=self.session.approval_store,
            approval_resume=approval_resume,
            memory=NullMemory(),
            skills=self.session.skills,
            timeline=self.session.timeline,
            tool_replay=self.session.tool_replay,
            action_verifier=self.session.action_verifier,
            loop_guard=self.session.loop_guard,
            artifact_store=self.session.artifact_store,
            cancel_token=self.session.cancel_token,
            config=ReActConfig(
                model=self.session.profile.model,
                max_iterations=budget.max_iterations,
                budget=budget,
            ),
        )

    async def _trace_manifest(
        self,
        result: ReActResult,
        *,
        session_manifest: dict[str, Any],
        prompt_manifest: dict[str, Any],
        resume_manifest: dict[str, Any],
        timeline_reduction_manifest: dict[str, Any],
        request: AgentRunRequest,
    ) -> dict[str, Any]:
        bundle = AgentRunTraceBundle(
            run_id=result.run_id,
            status=result.status,
            iterations=result.iterations,
            output_bytes=len(result.output.encode("utf-8")),
            session=session_manifest,
            prompt=prompt_manifest,
            journal_replay=self._journal_replay_manifest(result.run_id),
            provider=await _component_manifest(self.session.provider),
            tool_replay=await _component_manifest(self.session.tool_replay),
            approvals=await _component_manifest(self.session.approval_store),
            event_log=await _component_manifest(self.session.event_sink),
            resume=resume_manifest,
            timeline_reduction=timeline_reduction_manifest,
            metadata={
                "profile": self.session.profile.name,
                "request_metadata": dict(request.metadata),
            },
        )
        return bundle.manifest()

    def _journal_replay_manifest(self, run_id: str) -> dict[str, Any]:
        snapshot = getattr(self.session.harness, "snapshot", None)
        if not callable(snapshot):
            return {}
        return AgentJournalReplay.from_snapshot(snapshot(), run_id=run_id).manifest()


class AgentSessionManager:
    """Manage multiple agent sessions and background runs."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._runs: dict[str, ManagedAgentRun] = {}
        self._outcomes: dict[str, AgentRunOutcome] = {}
        self._tasks: dict[str, asyncio.Task[AgentRunOutcome]] = {}
        self._active_by_session: dict[str, str] = {}

    def register(self, session: AgentSession, *, name: str | None = None, replace: bool = False) -> str:
        session_name = (name or session.profile.name).strip()
        if not session_name:
            raise ValueError("session name is required")
        if session_name in self._sessions and not replace:
            raise ValueError(f"session already registered: {session_name}")
        self._sessions[session_name] = session
        return session_name

    def unregister(self, name: str) -> bool:
        if name in self._active_by_session:
            raise RuntimeError(f"session has active run: {name}")
        return self._sessions.pop(name, None) is not None

    def session(self, name: str) -> AgentSession:
        try:
            return self._sessions[name]
        except KeyError as exc:
            raise KeyError(f"unknown session: {name}") from exc

    def sessions(self) -> tuple[str, ...]:
        return tuple(sorted(self._sessions))

    async def run(self, session_name: str, request: AgentRunRequest | str) -> AgentRunOutcome:
        run_key = self._new_run_key(session_name)
        return await self._run_once(run_key, session_name, self._normalize_request(request))

    def start(self, session_name: str, request: AgentRunRequest | str) -> str:
        run_request = self._normalize_request(request)
        run_key = self._new_run_key(session_name)
        self._ensure_session_available(session_name)
        self._runs[run_key] = ManagedAgentRun(
            run_key=run_key,
            session_name=session_name,
            task=run_request.task,
            metadata=dict(run_request.metadata),
        )
        self._active_by_session[session_name] = run_key
        task = asyncio.create_task(self._run_once(run_key, session_name, run_request, preclaimed=True))
        self._tasks[run_key] = task
        return run_key

    def cancel(self, run_key: str, reason: str = "cancelled by manager") -> bool:
        run = self._runs.get(run_key)
        if run is None or run.status in {"completed", "cancelled", "failed"}:
            return False
        session = self._sessions.get(run.session_name)
        if session is not None:
            session.cancel_token.cancel(reason)
        self._runs[run_key] = _replace_run(run, status="cancelling", error=reason)
        return True

    async def wait(self, run_key: str) -> AgentRunOutcome:
        task = self._tasks.get(run_key)
        if task is None:
            outcome = self._outcomes.get(run_key)
            if outcome is None:
                raise KeyError(run_key)
            return outcome
        return await task

    def outcome(self, run_key: str) -> AgentRunOutcome | None:
        return self._outcomes.get(run_key)

    def run_state(self, run_key: str) -> ManagedAgentRun:
        try:
            return self._runs[run_key]
        except KeyError as exc:
            raise KeyError(f"unknown run: {run_key}") from exc

    def runs(self) -> tuple[ManagedAgentRun, ...]:
        return tuple(sorted(self._runs.values(), key=lambda item: item.run_key))

    def active_runs(self) -> tuple[ManagedAgentRun, ...]:
        return tuple(run for run in self.runs() if run.status in {"queued", "running", "cancelling"})

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-session-manager/v1",
            "sessions": {
                name: session.manifest()
                for name, session in sorted(self._sessions.items(), key=lambda item: item[0])
            },
            "runs": [run.manifest() for run in self.runs()],
            "active_by_session": dict(self._active_by_session),
        }

    def _new_run_key(self, session_name: str) -> str:
        if session_name not in self._sessions:
            raise KeyError(f"unknown session: {session_name}")
        return uuid4().hex

    async def _run_once(
        self,
        run_key: str,
        session_name: str,
        request: AgentRunRequest,
        *,
        preclaimed: bool = False,
    ) -> AgentRunOutcome:
        if not preclaimed:
            self._ensure_session_available(session_name)
            self._active_by_session[session_name] = run_key
            self._runs[run_key] = ManagedAgentRun(
                run_key=run_key,
                session_name=session_name,
                task=request.task,
                status="queued",
                metadata=dict(request.metadata),
            )
        session = self.session(session_name)
        current_run = self._runs[run_key]
        if session.cancel_token.cancelled and current_run.status != "cancelling":
            session.reset_cancel_token()
        self._runs[run_key] = _replace_run(self._runs[run_key], status="running")
        try:
            outcome = await AgentRunner(session).run(request)
        except Exception as exc:
            run = self._runs[run_key]
            self._runs[run_key] = _replace_run(run, status="failed", error=str(exc))
            record_error = getattr(session.harness, "record_error", None)
            if callable(record_error):
                await record_error(error=exc, metadata={"run_key": run_key, "session_name": session_name})
            raise
        finally:
            if self._active_by_session.get(session_name) == run_key:
                self._active_by_session.pop(session_name, None)
            self._tasks.pop(run_key, None)

        status = outcome.result.status
        if status == "cancelled":
            managed_status = "cancelled"
        elif status == "completed":
            managed_status = "completed"
        else:
            managed_status = status
        self._runs[run_key] = _replace_run(
            self._runs[run_key],
            status=managed_status,
            result_run_id=outcome.result.run_id,
        )
        self._outcomes[run_key] = outcome
        return outcome

    def _ensure_session_available(self, session_name: str) -> None:
        active = self._active_by_session.get(session_name)
        if active is not None:
            raise RuntimeError(f"session already has active run: {session_name}:{active}")

    @staticmethod
    def _normalize_request(request: AgentRunRequest | str) -> AgentRunRequest:
        return request if isinstance(request, AgentRunRequest) else AgentRunRequest(task=str(request))


def _budget_manifest(budget: RuntimeBudget) -> dict[str, Any]:
    return {
        "max_iterations": budget.max_iterations,
        "max_prompt_bytes": budget.max_prompt_bytes,
        "max_timeline_bytes": budget.max_timeline_bytes,
        "max_tool_result_bytes": budget.max_tool_result_bytes,
        "max_cost_usd": budget.max_cost_usd,
    }


def _context_reducer_manifest(reducer: ContextReducerPort | None) -> dict[str, Any]:
    if reducer is None:
        return {"enabled": False}
    reducer_manifest = getattr(reducer, "manifest", None)
    if callable(reducer_manifest):
        manifest = reducer_manifest()
        return {"enabled": True, **dict(manifest)}
    return {
        "enabled": True,
        "type": type(reducer).__name__,
    }


def _approval_resume_manifest(resume: ApprovalResumeContext | None) -> dict[str, Any]:
    if resume is None or resume.empty:
        return {}
    return resume.manifest()


async def _component_manifest(component: Any) -> dict[str, Any]:
    if component is None:
        return {}
    manifest = getattr(component, "manifest", None)
    if not callable(manifest):
        return {}
    value = manifest()
    if inspect.isawaitable(value):
        value = await value
    return dict(value) if isinstance(value, dict) else {}


def _resume_context_block(manifest: dict[str, Any]) -> str:
    state = manifest.get("state") if isinstance(manifest.get("state"), dict) else {}
    state_lines = []
    for key, value in sorted(state.items(), key=lambda item: str(item[0])):
        state_lines.append(f"- {key}: {value}")
    rendered_state = "\n".join(state_lines) if state_lines else "- empty"
    return (
        "== Resumed Checkpoint ==\n"
        f"run_id: {manifest.get('run_id')}\n"
        f"turn_id: {manifest.get('turn_id')}\n"
        f"checkpoint_id: {manifest.get('checkpoint_id')}\n"
        f"sequence: {manifest.get('sequence')}\n"
        "state:\n"
        f"{rendered_state}"
    )


def _approval_resume_context_block(manifest: dict[str, Any]) -> str:
    lines = [
        "== Approval Resume ==",
        f"grant_count: {manifest.get('grant_count')}",
        f"approved_count: {manifest.get('approved_count')}",
    ]
    for grant in manifest.get("grants") or ():
        if not isinstance(grant, dict):
            continue
        lines.append(
            "- "
            f"{grant.get('subject')} "
            f"status={grant.get('status')} "
            f"approval_id={grant.get('approval_id')} "
            f"actor={grant.get('actor')}"
        )
    return "\n".join(lines)


def _memory_context_block(hits: tuple[MemoryHit, ...]) -> str:
    lines = []
    for hit in hits:
        source = hit.source or "memory"
        lines.append(f"- {source} ({hit.score:.3f}): {hit.content}")
    return "[memory]\n" + "\n".join(lines)


def _replace_run(
    run: ManagedAgentRun,
    *,
    status: str | None = None,
    result_run_id: str | None = None,
    error: str | None = None,
) -> ManagedAgentRun:
    return ManagedAgentRun(
        run_key=run.run_key,
        session_name=run.session_name,
        task=run.task,
        status=status if status is not None else run.status,
        result_run_id=result_run_id if result_run_id is not None else run.result_run_id,
        error=error if error is not None else run.error,
        metadata=dict(run.metadata),
    )

