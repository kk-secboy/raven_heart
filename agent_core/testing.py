"""Test doubles for the provider-neutral agent core."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, AsyncIterator

from agent_core.harness import AgentHarness, Checkpoint, ResumeToken, RunState, TurnState
from agent_core.memory import MemoryHit, MemoryQuery, MemoryWrite
from agent_core.providers import LLMRequest, LLMResponse, LLMStreamEvent
from agent_core.tools import InMemoryToolReplay, ToolInvocation, ToolResult, ToolSpec


class MockLLMProvider:
    def __init__(self, responses: list[LLMResponse | dict[str, Any] | str]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.responses:
            return LLMResponse(action={"action": "finish", "arguments": {"output": ""}})
        item = self.responses.pop(0)
        if isinstance(item, LLMResponse):
            return item
        if isinstance(item, dict):
            return LLMResponse(action=item)
        return LLMResponse(content=item)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        response = await self.complete(request)
        if response.content:
            yield LLMStreamEvent(type="delta", delta=response.content)
        if response.action:
            yield LLMStreamEvent(type="action", action=response.action)
        yield LLMStreamEvent(type="message_end")


class MockToolRuntime:
    def __init__(self, results: dict[str, str] | None = None) -> None:
        self.results = dict(results or {})
        self.invocations: list[ToolInvocation] = []

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(ToolSpec(name=name) for name in self.results)

    async def invoke(self, invocation: ToolInvocation) -> ToolResult:
        self.invocations.append(invocation)
        if invocation.tool_name not in self.results:
            return ToolResult(
                call_id=invocation.call_id,
                tool_name=invocation.tool_name,
                status="failed",
                error=f"unknown tool: {invocation.tool_name}",
            )
        return ToolResult(
            call_id=invocation.call_id,
            tool_name=invocation.tool_name,
            status="completed",
            content=self.results[invocation.tool_name],
        )


class InMemoryHarness(AgentHarness):
    def __init__(self) -> None:
        self.runs: dict[str, RunState] = {}
        self.turns: list[TurnState] = []
        self.prompts: list[dict[str, Any]] = []
        self.model_events: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.checkpoints: list[Checkpoint] = []
        self.finished: list[dict[str, Any]] = []

    async def start_run(self, task: str, metadata: dict[str, Any] | None = None) -> RunState:
        run = RunState(task=task, metadata=dict(metadata or {}))
        self.runs[run.run_id] = run
        return run

    async def start_turn(self, run: RunState, index: int) -> TurnState:
        turn = TurnState(run_id=run.run_id, index=index)
        self.turns.append(turn)
        return turn

    async def record_prompt(self, turn: TurnState, manifest: dict[str, Any]) -> None:
        self.prompts.append({"turn_id": turn.turn_id, "manifest": manifest})

    async def record_model_event(self, turn: TurnState, event: dict[str, Any]) -> None:
        self.model_events.append({"turn_id": turn.turn_id, **event})

    async def record_tool_call(self, turn: TurnState, event: dict[str, Any]) -> None:
        self.tool_calls.append({"turn_id": turn.turn_id, **event})

    async def checkpoint(self, turn: TurnState, state: dict[str, Any]) -> Checkpoint:
        checkpoint = Checkpoint(
            run_id=turn.run_id,
            turn_id=turn.turn_id,
            sequence=len(self.checkpoints) + 1,
            state=dict(state),
        )
        self.checkpoints.append(checkpoint)
        return checkpoint

    async def resume(self, token: ResumeToken) -> Checkpoint:
        for checkpoint in self.checkpoints:
            if checkpoint.run_id == token.run_id and checkpoint.checkpoint_id == token.checkpoint_id:
                return checkpoint
        raise KeyError(token.checkpoint_id)

    async def finish_run(self, run: RunState, status: str, result: dict[str, Any]) -> None:
        self.runs[run.run_id] = replace(run, status=status)
        self.finished.append({"run_id": run.run_id, "status": status, "result": result})


class MockMemory:
    def __init__(self, hits: list[MemoryHit] | None = None) -> None:
        self.hits = tuple(hits or ())
        self.queries: list[MemoryQuery] = []
        self.writes: list[MemoryWrite] = []

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        self.queries.append(query)
        return self.hits

    async def write(self, item: MemoryWrite) -> None:
        self.writes.append(item)

