"""Standard event stream for agent core execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol


EventType = Literal[
    "run_started",
    "turn_started",
    "prompt_ready",
    "model_response",
    "model_stream",
    "action_parsed",
    "tool_started",
    "tool_finished",
    "timeline_updated",
    "checkpoint_created",
    "error",
    "run_finished",
    "run_cancelled",
]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class AgentEvent:
    type: EventType
    run_id: str = ""
    turn_id: str = ""
    sequence: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)


class EventSinkPort(Protocol):
    async def emit(self, event: AgentEvent) -> None:
        """Emit an agent event."""


class NullEventSink:
    async def emit(self, event: AgentEvent) -> None:
        return None


class ListEventSink:
    def __init__(self) -> None:
        self.events: list[AgentEvent] = []

    async def emit(self, event: AgentEvent) -> None:
        self.events.append(event)

