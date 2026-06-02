"""Context and timeline reduction primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_core.timeline import TimelineItem


@dataclass(frozen=True)
class ReducerRequest:
    items: tuple[TimelineItem, ...]
    max_bytes: int = 64 * 1024
    recent_keep_ratio: float = 0.25
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReducerResult:
    compressed_head: str
    retained_items: tuple[TimelineItem, ...]
    archive_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class ContextReducerPort(Protocol):
    async def reduce(self, request: ReducerRequest) -> ReducerResult:
        """Reduce timeline/context items when they exceed budget."""


class DefaultContextReducer:
    """Deterministic fallback reducer for tests and emergency trimming."""

    async def reduce(self, request: ReducerRequest) -> ReducerResult:
        items = tuple(item for item in request.items if not item.deleted)
        if not items:
            return ReducerResult(compressed_head="", retained_items=())
        total_bytes = sum(item.bytes for item in items)
        if total_bytes <= request.max_bytes:
            return ReducerResult(compressed_head="", retained_items=items)

        keep_bytes = max(1, int(request.max_bytes * request.recent_keep_ratio))
        retained: list[TimelineItem] = []
        used = 0
        for item in reversed(items):
            if item.pinned or used + item.bytes <= keep_bytes or not retained:
                retained.append(item)
                used += item.bytes
            else:
                break
        retained_ids = {item.item_id for item in retained}
        compressed = [item for item in items if item.item_id not in retained_ids]
        lines = [
            f"- {item.kind}:{item.item_id}: {item.content[:240]}"
            for item in compressed
            if item.content
        ]
        archive_refs = tuple(f"timeline:{item.item_id}" for item in compressed)
        return ReducerResult(
            compressed_head="\n".join(lines),
            retained_items=tuple(reversed(retained)),
            archive_refs=archive_refs,
            metadata={"compressed_item_count": len(compressed)},
        )

