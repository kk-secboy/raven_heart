"""Context and timeline reduction primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_core.timeline import TimelineItem, TimelineStore, TimelineView


@dataclass(frozen=True)
class ReducerRequest:
    items: tuple[TimelineItem, ...]
    max_bytes: int = 64 * 1024
    recent_keep_ratio: float = 0.25
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "ReducerRequest":
        return ReducerRequest(
            items=tuple(self.items),
            max_bytes=max(1, int(self.max_bytes)),
            recent_keep_ratio=min(1.0, max(0.0, float(self.recent_keep_ratio))),
            metadata=dict(self.metadata),
        )

    def manifest(self) -> dict[str, Any]:
        active = tuple(item for item in self.items if not item.deleted)
        return {
            "schema_version": "agent-core-reducer-request/v1",
            "item_count": len(self.items),
            "active_item_count": len(active),
            "active_bytes": sum(item.bytes for item in active),
            "max_bytes": self.max_bytes,
            "recent_keep_ratio": self.recent_keep_ratio,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReducerResult:
    compressed_head: str
    retained_items: tuple[TimelineItem, ...]
    archive_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-reducer-result/v1",
            "compressed_bytes": len(self.compressed_head.encode("utf-8")),
            "retained_item_count": len(self.retained_items),
            "retained_item_ids": [item.item_id for item in self.retained_items],
            "retained_bytes": sum(item.bytes for item in self.retained_items if not item.deleted),
            "archive_refs": list(self.archive_refs),
            "metadata": dict(self.metadata),
        }


class ContextReducerPort(Protocol):
    async def reduce(self, request: ReducerRequest) -> ReducerResult:
        """Reduce timeline/context items when they exceed budget."""


class DefaultContextReducer:
    """Deterministic fallback reducer for tests and emergency trimming."""

    async def reduce(self, request: ReducerRequest) -> ReducerResult:
        request = request.normalized()
        items = tuple(item for item in request.items if not item.deleted)
        if not items:
            return ReducerResult(
                compressed_head="",
                retained_items=(),
                metadata=_result_metadata(request, (), ()),
            )
        total_bytes = sum(item.bytes for item in items)
        if total_bytes <= request.max_bytes:
            return ReducerResult(
                compressed_head="",
                retained_items=items,
                metadata=_result_metadata(request, items, ()),
            )

        keep_bytes = max(1, int(request.max_bytes * request.recent_keep_ratio))
        retained_ids = {item.item_id for item in items if item.pinned}
        used = 0
        retained_recent = False
        for item in reversed(items):
            if item.item_id in retained_ids:
                continue
            if used + item.bytes <= keep_bytes or not retained_recent:
                retained_ids.add(item.item_id)
                used += item.bytes
                retained_recent = True
            else:
                break
        retained = tuple(item for item in items if item.item_id in retained_ids)
        compressed = [item for item in items if item.item_id not in retained_ids]
        lines = [
            f"- {item.kind}:{item.item_id}: {item.content[:240]}"
            for item in compressed
            if item.content
        ]
        archive_refs = tuple(f"timeline:{item.item_id}" for item in compressed)
        return ReducerResult(
            compressed_head="\n".join(lines),
            retained_items=retained,
            archive_refs=archive_refs,
            metadata=_result_metadata(request, retained, tuple(compressed)),
        )


def apply_reduction_to_timeline(timeline: TimelineStore, result: ReducerResult) -> TimelineView:
    """Apply a reducer result to a timeline without coupling to a storage backend."""

    retained_ids = {item.item_id for item in result.retained_items}
    if result.compressed_head:
        timeline.compressed_head = "\n".join(
            item for item in (timeline.compressed_head, result.compressed_head) if item
        )
    for ref in result.archive_refs:
        if ref not in timeline.archive_refs:
            timeline.archive_refs.append(ref)
    if result.archive_refs or result.compressed_head:
        for item in timeline.items:
            if item.deleted or item.item_id in retained_ids:
                continue
            timeline.soft_delete(item.item_id)
    return TimelineView(
        open_items=tuple(item for item in timeline.items if not item.deleted),
        compressed_head=timeline.compressed_head,
        archive_refs=tuple(timeline.archive_refs),
    )


def _result_metadata(
    request: ReducerRequest,
    retained: tuple[TimelineItem, ...],
    compressed: tuple[TimelineItem, ...],
) -> dict[str, Any]:
    active = tuple(item for item in request.items if not item.deleted)
    return {
        "schema_version": "agent-core-reducer-metadata/v1",
        "max_bytes": request.max_bytes,
        "recent_keep_ratio": request.recent_keep_ratio,
        "original_item_count": len(active),
        "original_bytes": sum(item.bytes for item in active),
        "retained_item_count": len(retained),
        "retained_bytes": sum(item.bytes for item in retained if not item.deleted),
        "compressed_item_count": len(compressed),
        "compressed_bytes": sum(item.bytes for item in compressed if not item.deleted),
    }

