"""Short-term working memory and timeline windowing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from agent_core.prompt import estimate_tokens


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class TimelineBudget:
    max_bytes: int = 64 * 1024
    recent_keep_ratio: float = 0.25


@dataclass(frozen=True)
class TimelineItem:
    content: str
    kind: str = "observation"
    item_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)
    pinned: bool = False
    deleted: bool = False

    @property
    def bytes(self) -> int:
        return len(self.content.encode("utf-8"))

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.content)

    def render(self) -> str:
        return f"[{self.kind}:{self.item_id}] {self.content}".strip()


@dataclass(frozen=True)
class TimelineView:
    frozen_items: tuple[TimelineItem, ...] = ()
    open_items: tuple[TimelineItem, ...] = ()
    compressed_head: str = ""
    archive_refs: tuple[str, ...] = ()

    def render_frozen(self) -> str:
        parts = []
        if self.compressed_head:
            parts.append(f"[compressed_head]\n{self.compressed_head}")
        parts.extend(item.render() for item in self.frozen_items if not item.deleted)
        if self.archive_refs:
            parts.append("[archive_refs]\n" + "\n".join(self.archive_refs))
        return "\n\n".join(parts)

    def render_open(self) -> str:
        return "\n\n".join(item.render() for item in self.open_items if not item.deleted)


class TimelineStore:
    def __init__(self, items: list[TimelineItem] | None = None) -> None:
        self._items: list[TimelineItem] = list(items or [])
        self.compressed_head = ""
        self.archive_refs: list[str] = []

    @property
    def items(self) -> tuple[TimelineItem, ...]:
        return tuple(self._items)

    def add(self, content: str, *, kind: str = "observation", **metadata: Any) -> TimelineItem:
        item = TimelineItem(content=str(content), kind=kind, metadata=dict(metadata))
        self._items.append(item)
        return item

    def soft_delete(self, item_id: str) -> bool:
        for index, item in enumerate(self._items):
            if item.item_id == item_id:
                self._items[index] = TimelineItem(
                    content=item.content,
                    kind=item.kind,
                    item_id=item.item_id,
                    created_at=item.created_at,
                    metadata=dict(item.metadata),
                    pinned=item.pinned,
                    deleted=True,
                )
                return True
        return False

    def truncate_after(self, item_id: str) -> int:
        seen = False
        deleted = 0
        for item in list(self._items):
            if seen and not item.deleted and self.soft_delete(item.item_id):
                deleted += 1
            if item.item_id == item_id:
                seen = True
        return deleted

    def group_by_bytes(self, max_bucket_bytes: int) -> tuple[tuple[TimelineItem, ...], ...]:
        buckets: list[list[TimelineItem]] = []
        current: list[TimelineItem] = []
        used = 0
        limit = max(1, max_bucket_bytes)
        for item in self._items:
            if item.deleted:
                continue
            item_bytes = item.bytes
            if current and used + item_bytes > limit:
                buckets.append(current)
                current = []
                used = 0
            current.append(item)
            used += item_bytes
        if current:
            buckets.append(current)
        return tuple(tuple(bucket) for bucket in buckets)

    def total_bytes(self) -> int:
        return sum(item.bytes for item in self._items if not item.deleted)

    def view(self, budget: TimelineBudget | None = None) -> TimelineView:
        budget = budget or TimelineBudget()
        active = [item for item in self._items if not item.deleted]
        if not active:
            return TimelineView(
                compressed_head=self.compressed_head,
                archive_refs=tuple(self.archive_refs),
            )
        total = sum(item.bytes for item in active)
        if total <= budget.max_bytes:
            return TimelineView(
                open_items=tuple(active),
                compressed_head=self.compressed_head,
                archive_refs=tuple(self.archive_refs),
            )
        keep_bytes = max(1, int(budget.max_bytes * budget.recent_keep_ratio))
        open_items: list[TimelineItem] = []
        used = 0
        for item in reversed(active):
            if item.pinned or used + item.bytes <= keep_bytes or not open_items:
                open_items.append(item)
                used += item.bytes
            else:
                break
        open_ids = {item.item_id for item in open_items}
        frozen_items = [item for item in active if item.item_id not in open_ids]
        return TimelineView(
            frozen_items=tuple(frozen_items),
            open_items=tuple(reversed(open_items)),
            compressed_head=self.compressed_head,
            archive_refs=tuple(self.archive_refs),
        )

