"""Short-term working memory and timeline windowing."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from agent_core.backends import storage_backend_manifest
from agent_core.prompt import estimate_tokens


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class TimelineBudget:
    max_bytes: int = 64 * 1024
    recent_keep_ratio: float = 0.25
    interval_minutes: int = 3
    prompt_block_bytes: int = 64 * 1024


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
class _TimelinePromptBlock:
    kind: str
    nonce: str
    content: str
    open: bool = False
    item_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self, tag_name: str = "TIMELINE") -> str:
        tag = _normalize_timeline_tag(tag_name)
        body = self.content.rstrip()
        if not body:
            return ""
        return f"<|{tag}_{self.nonce}|>\n{body}\n<|{tag}_END_{self.nonce}|>"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-timeline-prompt-block/v1",
            "kind": self.kind,
            "nonce": self.nonce,
            "open": self.open,
            "item_ids": list(self.item_ids),
            "bytes": len(self.content.encode("utf-8")),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TimelineArchiveRef:
    """Structured archive pointer kept out of the raw fact stream."""

    archive_id: str
    reason: str = "batch_compress"
    summary_preview: str = ""
    reducer_key_id: str = ""
    source_start_id: str = ""
    source_end_id: str = ""
    item_count: int = 0
    created_at: str = field(default_factory=utc_now_iso)

    def render(self) -> str:
        parts = [
            f"archive_id={self.archive_id}",
            f"reason={self.reason}",
            f"range={self.source_start_id}-{self.source_end_id}",
            f"items={self.item_count}",
        ]
        if self.summary_preview.strip():
            parts.append(f"summary={self.summary_preview.strip()}")
        return " ".join(parts)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-timeline-archive-ref/v1",
            "archive_id": self.archive_id,
            "reason": self.reason,
            "summary_preview": self.summary_preview,
            "reducer_key_id": self.reducer_key_id,
            "source_start_id": self.source_start_id,
            "source_end_id": self.source_end_id,
            "item_count": self.item_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> "TimelineArchiveRef":
        return cls(
            archive_id=str(data.get("archive_id") or ""),
            reason=str(data.get("reason") or "batch_compress"),
            summary_preview=str(data.get("summary_preview") or ""),
            reducer_key_id=str(data.get("reducer_key_id") or ""),
            source_start_id=str(data.get("source_start_id") or ""),
            source_end_id=str(data.get("source_end_id") or ""),
            item_count=int(data.get("item_count") or 0),
            created_at=str(data.get("created_at") or utc_now_iso()),
        )


@dataclass(frozen=True)
class TimelineCursor:
    """Stable cursor over the original timeline fact stream."""

    last_item_id: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-timeline-cursor/v1",
            "last_item_id": self.last_item_id,
        }


@dataclass(frozen=True)
class TimelineDiff:
    """Diff over original timeline items; independent from prompt view reduction."""

    cursor: TimelineCursor
    next_cursor: TimelineCursor
    items: tuple[TimelineItem, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.items)

    def render(self) -> str:
        return "\n\n".join(item.render() for item in self.items if not item.deleted)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-timeline-diff/v1",
            "cursor": self.cursor.manifest(),
            "next_cursor": self.next_cursor.manifest(),
            "item_count": len(self.items),
            "item_ids": [item.item_id for item in self.items],
            "kinds": [item.kind for item in self.items],
            "bytes": sum(item.bytes for item in self.items),
        }


@dataclass(frozen=True)
class TimelineView:
    frozen_items: tuple[TimelineItem, ...] = ()
    open_items: tuple[TimelineItem, ...] = ()
    compressed_head: str = ""
    archive_refs: tuple[str, ...] = ()
    frozen_blocks: tuple[_TimelinePromptBlock, ...] = ()
    open_blocks: tuple[_TimelinePromptBlock, ...] = ()
    archive_ref_records: tuple[TimelineArchiveRef, ...] = ()

    def render_frozen(self) -> str:
        if self.frozen_blocks:
            parts = [_render_prompt_blocks(self.frozen_blocks)]
            if self.archive_ref_records:
                parts.append(
                    "[archive_refs]\n"
                    + "\n".join(ref.render() for ref in self.archive_ref_records)
                )
            elif self.archive_refs:
                parts.append("[archive_refs]\n" + "\n".join(self.archive_refs))
            return "\n\n".join(part for part in parts if part)
        parts = []
        if self.compressed_head:
            parts.append(f"[compressed_head]\n{self.compressed_head}")
        parts.extend(item.render() for item in self.frozen_items if not item.deleted)
        if self.archive_refs:
            parts.append("[archive_refs]\n" + "\n".join(self.archive_refs))
        return "\n\n".join(parts)

    def render_open(self) -> str:
        if self.open_blocks:
            return _render_prompt_blocks(self.open_blocks)
        return "\n\n".join(item.render() for item in self.open_items if not item.deleted)


class TimelineStore:
    def __init__(self, items: list[TimelineItem] | None = None) -> None:
        self._items: list[TimelineItem] = list(items or [])
        self.compressed_head = ""
        self.archive_refs: list[str] = []
        self.archive_ref_records: list[TimelineArchiveRef] = []

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

    def cursor(self) -> TimelineCursor:
        for item in reversed(self._items):
            if not item.deleted:
                return TimelineCursor(last_item_id=item.item_id)
        return TimelineCursor()

    def diff_since(self, cursor: TimelineCursor | None = None) -> TimelineDiff:
        cursor = cursor or TimelineCursor()
        active = tuple(item for item in self._items if not item.deleted)
        if not cursor.last_item_id:
            items = active
        else:
            start = -1
            for index, item in enumerate(active):
                if item.item_id == cursor.last_item_id:
                    start = index
                    break
            items = active[start + 1 :] if start >= 0 else active
        next_cursor = (
            TimelineCursor(last_item_id=active[-1].item_id) if active else TimelineCursor()
        )
        return TimelineDiff(cursor=cursor, next_cursor=next_cursor, items=tuple(items))

    def view(self, budget: TimelineBudget | None = None) -> TimelineView:
        budget = budget or TimelineBudget()
        active = [item for item in self._items if not item.deleted]
        if not active:
            return TimelineView(
                compressed_head=self.compressed_head,
                archive_refs=tuple(self.archive_refs),
                archive_ref_records=tuple(self.archive_ref_records),
            )
        total = sum(item.bytes for item in active)
        if total <= budget.max_bytes:
            frozen_blocks, open_blocks = self._prompt_blocks(active, budget)
            return TimelineView(
                open_items=tuple(active),
                compressed_head=self.compressed_head,
                archive_refs=tuple(self.archive_refs),
                frozen_blocks=frozen_blocks,
                open_blocks=open_blocks,
                archive_ref_records=tuple(self.archive_ref_records),
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
        frozen_blocks, open_blocks = self._prompt_blocks(active, budget)
        return TimelineView(
            frozen_items=tuple(frozen_items),
            open_items=tuple(reversed(open_items)),
            compressed_head=self.compressed_head,
            archive_refs=tuple(self.archive_refs),
            frozen_blocks=frozen_blocks,
            open_blocks=open_blocks,
            archive_ref_records=tuple(self.archive_ref_records),
        )

    def _prompt_blocks(
        self,
        active: list[TimelineItem],
        budget: TimelineBudget,
    ) -> tuple[tuple[_TimelinePromptBlock, ...], tuple[_TimelinePromptBlock, ...]]:
        blocks: list[_TimelinePromptBlock] = []
        if self.compressed_head.strip():
            blocks.append(_reducer_prompt_block(self.compressed_head, active))
        blocks.extend(_interval_prompt_blocks(active, budget))
        frozen = tuple(block for block in blocks if not block.open)
        open_tail = tuple(block for block in blocks if block.open)
        return frozen, open_tail

    def add_archive_ref(self, ref: TimelineArchiveRef | str) -> None:
        if isinstance(ref, TimelineArchiveRef):
            if ref.archive_id and ref.archive_id not in self.archive_refs:
                self.archive_refs.append(ref.archive_id)
            if not any(
                existing.archive_id == ref.archive_id for existing in self.archive_ref_records
            ):
                self.archive_ref_records.append(ref)
            return
        value = str(ref)
        if value and value not in self.archive_refs:
            self.archive_refs.append(value)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-in-memory-timeline-store/v1",
            "item_count": len(self._items),
            "active_item_count": len([item for item in self._items if not item.deleted]),
            "total_bytes": self.total_bytes(),
            "compressed_head_bytes": len(self.compressed_head.encode("utf-8")),
            "archive_ref_count": len(self.archive_refs),
            "structured_archive_ref_count": len(self.archive_ref_records),
            "cursor": self.cursor().manifest(),
            "backend": storage_backend_manifest(
                role="timeline",
                kind="in_memory",
                capabilities=("append", "window", "soft_delete"),
            ),
        }


class TimelineStorePort(Protocol):
    @property
    def items(self) -> tuple[TimelineItem, ...]:
        """Return all known timeline items."""

    def add(self, content: str, *, kind: str = "observation", **metadata: Any) -> TimelineItem:
        """Append one timeline item."""

    def soft_delete(self, item_id: str) -> bool:
        """Mark one item deleted without losing auditability."""

    def truncate_after(self, item_id: str) -> int:
        """Soft-delete active items after the selected item."""

    def group_by_bytes(self, max_bucket_bytes: int) -> tuple[tuple[TimelineItem, ...], ...]:
        """Group active timeline items by byte budget."""

    def total_bytes(self) -> int:
        """Return active timeline bytes."""

    def cursor(self) -> TimelineCursor:
        """Return a cursor for the current original timeline fact stream."""

    def diff_since(self, cursor: TimelineCursor | None = None) -> TimelineDiff:
        """Return original timeline items added after the cursor."""

    def view(self, budget: TimelineBudget | None = None) -> TimelineView:
        """Return the prompt-facing timeline window."""

    def manifest(self) -> dict[str, Any]:
        """Return prompt-safe store metadata."""


class SQLiteTimelineStore(TimelineStore):
    """SQLite-backed timeline store for durable local SDK runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        super().__init__(_load_sqlite_items(self.path))
        state = _load_sqlite_state(self.path)
        self.compressed_head = str(state.get("compressed_head") or "")
        self.archive_refs = [str(value) for value in state.get("archive_refs") or ()]
        self.archive_ref_records = _archive_refs_from_manifest(state.get("archive_ref_records"))

    def add(self, content: str, *, kind: str = "observation", **metadata: Any) -> TimelineItem:
        item = super().add(content, kind=kind, **metadata)
        _upsert_sqlite_item(self.path, item)
        return item

    def soft_delete(self, item_id: str) -> bool:
        changed = super().soft_delete(item_id)
        if changed:
            for item in self._items:
                if item.item_id == item_id:
                    _upsert_sqlite_item(self.path, item)
                    break
        return changed

    def truncate_after(self, item_id: str) -> int:
        deleted = super().truncate_after(item_id)
        if deleted:
            for item in self._items:
                _upsert_sqlite_item(self.path, item)
        return deleted

    def save_state(self) -> None:
        _save_sqlite_state(
            self.path,
            {
                "compressed_head": self.compressed_head,
                "archive_refs": list(self.archive_refs),
                "archive_ref_records": [ref.manifest() for ref in self.archive_ref_records],
            },
        )

    def manifest(self) -> dict[str, Any]:
        base = super().manifest()
        return {
            **base,
            "schema_version": "agent-core-sqlite-timeline-store/v1",
            "path": str(self.path),
            "backend": storage_backend_manifest(
                role="timeline",
                kind="sqlite",
                durable=True,
                inspectable=True,
                queryable=True,
                capabilities=("append", "window", "soft_delete", "state"),
            ),
        }

    def _init_db(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_timeline_items (
                    item_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    pinned INTEGER NOT NULL,
                    deleted INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_timeline_state (
                    id TEXT PRIMARY KEY,
                    manifest_json TEXT NOT NULL
                )
                """
            )


class MarkdownTimelineStore(TimelineStore):
    """Markdown-backed timeline store for inspectable local SDK runs."""

    _START = "<!-- agent-timeline-store"
    _END = "agent-timeline-store -->"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        manifest = _load_markdown_timeline(self.path)
        super().__init__(_items_from_manifest(manifest))
        self.compressed_head = str(manifest.get("compressed_head") or "")
        self.archive_refs = [str(value) for value in manifest.get("archive_refs") or ()]
        self.archive_ref_records = _archive_refs_from_manifest(
            manifest.get("archive_ref_records")
        )

    def add(self, content: str, *, kind: str = "observation", **metadata: Any) -> TimelineItem:
        item = super().add(content, kind=kind, **metadata)
        self.save()
        return item

    def soft_delete(self, item_id: str) -> bool:
        changed = super().soft_delete(item_id)
        if changed:
            self.save()
        return changed

    def truncate_after(self, item_id: str) -> int:
        deleted = super().truncate_after(item_id)
        if deleted:
            self.save()
        return deleted

    def save(self) -> None:
        manifest = _timeline_manifest_for_storage(self)
        self.path.write_text(
            "\n".join(
                (
                    "# agent timeline",
                    "",
                    f"{self._START}",
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                    f"{self._END}",
                    "",
                )
            ),
            encoding="utf-8",
        )

    def manifest(self) -> dict[str, Any]:
        base = super().manifest()
        return {
            **base,
            "schema_version": "agent-core-markdown-timeline-store/v1",
            "path": str(self.path),
            "backend": storage_backend_manifest(
                role="timeline",
                kind="markdown",
                durable=True,
                inspectable=True,
                queryable=True,
                capabilities=("append", "window", "soft_delete", "state"),
            ),
        }


def _timeline_item_to_dict(item: TimelineItem) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "kind": item.kind,
        "content": item.content,
        "created_at": item.created_at,
        "metadata": dict(item.metadata),
        "pinned": item.pinned,
        "deleted": item.deleted,
    }


def _timeline_item_from_dict(data: dict[str, Any]) -> TimelineItem:
    return TimelineItem(
        item_id=str(data.get("item_id") or uuid4().hex),
        kind=str(data.get("kind") or "observation"),
        content=str(data.get("content") or ""),
        created_at=str(data.get("created_at") or utc_now_iso()),
        metadata=dict(data.get("metadata") or {}),
        pinned=bool(data.get("pinned")),
        deleted=bool(data.get("deleted")),
    )


def _archive_refs_from_manifest(value: Any) -> list[TimelineArchiveRef]:
    if not isinstance(value, list | tuple):
        return []
    refs: list[TimelineArchiveRef] = []
    for item in value:
        if isinstance(item, dict):
            ref = TimelineArchiveRef.from_manifest(dict(item))
            if ref.archive_id:
                refs.append(ref)
    return refs


def _interval_prompt_blocks(
    active: list[TimelineItem],
    budget: TimelineBudget,
) -> tuple[_TimelinePromptBlock, ...]:
    if not active:
        return ()
    interval_minutes = max(1, int(budget.interval_minutes or 3))
    bucket_bytes = int(budget.prompt_block_bytes or 0)
    ordered = sorted(
        enumerate(active),
        key=lambda pair: (_item_datetime(pair[1]), pair[0]),
    )
    grouped: dict[datetime, list[tuple[int, TimelineItem]]] = {}
    for original_index, item in ordered:
        bucket_start = _align_to_bucket(_item_datetime(item), interval_minutes)
        grouped.setdefault(bucket_start, []).append((original_index, item))

    blocks: list[_TimelinePromptBlock] = []
    for bucket_start in sorted(grouped):
        bucket_end = bucket_start + timedelta(minutes=interval_minutes)
        sub_buckets = _split_bucket_by_bytes(
            bucket_start,
            bucket_end,
            interval_minutes,
            grouped[bucket_start],
            bucket_bytes,
        )
        total = len(sub_buckets)
        for seq, sub_items in enumerate(sub_buckets):
            items = tuple(item for _, item in sub_items)
            nonce = _interval_nonce(bucket_start, interval_minutes, seq, total)
            content = _render_interval_block(bucket_start, bucket_end, interval_minutes, items)
            blocks.append(
                _TimelinePromptBlock(
                    kind="interval",
                    nonce=nonce,
                    content=content,
                    item_ids=tuple(item.item_id for item in items),
                    metadata={
                        "bucket_start": bucket_start.isoformat(),
                        "bucket_end": bucket_end.isoformat(),
                        "interval_minutes": interval_minutes,
                        "seq_in_bucket": seq,
                        "total_in_bucket": total,
                    },
                )
            )
    if blocks:
        blocks[-1] = _TimelinePromptBlock(
            kind=blocks[-1].kind,
            nonce=blocks[-1].nonce,
            content=blocks[-1].content,
            open=True,
            item_ids=blocks[-1].item_ids,
            metadata=dict(blocks[-1].metadata),
        )
    return tuple(blocks)


def _reducer_prompt_block(
    compressed_head: str,
    active: list[TimelineItem],
) -> _TimelinePromptBlock:
    ts = _item_datetime(active[0]) if active else datetime.fromtimestamp(0, UTC)
    reducer_key = active[0].item_id[:12] if active else "0"
    nonce = f"r{_nonce_safe(reducer_key)}t{int(ts.timestamp())}"
    content = "\n".join(
        (
            f"# reducer key={reducer_key} ts={int(ts.timestamp())}",
            f"{ts.strftime('%H:%M:%S')} [reducer/memory]",
            "[compressed_head]",
            _collapse_blank_lines(compressed_head),
        )
    ).strip()
    return _TimelinePromptBlock(
        kind="reducer",
        nonce=nonce,
        content=content,
        open=False,
        item_ids=(),
        metadata={"reducer_key": reducer_key, "ts": ts.isoformat()},
    )


def _split_bucket_by_bytes(
    bucket_start: datetime,
    bucket_end: datetime,
    interval_minutes: int,
    items: list[tuple[int, TimelineItem]],
    bucket_bytes: int,
) -> list[list[tuple[int, TimelineItem]]]:
    if bucket_bytes < 0:
        return [list(items)]
    limit = bucket_bytes or 64 * 1024
    header_bytes = len(_interval_header(bucket_start, bucket_end, interval_minutes).encode("utf-8"))
    out: list[list[tuple[int, TimelineItem]]] = []
    current: list[tuple[int, TimelineItem]] = []
    used = 0
    for indexed in items:
        item = indexed[1]
        item_bytes = _interval_item_bytes(item)
        next_bytes = item_bytes if not current else item_bytes + 1
        if current and used + next_bytes > limit:
            out.append(current)
            current = []
            used = 0
        if not current:
            used = header_bytes
        current.append(indexed)
        used += item_bytes if len(current) == 1 else next_bytes
    if current:
        out.append(current)
    return out


def _interval_item_bytes(item: TimelineItem) -> int:
    text = "\n".join(
        (
            f"{_item_datetime(item).strftime('%H:%M:%S')} [{_timeline_item_verbose(item)}]",
            _collapse_blank_lines(item.content),
        )
    ).strip()
    return len(text.encode("utf-8"))


def _render_interval_block(
    bucket_start: datetime,
    bucket_end: datetime,
    interval_minutes: int,
    items: tuple[TimelineItem, ...],
) -> str:
    parts = [_interval_header(bucket_start, bucket_end, interval_minutes)]
    for item in items:
        entry = "\n".join(
            (
                f"{_item_datetime(item).strftime('%H:%M:%S')} [{_timeline_item_verbose(item)}]",
                _collapse_blank_lines(item.content),
            )
        ).strip()
        if entry:
            parts.append(entry)
    return "\n".join(parts).rstrip()


def _interval_header(
    bucket_start: datetime,
    bucket_end: datetime,
    interval_minutes: int,
) -> str:
    return (
        f"# bucket={bucket_start.strftime('%Y/%m/%d %H:%M:%S')}-"
        f"{bucket_end.strftime('%H:%M:%S')} interval={interval_minutes}m"
    )


def _interval_nonce(
    bucket_start: datetime,
    interval_minutes: int,
    seq: int,
    total: int,
) -> str:
    base = f"b{interval_minutes}t{int(bucket_start.timestamp())}"
    if total > 1:
        return f"{base}s{seq}"
    return base


def _item_datetime(item: TimelineItem) -> datetime:
    raw = item.created_at.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        value = datetime.fromisoformat(raw)
    except ValueError:
        return datetime.fromtimestamp(0, UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _align_to_bucket(value: datetime, minutes: int) -> datetime:
    aligned_minute = (value.minute // minutes) * minutes
    return value.replace(minute=aligned_minute, second=0, microsecond=0)


def _timeline_item_verbose(item: TimelineItem) -> str:
    if item.kind == "tool":
        tool_name = str(item.metadata.get("tool_name") or item.metadata.get("name") or "result")
        status = str(item.metadata.get("status") or "ok")
        return f"tool/{tool_name} {status}"
    if item.kind in {"user", "user_interaction"}:
        return "user/input"
    if item.kind == "fact":
        return "text/fact"
    return f"text/{item.kind or 'raw'}"


def _collapse_blank_lines(text: str) -> str:
    lines: list[str] = []
    previous_blank = False
    for raw_line in str(text).strip().splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if previous_blank:
                continue
            previous_blank = True
            lines.append("")
            continue
        previous_blank = False
        lines.append(line)
    return "\n".join(lines).strip()


def _render_prompt_blocks(blocks: tuple[_TimelinePromptBlock, ...]) -> str:
    return "\n".join(block.render() for block in blocks if block.render()).strip()


def _normalize_timeline_tag(raw: str) -> str:
    cleaned = "".join(ch for ch in str(raw).strip() if ch.isalnum() or ch == "_")
    return cleaned or "TIMELINE"


def _nonce_safe(raw: str) -> str:
    cleaned = "".join(ch for ch in str(raw) if ch.isalnum())
    return cleaned or "0"


def _timeline_manifest_for_storage(store: TimelineStore) -> dict[str, Any]:
    return {
        "schema_version": "agent-core-timeline-store-snapshot/v1",
        "items": [_timeline_item_to_dict(item) for item in store.items],
        "compressed_head": store.compressed_head,
        "archive_refs": list(store.archive_refs),
        "archive_ref_records": [ref.manifest() for ref in store.archive_ref_records],
    }


def _items_from_manifest(manifest: dict[str, Any]) -> list[TimelineItem]:
    return [
        _timeline_item_from_dict(dict(item))
        for item in manifest.get("items", ())
        if isinstance(item, dict)
    ]


def _load_sqlite_items(path: Path) -> list[TimelineItem]:
    with sqlite3.connect(path) as conn:
        rows = conn.execute(
            """
            SELECT item_id, kind, content, created_at, metadata_json, pinned, deleted
            FROM agent_timeline_items
            ORDER BY rowid ASC
            """
        ).fetchall()
    return [
        TimelineItem(
            item_id=str(row[0]),
            kind=str(row[1]),
            content=str(row[2]),
            created_at=str(row[3]),
            metadata=json.loads(str(row[4] or "{}")),
            pinned=bool(row[5]),
            deleted=bool(row[6]),
        )
        for row in rows
    ]


def _upsert_sqlite_item(path: Path, item: TimelineItem) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO agent_timeline_items(
                item_id, kind, content, created_at, metadata_json, pinned, deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                kind=excluded.kind,
                content=excluded.content,
                created_at=excluded.created_at,
                metadata_json=excluded.metadata_json,
                pinned=excluded.pinned,
                deleted=excluded.deleted
            """,
            (
                item.item_id,
                item.kind,
                item.content,
                item.created_at,
                json.dumps(item.metadata, ensure_ascii=False, sort_keys=True),
                1 if item.pinned else 0,
                1 if item.deleted else 0,
            ),
        )


def _load_sqlite_state(path: Path) -> dict[str, Any]:
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT manifest_json FROM agent_timeline_state WHERE id = 'state'"
        ).fetchone()
    if row is None:
        return {}
    try:
        value = json.loads(str(row[0]))
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _save_sqlite_state(path: Path, state: dict[str, Any]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO agent_timeline_state(id, manifest_json)
            VALUES ('state', ?)
            ON CONFLICT(id) DO UPDATE SET manifest_json=excluded.manifest_json
            """,
            (json.dumps(dict(state), ensure_ascii=False, sort_keys=True),),
        )


def _load_markdown_timeline(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    start = text.find(MarkdownTimelineStore._START)
    end = text.find(MarkdownTimelineStore._END)
    if start == -1 or end == -1 or end <= start:
        return {}
    payload = text[start + len(MarkdownTimelineStore._START) : end].strip()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, dict) else {}

