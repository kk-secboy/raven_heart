"""Short-term working memory and timeline windowing."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
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

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-in-memory-timeline-store/v1",
            "item_count": len(self._items),
            "active_item_count": len([item for item in self._items if not item.deleted]),
            "total_bytes": self.total_bytes(),
            "compressed_head_bytes": len(self.compressed_head.encode("utf-8")),
            "archive_ref_count": len(self.archive_refs),
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


def _timeline_manifest_for_storage(store: TimelineStore) -> dict[str, Any]:
    return {
        "schema_version": "agent-core-timeline-store-snapshot/v1",
        "items": [_timeline_item_to_dict(item) for item in store.items],
        "compressed_head": store.compressed_head,
        "archive_refs": list(store.archive_refs),
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

