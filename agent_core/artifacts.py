"""Artifact storage primitives for large agent observations."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from agent_core.backends import storage_backend_manifest


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    uri: str
    content: bytes
    content_type: str = "text/plain; charset=utf-8"
    sha256: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return len(self.content)

    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def manifest(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "uri": self.uri,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "metadata": self.metadata,
        }


class ArtifactStorePort(Protocol):
    async def put_text(
        self,
        content: str,
        *,
        content_type: str = "text/plain; charset=utf-8",
        metadata: dict[str, object] | None = None,
    ) -> ArtifactRecord:
        """Store text content and return an artifact record."""

    async def get(self, artifact_id: str) -> ArtifactRecord:
        """Load one artifact by id."""


class InMemoryArtifactStore:
    def __init__(self) -> None:
        self.records: dict[str, ArtifactRecord] = {}

    async def put_text(
        self,
        content: str,
        *,
        content_type: str = "text/plain; charset=utf-8",
        metadata: dict[str, object] | None = None,
    ) -> ArtifactRecord:
        raw = str(content).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        artifact_id = uuid4().hex
        record = ArtifactRecord(
            artifact_id=artifact_id,
            uri=f"artifact://{artifact_id}",
            content=raw,
            content_type=content_type,
            sha256=digest,
            metadata=dict(metadata or {}),
        )
        self.records[artifact_id] = record
        return record

    async def get(self, artifact_id: str) -> ArtifactRecord:
        return self.records[artifact_id]

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": "agent-core-artifact-store/v1",
            "backend": storage_backend_manifest(role="artifact", kind="in_memory"),
            "artifacts": [record.manifest() for record in self.records.values()],
        }


class SQLiteArtifactStore:
    """SQLite-backed artifact store for durable local SDK runs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    async def put_text(
        self,
        content: str,
        *,
        content_type: str = "text/plain; charset=utf-8",
        metadata: dict[str, object] | None = None,
    ) -> ArtifactRecord:
        raw = str(content).encode("utf-8")
        record = _artifact_record_from_content(
            raw,
            content_type=content_type,
            metadata=dict(metadata or {}),
        )
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO artifacts(
                    artifact_id, uri, content, content_type, sha256, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.artifact_id,
                    record.uri,
                    record.content,
                    record.content_type,
                    record.sha256,
                    json.dumps(record.metadata, ensure_ascii=False, sort_keys=True),
                ),
            )
            conn.commit()
        return record

    async def get(self, artifact_id: str) -> ArtifactRecord:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                """
                SELECT artifact_id, uri, content, content_type, sha256, metadata_json
                FROM artifacts
                WHERE artifact_id = ?
                """,
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        return _artifact_record_from_row(row)

    def records(self) -> tuple[ArtifactRecord, ...]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                SELECT artifact_id, uri, content, content_type, sha256, metadata_json
                FROM artifacts
                ORDER BY rowid ASC
                """
            ).fetchall()
        return tuple(_artifact_record_from_row(row) for row in rows)

    def manifest(self) -> dict[str, object]:
        records = self.records()
        return {
            "schema_version": "agent-core-sqlite-artifact-store/v1",
            "backend": storage_backend_manifest(
                role="artifact",
                kind="sqlite",
                location=str(self.path),
                capabilities=("put_text", "get", "records"),
            ),
            "path": str(self.path),
            "artifact_count": len(records),
            "artifacts": [record.manifest() for record in records],
        }

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    uri TEXT NOT NULL,
                    content BLOB NOT NULL,
                    content_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.commit()


class MarkdownArtifactStore:
    """Markdown-backed artifact store for inspectable local SDK runs."""

    _START = "<!-- artifact-record "
    _END = " -->"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def put_text(
        self,
        content: str,
        *,
        content_type: str = "text/plain; charset=utf-8",
        metadata: dict[str, object] | None = None,
    ) -> ArtifactRecord:
        raw = str(content).encode("utf-8")
        record = _artifact_record_from_content(
            raw,
            content_type=content_type,
            metadata=dict(metadata or {}),
        )
        records = {item.artifact_id: item for item in self.records()}
        records[record.artifact_id] = record
        self._write(tuple(records.values()))
        return record

    async def get(self, artifact_id: str) -> ArtifactRecord:
        for record in self.records():
            if record.artifact_id == artifact_id:
                return record
        raise KeyError(artifact_id)

    def records(self) -> tuple[ArtifactRecord, ...]:
        if not self.path.exists():
            return ()
        text = self.path.read_text(encoding="utf-8")
        records: list[ArtifactRecord] = []
        for match in _ARTIFACT_MARKDOWN_RE.finditer(text):
            try:
                records.append(_artifact_record_from_payload(_decode_artifact_payload(match.group(1))))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return tuple(records)

    def manifest(self) -> dict[str, object]:
        records = self.records()
        return {
            "schema_version": "agent-core-markdown-artifact-store/v1",
            "backend": storage_backend_manifest(
                role="artifact",
                kind="markdown",
                location=str(self.path),
                capabilities=("put_text", "get", "records"),
            ),
            "path": str(self.path),
            "artifact_count": len(records),
            "artifacts": [record.manifest() for record in records],
        }

    def _write(self, records: tuple[ArtifactRecord, ...]) -> None:
        lines = [
            "# Artifact Records",
            "",
            "This file is managed by raven_heart. Artifact payloads are stored in comments.",
            "",
        ]
        for record in records:
            raw = _encode_artifact_payload(record)
            lines.append(f"{self._START}{raw}{self._END}")
            lines.append(f"- artifact_id: `{record.artifact_id}`")
            lines.append(f"- content_type: `{record.content_type}`")
            lines.append(f"- size_bytes: `{record.size_bytes}`")
            lines.append(f"- sha256: `{record.sha256}`")
            lines.append("")
        self.path.write_text("\n".join(lines), encoding="utf-8")


_ARTIFACT_MARKDOWN_RE = re.compile(
    r"<!--\s*artifact-record\s+([A-Za-z0-9+/=]+)\s*-->",
    re.DOTALL,
)


def _artifact_record_from_content(
    content: bytes,
    *,
    content_type: str,
    metadata: dict[str, object],
) -> ArtifactRecord:
    digest = hashlib.sha256(content).hexdigest()
    artifact_id = uuid4().hex
    return ArtifactRecord(
        artifact_id=artifact_id,
        uri=f"artifact://{artifact_id}",
        content=content,
        content_type=content_type,
        sha256=digest,
        metadata=dict(metadata),
    )


def _artifact_record_from_row(row: Any) -> ArtifactRecord:
    artifact_id, uri, content, content_type, sha256, metadata_json = row
    try:
        metadata = json.loads(str(metadata_json or "{}"))
    except json.JSONDecodeError:
        metadata = {}
    return ArtifactRecord(
        artifact_id=str(artifact_id),
        uri=str(uri),
        content=bytes(content),
        content_type=str(content_type),
        sha256=str(sha256),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _artifact_record_payload(record: ArtifactRecord) -> dict[str, object]:
    return {
        "schema_version": "agent-core-artifact-record-payload/v1",
        "artifact_id": record.artifact_id,
        "uri": record.uri,
        "content_base64": base64.b64encode(record.content).decode("ascii"),
        "content_type": record.content_type,
        "sha256": record.sha256,
        "metadata": dict(record.metadata),
    }


def _artifact_record_from_payload(raw: str) -> ArtifactRecord:
    payload = json.loads(raw or "{}")
    if not isinstance(payload, dict):
        raise ValueError("invalid artifact payload")
    content = base64.b64decode(str(payload.get("content_base64") or "").encode("ascii"))
    return ArtifactRecord(
        artifact_id=str(payload.get("artifact_id") or uuid4().hex),
        uri=str(payload.get("uri") or ""),
        content=content,
        content_type=str(payload.get("content_type") or "text/plain; charset=utf-8"),
        sha256=str(payload.get("sha256") or hashlib.sha256(content).hexdigest()),
        metadata=dict(payload.get("metadata") or {}),
    )


def _encode_artifact_payload(record: ArtifactRecord) -> str:
    raw = json.dumps(
        _artifact_record_payload(record),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _decode_artifact_payload(encoded: str) -> str:
    return base64.b64decode(encoded.encode("ascii")).decode("utf-8")

