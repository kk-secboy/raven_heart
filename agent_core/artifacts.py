"""Artifact storage primitives for large agent observations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4


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
            "artifacts": [record.manifest() for record in self.records.values()],
        }

