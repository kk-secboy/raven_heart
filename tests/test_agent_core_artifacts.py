from __future__ import annotations

import pytest

from agent_core import ArtifactTrace
from agent_core.artifacts import InMemoryArtifactStore, MarkdownArtifactStore, SQLiteArtifactStore


@pytest.mark.asyncio
async def test_sqlite_artifact_store_persists_text_records_across_instances(tmp_path) -> None:
    path = tmp_path / "artifacts.sqlite"
    first = SQLiteArtifactStore(path)

    record = await first.put_text(
        "large observation",
        metadata={"tool_name": "dump", "call_id": "call-1"},
    )
    second = SQLiteArtifactStore(path)
    restored = await second.get(record.artifact_id)
    manifest = second.manifest()

    assert restored.text() == "large observation"
    assert restored.sha256 == record.sha256
    assert restored.metadata["tool_name"] == "dump"
    assert manifest["schema_version"] == "agent-core-sqlite-artifact-store/v1"
    assert manifest["artifact_count"] == 1
    assert manifest["artifacts"][0]["sha256"] == record.sha256


@pytest.mark.asyncio
async def test_markdown_artifact_store_persists_text_records_without_visible_payload(tmp_path) -> None:
    path = tmp_path / "artifacts.md"
    first = MarkdownArtifactStore(path)

    record = await first.put_text(
        "payload with comment marker --> kept intact",
        metadata={"source": "test"},
    )
    second = MarkdownArtifactStore(path)
    restored = await second.get(record.artifact_id)
    manifest = second.manifest()
    text = path.read_text(encoding="utf-8")

    assert restored.text() == "payload with comment marker --> kept intact"
    assert restored.metadata["source"] == "test"
    assert "payload with comment marker --> kept intact" not in text
    assert manifest["schema_version"] == "agent-core-markdown-artifact-store/v1"
    assert manifest["artifact_count"] == 1


@pytest.mark.asyncio
async def test_in_memory_artifact_store_manifest_contains_prompt_safe_records() -> None:
    store = InMemoryArtifactStore()

    record = await store.put_text("secret body", metadata={"kind": "tool_result"})
    manifest = store.manifest()

    assert await store.get(record.artifact_id) == record
    assert manifest["artifacts"][0]["artifact_id"] == record.artifact_id
    assert manifest["artifacts"][0]["sha256"] == record.sha256
    assert "secret body" not in str(manifest)


@pytest.mark.asyncio
async def test_artifact_trace_summarizes_prompt_safe_records() -> None:
    store = InMemoryArtifactStore()
    record = await store.put_text(
        "large tool result",
        metadata={"kind": "tool_result", "tool_name": "dump", "call_id": "call-1"},
    )

    trace = ArtifactTrace.from_session({"artifact_store": store.manifest()}).manifest()

    assert trace["schema_version"] == "agent-core-artifact-trace/v1"
    assert trace["artifact_count"] == 1
    assert trace["total_bytes"] == record.size_bytes
    assert trace["max_artifact_bytes"] == record.size_bytes
    assert trace["kinds"] == {"tool_result": 1}
    assert trace["tool_names"] == {"dump": 1}
    assert trace["artifacts"][0]["sha256"] == record.sha256
    assert "large tool result" not in str(trace)
