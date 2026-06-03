from __future__ import annotations

import pytest

from agent_core.events import AgentEvent, ListEventSink, MarkdownEventSink, SQLiteEventSink


@pytest.mark.asyncio
async def test_event_sinks_persist_manifests_and_run_filters(tmp_path) -> None:
    sinks = (
        ListEventSink(),
        SQLiteEventSink(tmp_path / "events.sqlite"),
        MarkdownEventSink(tmp_path / "events.md"),
    )

    for sink in sinks:
        await sink.emit(AgentEvent(type="run_started", run_id="run-1", payload={"task": "inspect"}))
        await sink.emit(AgentEvent(type="tool_finished", run_id="run-1", turn_id="turn-1", payload={"ok": True}))
        await sink.emit(AgentEvent(type="run_finished", run_id="run-2"))

        records = sink.records()
        run_records = sink.records(run_id="run-1")
        manifest = sink.manifest()

        assert [event.sequence for event in records] == [1, 2, 3]
        assert [event.type for event in run_records] == ["run_started", "tool_finished"]
        assert manifest["event_count"] == 3
        assert manifest["events"][1]["payload"]["ok"] is True


@pytest.mark.asyncio
async def test_durable_event_sinks_reload_existing_records(tmp_path) -> None:
    sqlite_path = tmp_path / "events.sqlite"
    markdown_path = tmp_path / "events.md"

    for sink in (SQLiteEventSink(sqlite_path), MarkdownEventSink(markdown_path)):
        await sink.emit(AgentEvent(type="run_started", run_id="run-1"))
        await sink.emit(AgentEvent(type="error", run_id="run-1", payload={"message": "failed"}))

    sqlite_reloaded = SQLiteEventSink(sqlite_path)
    markdown_reloaded = MarkdownEventSink(markdown_path)

    assert sqlite_reloaded.manifest()["schema_version"] == "agent-core-sqlite-event-sink/v1"
    assert markdown_reloaded.manifest()["schema_version"] == "agent-core-markdown-event-sink/v1"
    assert sqlite_reloaded.records()[1].payload["message"] == "failed"
    assert markdown_reloaded.records()[1].payload["message"] == "failed"
