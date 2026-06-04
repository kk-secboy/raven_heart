from __future__ import annotations

import pytest

from agent_core.events import (
    AgentEvent,
    EventStreamBatch,
    EventStreamCursor,
    ListEventSink,
    MarkdownEventSink,
    SQLiteEventSink,
)


@pytest.mark.asyncio
async def test_event_sinks_persist_manifests_and_run_filters(tmp_path) -> None:
    sinks = (
        ListEventSink(),
        SQLiteEventSink(tmp_path / "events.sqlite"),
        MarkdownEventSink(tmp_path / "events.md"),
    )

    for sink in sinks:
        await sink.emit(AgentEvent(type="run_started", run_id="run-1", payload={"task": "inspect"}))
        await sink.emit(AgentEvent(type="policy_decision", run_id="run-1", turn_id="turn-1", payload={"decision_id": "d1"}))
        await sink.emit(AgentEvent(type="tool_finished", run_id="run-1", turn_id="turn-1", payload={"ok": True}))
        await sink.emit(AgentEvent(type="run_finished", run_id="run-2"))

        records = sink.records()
        run_records = sink.records(run_id="run-1")
        manifest = sink.manifest()

        assert [event.sequence for event in records] == [1, 2, 3, 4]
        assert [event.type for event in run_records] == ["run_started", "policy_decision", "tool_finished"]
        assert manifest["event_count"] == 4
        assert manifest["events"][1]["payload"]["decision_id"] == "d1"
        assert manifest["events"][2]["payload"]["ok"] is True


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


@pytest.mark.asyncio
async def test_event_stream_batch_pages_by_run_sequence_and_terminal_status() -> None:
    sink = ListEventSink()
    await sink.emit(AgentEvent(type="run_started", run_id="run-1"))
    await sink.emit(AgentEvent(type="turn_started", run_id="run-1", turn_id="turn-1"))
    await sink.emit(AgentEvent(type="tool_finished", run_id="run-1", turn_id="turn-1"))
    await sink.emit(AgentEvent(type="run_finished", run_id="run-1"))
    await sink.emit(AgentEvent(type="run_started", run_id="run-2"))

    first = EventStreamBatch.from_log(
        sink,
        EventStreamCursor(run_id="run-1", after_sequence=1, limit=2),
    )
    second = EventStreamBatch.from_log(sink, first.next_cursor())
    manifest = first.manifest()

    assert [event.sequence for event in first.events] == [2, 3]
    assert first.has_more is True
    assert first.terminal is False
    assert first.next_after_sequence == 3
    assert manifest["schema_version"] == "agent-core-event-stream-batch/v1"
    assert manifest["next_cursor"]["after_sequence"] == 3
    assert [event.type for event in second.events] == ["run_finished"]
    assert second.terminal is True
    assert second.has_more is False


@pytest.mark.asyncio
async def test_event_stream_cursor_filters_event_types() -> None:
    sink = ListEventSink()
    await sink.emit(AgentEvent(type="run_started", run_id="run-1"))
    await sink.emit(AgentEvent(type="model_stream", run_id="run-1", payload={"delta_bytes": 4}))
    await sink.emit(AgentEvent(type="tool_finished", run_id="run-1"))

    batch = EventStreamBatch.from_log(
        sink,
        EventStreamCursor(run_id="run-1", event_types=("model_stream",), limit=10),
    )

    assert [event.type for event in batch.events] == ["model_stream"]
    assert batch.manifest()["cursor"]["event_types"] == ["model_stream"]


@pytest.mark.asyncio
async def test_event_stream_cursor_filters_manager_run_metadata() -> None:
    sink = ListEventSink()
    await sink.emit(
        AgentEvent(
            type="run_started",
            run_id="inner-1",
            payload={"run_key": "run-a", "session_name": "alpha"},
        )
    )
    await sink.emit(
        AgentEvent(
            type="run_started",
            run_id="inner-2",
            payload={"run_key": "run-b", "session_name": "alpha"},
        )
    )
    await sink.emit(
        AgentEvent(
            type="run_finished",
            run_id="inner-3",
            payload={"run_key": "run-a", "session_name": "beta"},
        )
    )

    batch = EventStreamBatch.from_log(
        sink,
        EventStreamCursor(run_key="run-a", session_name="alpha"),
    )
    manifest = batch.manifest()

    assert [event.run_id for event in batch.events] == ["inner-1"]
    assert manifest["cursor"]["run_key"] == "run-a"
    assert manifest["cursor"]["session_name"] == "alpha"
    assert manifest["next_cursor"]["run_key"] == "run-a"
