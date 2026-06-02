from __future__ import annotations

import pytest

from agent_core.actions import ActionRegistry
from agent_core.errors import HarnessError, ResumeError
from agent_core.harness import (
    InMemoryAgentJournal,
    InMemoryJournalStore,
    MarkdownJournalStore,
    PersistentAgentJournal,
    SQLiteAgentJournal,
    SQLiteJournalStore,
)
from agent_core.prompt import PromptIR
from agent_core.react import ReActConfig, ReActExecutor
from agent_core.testing import MockLLMProvider, MockToolRuntime


@pytest.mark.asyncio
async def test_in_memory_agent_journal_checkpoints_resume_and_round_trips_snapshot() -> None:
    journal = InMemoryAgentJournal()
    run = await journal.start_run("task", metadata={"profile": "test"})
    turn = await journal.start_turn(run, 0)
    await journal.record_prompt(turn, {"prompt": "manifest"})
    await journal.record_model_event(turn, {"content": "model"})
    await journal.record_tool_call(turn, {"tool_name": "lookup", "status": "completed"})
    checkpoint = await journal.checkpoint(turn, {"status": "tool_finished", "value": 1})
    await journal.finish_run(run, "completed", {"output": "done"})

    token = journal.resume_token(checkpoint)
    resumed = await journal.resume(token)
    restored = InMemoryAgentJournal.from_manifest(journal.snapshot().manifest())
    restored_checkpoint = await restored.resume(token)

    assert resumed.state == {"status": "tool_finished", "value": 1}
    assert restored_checkpoint.state == resumed.state
    assert restored.runs[run.run_id].status == "completed"
    assert restored.prompts[0]["manifest"]["prompt"] == "manifest"
    assert restored.tool_calls[0]["tool_name"] == "lookup"
    assert restored.finished[0]["result"]["output"] == "done"
    assert restored.turns[0].status == "completed"


@pytest.mark.asyncio
async def test_persistent_agent_journal_uses_store_port_for_resume() -> None:
    store = InMemoryJournalStore()
    journal = PersistentAgentJournal(store)
    run = await journal.start_run("durable task", metadata={"profile": "store"})
    turn = await journal.start_turn(run, 0)
    checkpoint = await journal.checkpoint(turn, {"status": "tool_finished", "value": 2})
    token = journal.resume_token(checkpoint)
    await journal.finish_run(run, "completed", {"output": "done"})

    restored = PersistentAgentJournal(store)
    resumed = await restored.resume(token)
    manifest = restored.manifest()

    assert resumed.state == {"status": "tool_finished", "value": 2}
    assert restored.runs[run.run_id].status == "completed"
    assert restored.finished[0]["result"]["output"] == "done"
    assert manifest["schema_version"] == "agent-core-persistent-journal/v1"
    assert manifest["store"]["schema_version"] == "agent-core-in-memory-journal-store/v1"


@pytest.mark.asyncio
async def test_sqlite_journal_store_persists_checkpoint_resume_and_manifest(tmp_path) -> None:
    path = tmp_path / "journal.sqlite"
    journal = PersistentAgentJournal(SQLiteJournalStore(path))
    run = await journal.start_run("durable task", metadata={"profile": "sqlite"})
    turn = await journal.start_turn(run, 0)
    await journal.record_prompt(turn, {"bucket_hash": "abc"})
    await journal.record_model_event(turn, {"type": "message_end"})
    await journal.record_tool_call(turn, {"tool_name": "lookup", "status": "completed"})
    checkpoint = await journal.checkpoint(turn, {"status": "tool_finished", "value": 2})
    token = journal.resume_token(checkpoint)
    await journal.record_error(run=run, turn=turn, error=ValueError("durable error"))
    await journal.finish_run(run, "completed", {"output": "done"})

    restored = PersistentAgentJournal(SQLiteJournalStore(path))
    resumed = await restored.resume(token)
    manifest = restored.run_manifest(run.run_id)
    resumable = restored.resumable_runs()

    assert resumed.state == {"status": "tool_finished", "value": 2}
    assert restored.runs[run.run_id].status == "completed"
    assert manifest["run"]["metadata"]["profile"] == "sqlite"
    assert manifest["prompts"][0]["manifest"]["bucket_hash"] == "abc"
    assert manifest["tool_calls"][0]["tool_name"] == "lookup"
    assert manifest["errors"][0]["message"] == "durable error"
    assert manifest["finished"][0]["result"]["output"] == "done"
    assert resumable[0]["checkpoint_id"] == checkpoint.checkpoint_id
    assert restored.manifest()["schema_version"] == "agent-core-persistent-journal/v1"
    assert restored.manifest()["store"]["schema_version"] == "agent-core-sqlite-journal-store/v1"


@pytest.mark.asyncio
async def test_markdown_journal_store_persists_checkpoint_resume_and_manifest(tmp_path) -> None:
    path = tmp_path / "journal.md"
    journal = PersistentAgentJournal(MarkdownJournalStore(path))
    run = await journal.start_run("markdown task", metadata={"profile": "markdown"})
    turn = await journal.start_turn(run, 0)
    checkpoint = await journal.checkpoint(turn, {"status": "ready", "value": 3})
    token = journal.resume_token(checkpoint)
    await journal.finish_run(run, "completed", {"output": "done"})

    restored = PersistentAgentJournal(MarkdownJournalStore(path))
    resumed = await restored.resume(token)
    text = path.read_text(encoding="utf-8")

    assert resumed.state == {"status": "ready", "value": 3}
    assert restored.runs[run.run_id].metadata["profile"] == "markdown"
    assert restored.finished[0]["result"]["output"] == "done"
    assert "<!-- agent-journal-snapshot" in text
    assert restored.manifest()["store"]["schema_version"] == "agent-core-markdown-journal-store/v1"


@pytest.mark.asyncio
async def test_sqlite_agent_journal_is_convenience_wrapper(tmp_path) -> None:
    path = tmp_path / "journal.sqlite"
    journal = SQLiteAgentJournal(path)
    run = await journal.start_run("wrapper task")
    turn = await journal.start_turn(run, 0)
    checkpoint = await journal.checkpoint(turn, {"status": "ready"})

    restored = SQLiteAgentJournal(path)
    assert (await restored.resume(journal.resume_token(checkpoint))).state == {"status": "ready"}
    assert restored.path == path


@pytest.mark.asyncio
async def test_react_executor_records_replayable_journal_snapshot() -> None:
    provider = MockLLMProvider(
        [
            {"action": "call_tool", "arguments": {"tool_name": "lookup", "arguments": {"q": "x"}}},
            {"action": "finish", "arguments": {"output": "done"}},
        ]
    )
    journal = InMemoryAgentJournal()
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime({"lookup": "found"}),
        action_registry=ActionRegistry(),
        harness=journal,
        config=ReActConfig(max_iterations=3),
    )

    result = await executor.run("journal task", PromptIR.from_parts(dynamic="task"))
    latest = journal.latest_checkpoint(result.run_id)
    restored = InMemoryAgentJournal.from_manifest(journal.snapshot().manifest())
    restored_latest = restored.latest_checkpoint(result.run_id)

    assert result.status == "completed"
    assert latest is not None
    assert latest.state["status"] == "finished"
    assert restored_latest is not None
    assert restored_latest.state == latest.state
    assert len(restored.prompts) == 2
    assert len(restored.model_events) == 2
    assert len(restored.tool_calls) == 1
    assert restored.finished[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_journal_rejects_invalid_lifecycle_transitions() -> None:
    journal = InMemoryAgentJournal()
    run = await journal.start_run("task")
    turn = await journal.start_turn(run, 0)
    await journal.checkpoint(turn, {"status": "ready"})
    await journal.finish_run(run, "completed", {"output": "done"})

    with pytest.raises(HarnessError):
        await journal.start_turn(run, 1)
    with pytest.raises(HarnessError):
        await journal.record_prompt(turn, {"after": "finished"})
    with pytest.raises(HarnessError):
        await journal.finish_run(run, "completed", {"output": "again"})
    with pytest.raises(HarnessError):
        await journal.finish_run(run, "running", {})


@pytest.mark.asyncio
async def test_agent_journal_resume_validates_token_sequence() -> None:
    journal = InMemoryAgentJournal()
    run = await journal.start_run("task")
    turn = await journal.start_turn(run, 0)
    checkpoint = await journal.checkpoint(turn, {"status": "ready"})
    token = journal.resume_token(checkpoint)

    bad_token = type(token)(
        run_id=token.run_id,
        checkpoint_id=token.checkpoint_id,
        metadata={**token.metadata, "sequence": token.metadata["sequence"] + 1},
    )

    with pytest.raises(ResumeError):
        await journal.resume(bad_token)
    with pytest.raises(ResumeError):
        await journal.resume(type(token)(run_id=run.run_id, checkpoint_id="missing"))


@pytest.mark.asyncio
async def test_agent_journal_records_errors_and_run_manifest() -> None:
    journal = InMemoryAgentJournal()
    run = await journal.start_run("task", metadata={"profile": "core"})
    turn = await journal.start_turn(run, 0)
    checkpoint = await journal.checkpoint(turn, {"status": "mid"})
    await journal.record_error(run=run, turn=turn, error=ValueError("bad model"), metadata={"stage": "model"})

    manifest = journal.run_manifest(run.run_id)
    resumable = journal.resumable_runs()
    restored = InMemoryAgentJournal.from_manifest(journal.snapshot().manifest())

    assert manifest["run"]["metadata"]["profile"] == "core"
    assert manifest["errors"][0]["message"] == "bad model"
    assert manifest["errors"][0]["metadata"]["stage"] == "model"
    assert resumable[0]["checkpoint_id"] == checkpoint.checkpoint_id
    assert restored.errors[0]["error_type"] == "ValueError"
    assert restored.resumable_runs()[0]["checkpoint_sequence"] == 1

