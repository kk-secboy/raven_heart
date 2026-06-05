from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_event_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_event_acceptance(
        metadata={"test": "event_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-event-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["streaming_run"]["status"] == "completed"
    assert manifest["streaming_run"]["stream_requested"] is True
    assert manifest["streaming_run"]["model_stream_event_count"] >= 1
    assert manifest["streaming_run"]["tail"]["terminal"] is True
    assert manifest["manager_stream"]["status"] == "completed"
    assert manifest["manager_stream"]["before"]["event_count"] >= 1
    assert manifest["manager_stream"]["before"]["terminal"] is False
    assert manifest["manager_stream"]["after"]["terminal"] is True
    assert manifest["event_log"]["event_count"] >= 1
    assert manifest["trace_eval"]["ok"] is True
    assert manifest["trace_eval"]["summary"]["event_log_sequence_monotonic"] is True


def test_event_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "event_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreEventAcceptanceHarness" in agent_core.__all__
    assert "AgentCoreEventAcceptanceReport" in agent_core.__all__
    assert "run_agent_core_event_acceptance" in agent_core.__all__
    assert "AgentCoreEventAcceptanceHarness" not in readiness["matched"]["public_api"]
    assert "AgentCoreEventAcceptanceHarness" not in stability["present_stable_api"]
