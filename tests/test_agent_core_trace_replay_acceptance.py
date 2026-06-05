from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_trace_replay_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_trace_replay_acceptance(
        metadata={"test": "trace_replay_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-trace-replay-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["current_replay"]["step_count"] >= 6
    assert "tool_finished" in manifest["current_replay"]["event_types"]
    assert "provider_call_completed" in manifest["current_replay"]["event_types"]
    assert manifest["current_replay"]["trace_store"]["record_count"] == 1
    assert manifest["stored_eval"]["ok"] is True
    assert manifest["compatibility_compare"]["ok"] is True
    assert manifest["drift_detection"]["ok"] is False
    assert manifest["drift_detection"]["issue_count"] >= 1
    assert "missing_step" in manifest["drift_detection"]["issue_codes"]
    assert manifest["legacy_replay"]["step_count"] == 2
    assert manifest["legacy_replay"]["eval_ok"] is True
    assert manifest["legacy_replay"]["event_types"] == ["run_started", "run_finished"]


def test_trace_replay_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "trace_replay_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreTraceReplayAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "AgentCoreTraceReplayAcceptanceReport" in readiness["matched"]["public_api"]
    assert "run_agent_core_trace_replay_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreTraceReplayAcceptanceHarness" in stability["present_stable_api"]
    assert "AgentCoreTraceReplayAcceptanceReport" in stability["present_stable_api"]
    assert "run_agent_core_trace_replay_acceptance" in stability["present_stable_api"]
