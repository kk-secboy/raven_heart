from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_durable_session_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_durable_session_acceptance(
        metadata={"test": "durable_session_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-durable-session-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["trace_eval"]["sqlite"]["ok"] is True
    assert manifest["trace_eval"]["markdown"]["ok"] is True
    for backend in ("sqlite", "markdown"):
        roundtrip = manifest[f"{backend}_roundtrip"]
        reopened = roundtrip["reopened"]
        assert roundtrip["status"] == "completed"
        assert roundtrip["output"] == f"{backend}:durable-ok"
        assert roundtrip["provider_request_count"] == 2
        assert reopened["journal_run_count"] == 1
        assert reopened["journal_checkpoint_count"] >= 1
        assert reopened["journal_finished_count"] == 1
        assert reopened["trace_count"] == 1
        assert reopened["loaded_trace_loaded"] is True
        assert reopened["event_count"] >= 1
        assert reopened["tool_replay_count"] == 1
        assert reopened["memory_hit_count"] == 1
        assert reopened["context_material_count"] == 1
        assert reopened["policy_decision_count"] >= 1
        assert reopened["manager_run_count"] == 1
        assert reopened["manager_completed_count"] == 1


def test_durable_session_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "durable_session_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreDurableSessionAcceptanceHarness" in agent_core.__all__
    assert "AgentCoreDurableSessionAcceptanceReport" in agent_core.__all__
    assert "run_agent_core_durable_session_acceptance" in agent_core.__all__
    assert "AgentCoreDurableSessionAcceptanceHarness" not in readiness["matched"][
        "public_api"
    ]
    assert "AgentCoreDurableSessionAcceptanceHarness" not in stability[
        "present_stable_api"
    ]
