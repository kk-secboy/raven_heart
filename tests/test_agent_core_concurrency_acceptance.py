from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_concurrency_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_concurrency_acceptance(
        metadata={"test": "concurrency_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-concurrency-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["runs"]["statuses"] == {"alpha": "completed", "beta": "completed"}
    assert len(set(manifest["runs"]["run_keys"])) == 2
    assert len(set(manifest["runs"]["run_ids"])) == 2
    assert manifest["active_snapshot"]["active_run_count"] == 2
    assert manifest["manager"]["active_run_count"] == 0
    assert manifest["manager"]["queued_run_count"] == 0
    assert manifest["providers"]["alpha_request_count"] == 2
    assert manifest["providers"]["beta_request_count"] == 2
    assert manifest["tools"]["alpha_arguments"] == [{"owner": "alpha"}]
    assert manifest["tools"]["beta_arguments"] == [{"owner": "beta"}]
    assert manifest["event_streams"]["alpha"]["terminal"] is True
    assert manifest["event_streams"]["alpha"]["session_names"] == ["alpha"]
    assert manifest["event_streams"]["beta"]["terminal"] is True
    assert manifest["event_streams"]["beta"]["session_names"] == ["beta"]
    assert manifest["traces"]["record_count"] == 2
    assert manifest["trace_evals"]["alpha"]["ok"] is True
    assert manifest["trace_evals"]["beta"]["ok"] is True


def test_concurrency_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "concurrency_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreConcurrencyAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "AgentCoreConcurrencyAcceptanceReport" in readiness["matched"]["public_api"]
    assert "run_agent_core_concurrency_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreConcurrencyAcceptanceHarness" in stability["present_stable_api"]
    assert "AgentCoreConcurrencyAcceptanceReport" in stability["present_stable_api"]
    assert "run_agent_core_concurrency_acceptance" in stability["present_stable_api"]
