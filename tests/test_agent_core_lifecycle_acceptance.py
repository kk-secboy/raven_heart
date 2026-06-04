from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_lifecycle_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_lifecycle_acceptance(
        metadata={"test": "lifecycle_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-lifecycle-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["queue"]["before"]["queued_run_count"] == 1
    assert manifest["queue"]["after"]["queued_run_count"] == 0
    assert manifest["queue"]["first_status"] == "completed"
    assert manifest["queue"]["second_status"] == "completed"
    assert manifest["queued_cancel"]["cancel_result"] is True
    assert manifest["queued_cancel"]["cancelled_status"] == "cancelled"
    assert manifest["queued_cancel"]["queued_interrupt"]["queued"] is True
    assert manifest["queued_cancel"]["second_provider_request_count"] == 0
    assert manifest["active_cancel"]["cancel_result"] is True
    assert manifest["active_cancel"]["cancelling"]["status"] == "cancelling"
    assert manifest["active_cancel"]["status"] == "cancelled"
    assert manifest["timeout"]["status"] == "timeout"
    assert manifest["timeout"]["run_status"] == "timeout"
    assert manifest["timeout"]["interrupt"]["kind"] == "timeout"
    assert [event["type"] for event in manifest["timeout"]["event_log"]["events"]][
        -1
    ] == "run_timeout"
    assert manifest["trace_eval"]["ok"] is True


def test_lifecycle_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "lifecycle_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreLifecycleAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "AgentCoreLifecycleAcceptanceReport" in readiness["matched"]["public_api"]
    assert "run_agent_core_lifecycle_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreLifecycleAcceptanceHarness" in stability["present_stable_api"]
    assert "AgentCoreLifecycleAcceptanceReport" in stability["present_stable_api"]
    assert "run_agent_core_lifecycle_acceptance" in stability["present_stable_api"]
