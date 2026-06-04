from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_approval_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_approval_acceptance(
        metadata={"test": "approval_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-approval-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["blocked_run"]["approval_pending_count"] == 1
    assert manifest["blocked_run"]["provider_request_count"] == 2
    assert manifest["approved_run"]["approval_approved_count"] == 1
    assert manifest["approved_run"]["approval_pending_count"] == 0
    assert manifest["approved_run"]["context_injection_count"] == 1
    assert manifest["approval_queue"]["queue"]["approved_count"] == 1
    assert manifest["approval_queue"]["queue"]["pending_count"] == 0
    assert manifest["approval_resume"]["approved_count"] == 1
    assert manifest["approval_resume"]["grants"][0]["subject"] == "tool:deploy"
    assert manifest["policy_decisions"]["record_count"] >= 2
    assert manifest["blocked_trace_eval"]["ok"] is True
    assert manifest["approved_trace_eval"]["ok"] is True


def test_approval_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "approval_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreApprovalAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "AgentCoreApprovalAcceptanceReport" in readiness["matched"]["public_api"]
    assert "run_agent_core_approval_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreApprovalAcceptanceHarness" in stability["present_stable_api"]
    assert "AgentCoreApprovalAcceptanceReport" in stability["present_stable_api"]
    assert "run_agent_core_approval_acceptance" in stability["present_stable_api"]
