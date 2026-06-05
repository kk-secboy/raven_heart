from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_interrupt_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_interrupt_acceptance(
        metadata={"test": "interrupt_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-interrupt-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["pre_cancel"]["status"] == "cancelled"
    assert manifest["pre_cancel"]["interrupt"]["kind"] == "cancelled"
    assert manifest["pre_cancel"]["provider_request_count"] == 0
    assert "run_cancelled" in manifest["pre_cancel"]["event_types"]
    assert manifest["provider_timeout"]["status"] == "timeout"
    assert manifest["provider_timeout"]["interrupt"]["kind"] == "timeout"
    assert manifest["provider_timeout"]["provider_request_count"] == 1
    assert "provider" in manifest["provider_timeout"]["checkpoint_phases"]
    assert manifest["tool_timeout"]["status"] == "timeout"
    assert manifest["tool_timeout"]["interrupt"]["kind"] == "timeout"
    assert manifest["tool_timeout"]["tool_invocation_count"] == 1
    assert "tool" in manifest["tool_timeout"]["checkpoint_phases"]
    assert manifest["manager_cancel_reset"]["cancel_result"] is True
    assert manifest["manager_cancel_reset"]["first_status"] == "cancelled"
    assert manifest["manager_cancel_reset"]["second_status"] == "completed"
    assert manifest["manager_cancel_reset"]["cancel_token_after_second"]["cancelled"] is False
    assert manifest["manager_cancel_reset"]["provider_request_count"] == 2
    assert manifest["trace_evals"]["pre_cancel"]["ok"] is True
    assert manifest["trace_evals"]["provider_timeout"]["ok"] is True
    assert manifest["trace_evals"]["tool_timeout"]["ok"] is True
    assert manifest["trace_evals"]["manager_cancel_reset"]["ok"] is True


def test_interrupt_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "interrupt_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreInterruptAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "AgentCoreInterruptAcceptanceReport" in readiness["matched"]["public_api"]
    assert "run_agent_core_interrupt_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreInterruptAcceptanceHarness" in stability["present_stable_api"]
    assert "AgentCoreInterruptAcceptanceReport" in stability["present_stable_api"]
    assert "run_agent_core_interrupt_acceptance" in stability["present_stable_api"]
