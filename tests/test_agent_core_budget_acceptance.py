from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_budget_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_budget_acceptance(
        metadata={"test": "budget_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-budget-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["estimated_cost_block"]["blocked"] is True
    assert manifest["estimated_cost_block"]["provider_request_count"] == 0
    assert manifest["actual_cost_block"]["blocked"] is True
    assert manifest["actual_cost_block"]["provider_request_count"] == 1
    assert manifest["actual_cost_block"]["center_after_block"]["usage"]["cost_usd"] == 0.0
    assert manifest["override_success"]["completed"] is True
    assert manifest["override_success"]["center"]["usage"]["cost_usd"] == 0.01
    assert manifest["call_attempt_block"]["blocked"] is True
    assert manifest["call_attempt_block"]["provider_request_count"] == 1
    assert manifest["token_block"]["estimated_blocked"] is True
    assert manifest["token_block"]["estimated_provider_request_count"] == 0
    assert manifest["token_block"]["actual_blocked"] is True
    assert manifest["token_block"]["actual_provider_request_count"] == 1
    assert manifest["token_block"]["actual_center"]["usage"]["total_tokens"] == 0
    assert manifest["trace_eval"]["ok"] is True


def test_budget_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "budget_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreBudgetAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "AgentCoreBudgetAcceptanceReport" in readiness["matched"]["public_api"]
    assert "run_agent_core_budget_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreBudgetAcceptanceHarness" in stability["present_stable_api"]
    assert "AgentCoreBudgetAcceptanceReport" in stability["present_stable_api"]
    assert "run_agent_core_budget_acceptance" in stability["present_stable_api"]
