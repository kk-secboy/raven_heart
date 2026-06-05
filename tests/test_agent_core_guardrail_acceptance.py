from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_guardrail_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_guardrail_acceptance(
        metadata={"test": "guardrail_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-guardrail-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["preflight"]["status"] == "denied"
    assert manifest["preflight"]["preflight_status"] == "blocked"
    assert set(manifest["preflight"]["blocking_codes"]) == {
        "missing_tool",
        "task_bytes_exceeded",
    }
    assert manifest["policy"]["status"] == "completed"
    assert manifest["policy"]["output"] == "safe"
    assert manifest["policy"]["denied_count"] == 1
    assert manifest["policy"]["denied_subjects"] == ["tool:delete_target"]
    assert manifest["policy"]["executed_tool_count"] == 0
    assert manifest["structured_output"]["status"] == "completed"
    assert manifest["structured_output"]["repair_count"] == 1
    assert manifest["structured_output"]["ok_count"] == 1
    assert manifest["structured_output"]["schema_names"]["risk_summary"] == 1
    assert {name: item["ok"] for name, item in manifest["trace_evals"].items()} == {
        "preflight": True,
        "policy": True,
        "structured_output": True,
    }


def test_guardrail_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "guardrail_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreGuardrailAcceptanceHarness" in agent_core.__all__
    assert "AgentCoreGuardrailAcceptanceReport" in agent_core.__all__
    assert "run_agent_core_guardrail_acceptance" in agent_core.__all__
    assert "AgentCoreGuardrailAcceptanceHarness" not in readiness["matched"]["public_api"]
    assert "AgentCoreGuardrailAcceptanceHarness" not in stability["present_stable_api"]
