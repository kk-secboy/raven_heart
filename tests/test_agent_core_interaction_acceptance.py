from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_interaction_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_interaction_acceptance(
        metadata={"test": "interaction_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-interaction-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["run"]["status"] == "completed"
    assert manifest["run"]["stream_request_count"] == 2
    assert manifest["run"]["native_tool_calls"] is True
    assert manifest["provider_request"]["first_tool_names"] == ["lookup"]
    assert manifest["provider_request"]["first_response_format"]["name"] == "risk_summary"
    assert manifest["provider_request"]["second_last_message"]["role"] == "tool"
    assert manifest["provider_stream"]["tool_call_names"] == ["lookup"]
    assert "tool_call" in manifest["provider_stream"]["event_types"]
    assert "action" in manifest["provider_stream"]["event_types"]
    assert manifest["tool_runtime"]["invocation_count"] == 1
    assert manifest["tool_runtime"]["call_ids"] == ["interaction-call-1"]
    assert manifest["structured_output"]["ok_count"] == 1
    assert manifest["structured_output"]["schema_names"]["risk_summary"] == 1
    assert manifest["event_log"]["model_stream_event_count"] >= 1
    assert manifest["event_log"]["terminal"] is True
    assert manifest["trace_summary"]["provider_stream_call_count"] == 2
    assert manifest["trace_summary"]["provider_tool_result_execution_count"] == 1
    assert manifest["trace_summary"]["structured_output_ok"] is True
    assert manifest["trace_eval"]["ok"] is True


def test_interaction_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "interaction_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreInteractionAcceptanceHarness" in agent_core.__all__
    assert "AgentCoreInteractionAcceptanceReport" in agent_core.__all__
    assert "run_agent_core_interaction_acceptance" in agent_core.__all__
    assert "AgentCoreInteractionAcceptanceHarness" not in readiness["matched"]["public_api"]
    assert "AgentCoreInteractionAcceptanceHarness" not in stability["present_stable_api"]
