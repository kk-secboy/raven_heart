from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_native_tool_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_native_tool_acceptance(
        metadata={"test": "native_tool_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-native-tool-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["run"]["status"] == "completed"
    assert manifest["run"]["native_tool_calls"] is True
    assert manifest["provider_request"]["first_tool_names"] == ["lookup"]
    assert manifest["provider_request"]["second_last_message"]["role"] == "tool"
    assert manifest["provider_request"]["second_last_message"]["content_bytes"] == len(
        "found"
    )
    assert manifest["tool_runtime"]["invocation_count"] == 1
    assert manifest["tool_runtime"]["call_ids"] == ["call-1"]
    assert manifest["journal_tool_call"]["metadata"]["provider_tool_call"][
        "call_id"
    ] == "call-1"
    assert manifest["trace_summary"]["provider_tool_call_count"] == 1
    assert manifest["trace_summary"]["provider_tool_result_execution_count"] == 1
    assert manifest["trace_summary"]["tool_schema_invalid_count"] == 0
    assert manifest["trace_eval"]["ok"] is True


def test_native_tool_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "native_tool_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreNativeToolAcceptanceHarness" in agent_core.__all__
    assert "AgentCoreNativeToolAcceptanceReport" in agent_core.__all__
    assert "run_agent_core_native_tool_acceptance" in agent_core.__all__
    assert "AgentCoreNativeToolAcceptanceHarness" not in readiness["matched"]["public_api"]
    assert "AgentCoreNativeToolAcceptanceHarness" not in stability["present_stable_api"]
