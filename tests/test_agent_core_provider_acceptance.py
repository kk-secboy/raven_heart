from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_provider_acceptance_matrix_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_provider_acceptance(
        metadata={"test": "provider_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-provider-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["route_matrix"]["selected"] == {
        "structured": "structured",
        "text": "text",
        "tools": "tools",
        "vision": "vision",
    }
    assert manifest["route_matrix"]["shape_adjusted"]["structured"] is True
    assert manifest["codec_matrix"]["encoded_has_tools"] is True
    assert manifest["codec_matrix"]["encoded_has_response_format"] is True
    assert manifest["codec_matrix"]["encoded_has_multimodal_content"] is True
    assert manifest["codec_matrix"]["decoded_tool_call_names"] == ["lookup"]
    assert (
        manifest["contract_matrix"]["default_payload_schema"]
        == "agent-core-llm-transport-request/v1"
    )
    assert manifest["contract_matrix"]["default_payload_has_content_parts"] is True
    assert manifest["contract_matrix"]["default_payload_tool_choice_mode"] == "tool"
    assert manifest["contract_matrix"]["default_payload_response_format_kind"] == "json_schema"
    assert manifest["contract_matrix"]["request_manifest_redacts_uri"] is True
    assert manifest["contract_matrix"]["default_response_tool_call_names"] == ["lookup"]
    assert manifest["contract_matrix"]["default_response_usage_total_tokens"] == 13
    assert (
        manifest["contract_matrix"]["default_response_transport_provider"]
        == "default-contract"
    )
    assert manifest["contract_matrix"]["default_stream_event_types"] == [
        "delta",
        "tool_call",
        "usage",
        "message_end",
    ]
    assert manifest["contract_matrix"]["default_stream_tool_call_names"] == ["lookup"]
    assert manifest["contract_matrix"]["default_stream_usage_total_tokens"] == [8]
    assert manifest["contract_matrix"]["openai_stream_tool_event_type"] == "tool_call"
    assert manifest["contract_matrix"]["openai_stream_tool_call_name"] == "lookup"
    assert manifest["contract_matrix"]["openai_stream_error_type"] == "error"
    assert manifest["contract_matrix"]["openai_stream_error_retryable"] is True
    assert manifest["transport_matrix"]["fallback_used"] is True
    assert manifest["transport_matrix"]["call_statuses"] == ["failed", "completed"]
    assert manifest["streaming_matrix"]["selected_provider"] == "stream"
    assert manifest["streaming_matrix"]["event_types"] == [
        "delta",
        "usage",
        "message_end",
    ]


def test_provider_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "provider_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreProviderAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderAcceptanceReport" in readiness["matched"]["public_api"]
    assert "run_agent_core_provider_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderAcceptanceHarness" in stability["present_stable_api"]
    assert "AgentCoreProviderAcceptanceReport" in stability["present_stable_api"]
    assert "run_agent_core_provider_acceptance" in stability["present_stable_api"]
