from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_provider_conformance_default_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_provider_conformance(
        metadata={"test": "provider_conformance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-provider-conformance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["text"]["content_bytes"] > 0
    assert manifest["streaming"]["event_types"] == ["delta", "usage", "message_end"]
    assert manifest["json_mode"]["parsed_ok"] is True
    assert manifest["json_mode"]["parsed_keys"] == ["source", "status"]
    assert manifest["tool_calls"]["tool_call_names"] == ["lookup"]


@pytest.mark.asyncio
async def test_agent_core_provider_conformance_blocks_bad_provider() -> None:
    import agent_core
    from agent_core.providers import LLMRequest, LLMResponse

    class BadProvider:
        async def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(content="")

    spec = agent_core.AgentCoreProviderConformanceSpec(
        require_streaming=False,
        require_json_mode=False,
        require_tool_calls=False,
    )
    report = await agent_core.run_agent_core_provider_conformance(
        provider=BadProvider(),
        spec=spec,
        metadata={"test": "provider_conformance_bad"},
    )
    manifest = report.manifest()
    codes = {issue["code"] for issue in manifest["issues"]}

    assert manifest["status"] == "blocked"
    assert manifest["ready"] is False
    assert "text_response_empty" in codes


def test_provider_conformance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "provider_conformance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreProviderConformanceHarness" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderConformanceReport" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderConformanceSpec" in readiness["matched"]["public_api"]
    assert "run_agent_core_provider_conformance" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderConformanceHarness" in stability["present_stable_api"]
    assert "AgentCoreProviderConformanceReport" in stability["present_stable_api"]
    assert "AgentCoreProviderConformanceSpec" in stability["present_stable_api"]
    assert "run_agent_core_provider_conformance" in stability["present_stable_api"]
