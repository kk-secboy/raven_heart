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
    assert manifest["check_matrix"]["provider_supplied"] is False
    assert manifest["check_matrix"]["provider_source"] == "deterministic"
    assert manifest["check_matrix"]["required_checks"] == [
        "text",
        "streaming",
        "json_mode",
        "tool_calls",
    ]
    assert manifest["check_matrix"]["completed_required_checks"] == [
        "text",
        "streaming",
        "json_mode",
        "tool_calls",
    ]
    assert manifest["check_matrix"]["failed_required_checks"] == []
    assert "LLMProviderPort.complete" in manifest["check_matrix"]["core_contracts"]
    assert (
        "API keys and secret loading"
        in manifest["check_matrix"]["runtime_responsibilities"]
    )


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
    assert manifest["check_matrix"]["provider_supplied"] is True
    assert manifest["check_matrix"]["provider_source"] == "external"
    assert manifest["check_matrix"]["required_checks"] == ["text"]
    assert manifest["check_matrix"]["optional_checks"] == [
        "streaming",
        "json_mode",
        "tool_calls",
    ]
    assert manifest["check_matrix"]["skipped_checks"] == [
        "streaming",
        "json_mode",
        "tool_calls",
    ]
    assert manifest["check_matrix"]["error_issue_codes"] == ["text_response_empty"]
    assert manifest["check_matrix"]["ready_for_real_provider_smoke"] is False


@pytest.mark.asyncio
async def test_provider_conformance_marks_external_provider_smoke_ready() -> None:
    import agent_core
    from agent_core.providers import LLMRequest, LLMResponse, UsageInfo

    class TextProvider:
        async def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(
                content="external-provider-ok",
                finish_reason="stop",
                usage=UsageInfo(total_tokens=3),
            )

    report = await agent_core.run_agent_core_provider_conformance(
        provider=TextProvider(),
        spec=agent_core.AgentCoreProviderConformanceSpec(
            provider_name="external-text",
            require_streaming=False,
            require_json_mode=False,
            require_tool_calls=False,
        ),
    )
    manifest = report.manifest()

    assert manifest["status"] == "ready"
    assert manifest["check_matrix"]["provider_supplied"] is True
    assert manifest["check_matrix"]["provider_source"] == "external"
    assert manifest["check_matrix"]["required_checks"] == ["text"]
    assert manifest["check_matrix"]["error_issue_codes"] == []
    assert manifest["check_matrix"]["ready_for_real_provider_smoke"] is True


def test_provider_conformance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "provider_conformance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreProviderConformanceHarness" in agent_core.__all__
    assert "AgentCoreProviderConformanceReport" in agent_core.__all__
    assert "AgentCoreProviderConformanceSpec" in agent_core.__all__
    assert "run_agent_core_provider_conformance" in agent_core.__all__
    assert "AgentCoreProviderConformanceHarness" not in readiness["matched"]["public_api"]
    assert "AgentCoreProviderConformanceHarness" not in stability["present_stable_api"]
