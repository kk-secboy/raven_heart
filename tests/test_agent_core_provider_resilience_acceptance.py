from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_provider_resilience_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_provider_resilience_acceptance(
        metadata={"test": "provider_resilience_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-provider-resilience-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["retry_completion"]["status"] == "completed"
    assert manifest["retry_completion"]["primary_request_count"] == 2
    assert manifest["retry_completion"]["fallback_request_count"] == 1
    assert manifest["retry_completion"]["retryable_failed_call_count"] == 2
    assert manifest["retry_after"]["status"] == "completed"
    assert manifest["retry_after"]["primary_request_count"] == 1
    assert manifest["retry_after"]["fallback_request_count"] == 1
    assert manifest["retry_after"]["retryable_failed_call_count"] == 0
    assert manifest["retry_after"]["max_retry_after_seconds"] == 5.0
    assert manifest["non_retry_fallback"]["status"] == "completed"
    assert manifest["non_retry_fallback"]["primary_request_count"] == 1
    assert manifest["non_retry_fallback"]["fallback_request_count"] == 1
    assert manifest["explicit_provider"]["status"] == "blocked_as_expected"
    assert manifest["explicit_provider"]["fallback_request_count"] == 0
    assert manifest["stream_fallback"]["status"] == "completed"
    assert manifest["stream_fallback"]["primary_request_count"] == 2
    assert manifest["stream_fallback"]["fallback_request_count"] == 1
    assert manifest["stream_fallback"]["event_types"] == ["delta", "usage", "message_end"]
    assert manifest["trace_evals"]["retry_completion"]["ok"] is True
    assert manifest["trace_evals"]["retry_after"]["ok"] is True
    assert manifest["trace_evals"]["non_retry_fallback"]["ok"] is True
    assert manifest["trace_evals"]["explicit_provider"]["ok"] is True
    assert manifest["trace_evals"]["stream_fallback"]["ok"] is True


def test_provider_resilience_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "provider_resilience_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreProviderResilienceAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderResilienceAcceptanceReport" in readiness["matched"]["public_api"]
    assert "run_agent_core_provider_resilience_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderResilienceAcceptanceHarness" in stability["present_stable_api"]
    assert "AgentCoreProviderResilienceAcceptanceReport" in stability["present_stable_api"]
    assert "run_agent_core_provider_resilience_acceptance" in stability["present_stable_api"]
