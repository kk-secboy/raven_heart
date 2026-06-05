from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_recovery_acceptance_records_provider_fallback_and_tool_retry() -> None:
    import agent_core

    report = await agent_core.run_agent_core_recovery_acceptance(metadata={"test": "recovery"})
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-recovery-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["provider_recovery"]["status"] == "completed"
    assert manifest["provider_recovery"]["fallback_used"] is True
    assert manifest["provider_recovery"]["primary_request_count"] == 2
    assert manifest["provider_recovery"]["fallback_request_count"] == 1
    assert manifest["provider_recovery"]["call_statuses"] == [
        "failed",
        "failed",
        "completed",
    ]
    assert manifest["provider_recovery"]["retryable_failed_call_count"] == 2
    assert manifest["provider_recovery"]["error_kinds"] == [
        "provider_failed",
        "provider_failed",
    ]
    assert manifest["tool_recovery"]["ok"] is True
    assert manifest["tool_recovery"]["retried"] is True
    assert manifest["tool_recovery"]["attempt_count"] == 2
    assert manifest["tool_recovery"]["retryable_attempts"] == [1]
    assert manifest["tool_recovery"]["attempt_statuses"] == ["failed", "completed"]
    assert manifest["tool_recovery"]["error_kinds"] == ["tool_failed"]


def test_recovery_acceptance_is_declared_in_sdk_readiness_profile() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()

    assert "recovery_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreRecoveryHarness" in agent_core.__all__
    assert "run_agent_core_recovery_acceptance" in agent_core.__all__
    assert "AgentCoreRecoveryHarness" not in readiness["matched"]["public_api"]
