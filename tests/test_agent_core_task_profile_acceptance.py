from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_task_profile_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_task_profile_acceptance(
        metadata={"test": "task_profile_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-task-profile-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["matrix"]["profile_names"] == ["code", "ops", "security"]
    assert manifest["matrix"]["completed_profiles"] == ["code", "ops", "security"]
    assert manifest["matrix"]["tool_enabled_profiles"] == ["code", "ops", "security"]
    assert manifest["matrix"]["memory_enabled_profiles"] == ["code", "ops", "security"]
    assert manifest["matrix"]["context_enabled_profiles"] == ["code", "ops", "security"]
    assert manifest["matrix"]["skill_names"] == [
        "code-review",
        "ops-triage",
        "security-review",
    ]
    for profile in manifest["profiles"]:
        assert profile["status"] == "completed"
        assert profile["provider_request_count"] == 2
        assert profile["tool_call_count"] == 1
        assert profile["memory_hit_count"] == 1
        assert profile["context_material_selected_count"] == 1
        assert profile["context_injection_count"] == 2
        assert profile["trace_eval"]["ok"] is True


def test_task_profile_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "task_profile_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreTaskProfileAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "AgentCoreTaskProfileAcceptanceReport" in readiness["matched"]["public_api"]
    assert "run_agent_core_task_profile_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreTaskProfileAcceptanceHarness" in stability["present_stable_api"]
    assert "AgentCoreTaskProfileAcceptanceReport" in stability["present_stable_api"]
    assert "run_agent_core_task_profile_acceptance" in stability["present_stable_api"]
