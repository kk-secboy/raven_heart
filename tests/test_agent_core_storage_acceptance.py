from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_storage_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_storage_acceptance(
        metadata={"test": "storage_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-storage-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["memory_roundtrip"]["sqlite_hit_count"] == 1
    assert manifest["memory_roundtrip"]["markdown_hit_count"] == 1
    assert manifest["memory_roundtrip"]["center_hit_count"] == 2
    assert manifest["context_roundtrip"]["sqlite_hit_count"] == 1
    assert manifest["context_roundtrip"]["markdown_hit_count"] == 1
    assert manifest["context_roundtrip"]["center_hit_count"] == 2
    assert "memory" in manifest["builtin_matrix"]["roles"]
    assert "sqlite" in manifest["builtin_matrix"]["kinds"]
    assert "markdown" in manifest["builtin_matrix"]["kinds"]
    assert "postgres" in {
        backend["kind"] for backend in manifest["external_contracts"]["backends"]
    }
    assert manifest["preflight"]["ready"] is True
    assert {"postgres", "vector", "object_storage"} <= set(
        manifest["preflight"]["selected_kinds"]
    )
    assert manifest["blocked_preflight"]["status"] == "blocked"
    assert manifest["blocked_preflight"]["ready"] is False


def test_storage_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "storage_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreStorageAcceptanceHarness" in agent_core.__all__
    assert "AgentCoreStorageAcceptanceReport" in agent_core.__all__
    assert "run_agent_core_storage_acceptance" in agent_core.__all__
    assert "AgentCoreStorageAcceptanceHarness" not in readiness["matched"]["public_api"]
    assert "AgentCoreStorageAcceptanceHarness" not in stability["present_stable_api"]
