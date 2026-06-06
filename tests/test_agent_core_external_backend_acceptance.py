from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_external_backend_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_external_backend_acceptance(
        metadata={"test": "external_backend_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-external-backend-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert set(manifest["backend_matrix"]["builtin_kinds"]) == {"sqlite", "markdown"}
    assert set(manifest["backend_matrix"]["external_kinds"]) == {
        "postgres",
        "vector",
        "graph",
        "product",
        "object_storage",
        "custom",
    }
    assert manifest["run"]["status"] == "completed"
    assert manifest["external_memory"]["call_count"] == 5
    assert set(manifest["external_memory"]["kinds"]) == {
        "postgres",
        "graph",
        "product",
        "custom",
    }
    assert manifest["external_context_material"]["call_count"] == 2
    assert set(manifest["external_context_material"]["kinds"]) == {
        "vector",
        "object_storage",
    }
    assert manifest["storage_preflight"]["ready"] is True
    assert set(manifest["storage_preflight"]["selected_kinds"]) == {"product", "vector"}
    assert manifest["trace_summary"]["external_storage_backend_count"] == 6
    assert manifest["trace_summary"]["memory_search_hit_count"] == 4
    assert manifest["trace_summary"]["context_material_selected_count"] == 2
    assert manifest["trace_eval"]["ok"] is True


def test_external_backend_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "external_backend_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreExternalBackendAcceptanceHarness" in agent_core.__all__
    assert "AgentCoreExternalBackendAcceptanceReport" in agent_core.__all__
    assert "run_agent_core_external_backend_acceptance" in agent_core.__all__
    assert "AgentCoreExternalBackendAcceptanceHarness" not in readiness["matched"][
        "public_api"
    ]
    assert "AgentCoreExternalBackendAcceptanceHarness" not in stability[
        "present_stable_api"
    ]
