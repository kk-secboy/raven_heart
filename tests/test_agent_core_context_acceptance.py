from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_context_acceptance_scenario_passes_context_gate() -> None:
    import agent_core

    report = await agent_core.run_agent_core_context_acceptance(
        metadata={"test": "context_acceptance"}
    )
    manifest = report.manifest()
    summary = manifest["context_summary"]

    assert manifest["schema_version"] == "agent-core-context-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["trace_eval"]["ok"] is True
    assert summary["context_material_selection_count"] == 5
    assert summary["context_material_selected_count"] == 3
    assert summary["context_material_dropped_count"] == 2
    assert summary["selection_statuses"] == {
        "count_exceeded": 1,
        "selected": 3,
        "target_denied": 1,
    }
    assert summary["context_injection_count"] == 4
    assert summary["context_injection_trimmed_count"] == 3
    assert summary["prompt_bucket_budget_trimmed_count"] == 1
    assert summary["prompt_semantic_trimmed_count"] == 1
    assert summary["semantic_trim_statuses"]["timeline_open"] == "trimmed"
    assert summary["bucket_budget_statuses"]["semi_dynamic_1"] == "trimmed"


def test_context_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "context_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreContextAcceptanceHarness" in agent_core.__all__
    assert "AgentCoreContextAcceptanceReport" in agent_core.__all__
    assert "run_agent_core_context_acceptance" in agent_core.__all__
    assert "AgentCoreContextAcceptanceHarness" not in readiness["matched"]["public_api"]
    assert "AgentCoreContextAcceptanceHarness" not in stability["present_stable_api"]
