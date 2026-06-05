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
    pressure = manifest["prompt_pressure_summary"]

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
    assert pressure["schema_version"] == "agent-core-context-pressure-summary/v1"
    assert pressure["within_target_bytes"] is True
    assert pressure["prompt_bytes"] <= pressure["target_prompt_bytes"]
    assert pressure["provider_request_prompt_bytes"] == pressure["prompt_bytes"]
    assert pressure["selected_context_names"] == [
        "auth_trace",
        "operator_hint",
        "risk_schema",
    ]
    assert pressure["dropped_context_statuses"] == {
        "denied_static": "target_denied",
        "old_dns_note": "count_exceeded",
    }
    assert pressure["required_fragments_present"] == {
        "auth_trace_marker": True,
        "current_task": True,
        "memory_recall_marker": True,
        "operator_hint_marker": True,
        "risk_schema_contract": True,
    }
    assert pressure["forbidden_fragments_absent"] == {
        "backup_noise": True,
        "denied_static": True,
        "inventory_noise": True,
        "old_dns_note": True,
    }
    assert pressure["semantic_trim_roles"] == ["timeline_open"]
    assert pressure["semantic_trim_dropped_units"] > 0
    assert pressure["bucket_budget_trimmed_roles"] == ["semi_dynamic_1"]


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
