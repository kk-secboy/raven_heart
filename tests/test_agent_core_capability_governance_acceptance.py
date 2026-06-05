from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_capability_governance_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_capability_governance_acceptance(
        metadata={"test": "capability_governance_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == (
        "agent-core-capability-governance-acceptance-report/v1"
    )
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["tool_governance"]["visible_tool_names"] == [
        "mcp__fs__read_file",
        "safe_lookup",
    ]
    assert manifest["tool_governance"]["local_disabled_plan"]["ready"] is False
    assert manifest["tool_governance"]["mcp_disabled_plan"]["ready"] is False
    assert manifest["tool_governance"]["disabled_local_status"] == "failed"
    assert manifest["tool_governance"]["disabled_mcp_status"] == "failed"
    assert manifest["mcp_governance"]["inventory_statuses"] == {
        "broken": "failed",
        "disabled": "disabled",
        "fs": "refreshed",
    }
    assert manifest["mcp_governance"]["visible_mcp_tools"] == ["mcp__fs__read_file"]
    assert manifest["mcp_governance"]["audit_mcp_tools"] == [
        "mcp__fs__delete_file",
        "mcp__fs__read_file",
    ]
    assert "danger-admin" not in manifest["skill_governance"]["auto_selected"]
    assert manifest["skill_governance"]["safe_selected"] == ["safe-review"]
    assert manifest["prompt_governance"]["contains_delete_target"] is False
    assert manifest["prompt_governance"]["contains_mcp_delete_file"] is False
    assert manifest["traces"]["tool_center"]["failed_count"] == 2
    assert manifest["traces"]["mcp_center"]["failed_servers"] == ["broken"]
    assert manifest["traces"]["mcp_center"]["disabled_servers"] == ["disabled"]
    assert manifest["traces"]["skill_center"]["loaded_skill_names"] == ["safe-review"]


def test_capability_governance_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "capability_governance_acceptance_harness" in readiness["matched"][
        "capabilities"
    ]
    assert "AgentCoreCapabilityGovernanceAcceptanceHarness" in agent_core.__all__
    assert "AgentCoreCapabilityGovernanceAcceptanceReport" in agent_core.__all__
    assert "run_agent_core_capability_governance_acceptance" in agent_core.__all__
    assert "AgentCoreCapabilityGovernanceAcceptanceHarness" not in readiness["matched"][
        "public_api"
    ]
    assert "AgentCoreCapabilityGovernanceAcceptanceHarness" not in stability[
        "present_stable_api"
    ]
