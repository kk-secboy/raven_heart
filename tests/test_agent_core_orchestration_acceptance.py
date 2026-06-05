from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_orchestration_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_orchestration_acceptance(
        metadata={"test": "orchestration_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-orchestration-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["capability_discovery"]["counts"]["action"] == 1
    assert manifest["capability_discovery"]["counts"]["tool"] >= 1
    assert manifest["capability_discovery"]["counts"]["skill"] == 1
    assert manifest["capability_discovery"]["counts"]["mcp_tool"] == 1
    assert manifest["capability_discovery"]["counts"]["mcp_resource"] == 1
    assert manifest["capability_discovery"]["counts"]["mcp_prompt"] == 1
    assert manifest["capability_discovery"]["counts"]["mcp_server"] == 1
    assert manifest["capability_prompt"]["sections"]["action_inventory"] is True
    assert manifest["capability_prompt"]["sections"]["tool_inventory"] is True
    assert manifest["capability_prompt"]["sections"]["mcp_servers"] is True
    assert manifest["capability_prompt"]["sections"]["skills_context"] is True
    assert manifest["tool_center"]["call_count"] == 2
    assert manifest["mcp_inventory"]["results"][0]["status"] == "refreshed"
    assert manifest["mcp_context_materials"]["material_count"] == 2
    assert manifest["mcp_context_materials"]["failed_count"] == 0
    assert manifest["skill_center"]["loaded_skills"][0]["name"] == "project-review"
    assert manifest["traces"]["tool_center"]["failed_count"] == 0
    assert manifest["traces"]["mcp_center"]["refreshed_servers"] == ["fs"]
    assert manifest["traces"]["skill_center"]["loaded_skill_names"] == ["project-review"]
    assert manifest["execution_summary"]["local_tool_ok"] is True
    assert manifest["execution_summary"]["mcp_tool_ok"] is True


def test_orchestration_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "orchestration_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreOrchestrationAcceptanceHarness" in agent_core.__all__
    assert "AgentCoreOrchestrationAcceptanceReport" in agent_core.__all__
    assert "run_agent_core_orchestration_acceptance" in agent_core.__all__
    assert "AgentCoreOrchestrationAcceptanceHarness" not in readiness["matched"][
        "public_api"
    ]
    assert "AgentCoreOrchestrationAcceptanceHarness" not in stability[
        "present_stable_api"
    ]
