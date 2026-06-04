from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_coordination_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_coordination_acceptance(
        metadata={"test": "coordination_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-coordination-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["planner"]["status"] == "completed"
    assert manifest["planner"]["plan"]["plan_id"] == "coordination-plan"
    assert manifest["planner"]["plan"]["status_counts"]["completed"] == 2
    assert manifest["handoff"]["decision"]["status"] == "selected"
    assert manifest["handoff"]["decision"]["selected_session"] == "code-reviewer"
    assert manifest["handoff"]["outcome"]["result"]["status"] == "completed"
    assert manifest["agent_tool"]["result"]["status"] == "completed"
    assert manifest["agent_tool"]["runtime"]["record_count"] == 1
    assert manifest["artifacts"]["artifacts"][0]["content_type"] == "text/markdown"
    assert manifest["artifacts"]["artifacts"][0]["metadata"]["kind"] == "coordination_evidence"
    assert manifest["traces"]["planner"]["execution_report_count"] == 1
    assert manifest["traces"]["handoff"]["selected_sessions"] == {"code-reviewer": 1}
    assert manifest["traces"]["agent_tool"]["sessions"] == {"code-reviewer": 1}
    assert manifest["traces"]["artifact"]["artifact_count"] == 1
    assert manifest["trace_bundle"]["summary"]["planner_execution_report_count"] == 1
    assert manifest["trace_bundle"]["summary"]["handoff_selected_count"] == 1
    assert manifest["trace_bundle"]["summary"]["agent_tool_completed_count"] == 1
    assert manifest["trace_bundle"]["summary"]["artifact_count"] == 1
    assert manifest["trace_eval"]["ok"] is True


def test_coordination_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "coordination_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCoreCoordinationAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "AgentCoreCoordinationAcceptanceReport" in readiness["matched"]["public_api"]
    assert "run_agent_core_coordination_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreCoordinationAcceptanceHarness" in stability["present_stable_api"]
    assert "AgentCoreCoordinationAcceptanceReport" in stability["present_stable_api"]
    assert "run_agent_core_coordination_acceptance" in stability["present_stable_api"]
