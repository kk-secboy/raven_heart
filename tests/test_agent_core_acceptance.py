from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_acceptance_scenario_passes_replacement_gate() -> None:
    import agent_core

    report = await agent_core.run_agent_core_acceptance(metadata={"test": "acceptance"})
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["readiness"]["status"] == "ready"
    assert manifest["trace_eval"]["ok"] is True
    assert manifest["run_summary"]["status"] == "completed"
    assert manifest["run_summary"]["output"] == "accepted"
    assert manifest["run_summary"]["provider_request_count"] == 2
    assert manifest["run_summary"]["tool_call_count"] == 1
    assert manifest["run_summary"]["memory_hit_count"] == 1
    assert manifest["run_summary"]["context_injection_count"] >= 1
    assert manifest["trace_summary"]["tool_center_failed_count"] == 0
    assert manifest["trace_summary"]["failure_count"] == 0
    assert "AgentRunner" in manifest["readiness"]["matched"]["public_api"]
    assert "prompt_context_semantics" in manifest["readiness"]["matched"]["capabilities"]
    assert "replacement_acceptance_harness" in manifest["readiness"]["matched"]["capabilities"]


@pytest.mark.asyncio
async def test_agent_core_acceptance_reports_readiness_blockers() -> None:
    import agent_core

    sdk_manifest = agent_core.agent_core_sdk_manifest().manifest()
    sdk_manifest["public_api"] = [
        name for name in sdk_manifest["public_api"] if name != "AgentRunner"
    ]

    report = await agent_core.AgentCoreAcceptanceHarness().run(sdk_manifest=sdk_manifest)
    manifest = report.manifest()
    issue_codes = {issue["code"] for issue in manifest["issues"]}

    assert manifest["status"] == "blocked"
    assert manifest["ready"] is False
    assert manifest["readiness"]["status"] == "blocked"
    assert "required_public_api_missing" in issue_codes
    assert manifest["trace_eval"]["ok"] is True
