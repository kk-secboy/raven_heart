from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_validation_suite_runs_all_sdk_gates() -> None:
    import agent_core

    report = await agent_core.run_agent_core_validation(metadata={"test": "validation"})
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-validation-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["readiness"]["ready"] is True
    assert manifest["api_stability"]["ready"] is True
    assert manifest["acceptance"]["ready"] is True
    assert manifest["recovery"]["ready"] is True
    assert manifest["resume"]["ready"] is True
    assert manifest["acceptance"]["run_summary"]["tool_call_count"] == 1
    assert manifest["recovery"]["provider_recovery"]["fallback_used"] is True
    assert manifest["resume"]["trace_eval"]["ok"] is True


@pytest.mark.asyncio
async def test_agent_core_validation_suite_blocks_missing_stable_api() -> None:
    import agent_core

    sdk_manifest = agent_core.agent_core_sdk_manifest().manifest()
    sdk_manifest["public_api"] = [
        name for name in sdk_manifest["public_api"] if name != "AgentRunner"
    ]

    report = await agent_core.AgentCoreValidationSuite().run(sdk_manifest=sdk_manifest)
    manifest = report.manifest()
    issue_codes = {issue["code"] for issue in manifest["issues"]}

    assert manifest["status"] == "blocked"
    assert manifest["ready"] is False
    assert manifest["api_stability"]["ready"] is False
    assert "stable_api_missing" in issue_codes


def test_validation_suite_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "validation_suite" in readiness["matched"]["capabilities"]
    assert "AgentCoreValidationSuite" in readiness["matched"]["public_api"]
    assert "run_agent_core_validation" in readiness["matched"]["public_api"]
    assert "AgentCoreValidationSuite" in stability["present_stable_api"]
    assert "run_agent_core_validation" in stability["present_stable_api"]
