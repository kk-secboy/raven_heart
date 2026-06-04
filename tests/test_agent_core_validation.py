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
    assert manifest["runtime_boundary"]["ready"] is True
    assert manifest["runtime_boundary"]["hit_count"] == 0
    assert manifest["readiness"]["ready"] is True
    assert manifest["api_stability"]["ready"] is True
    assert manifest["acceptance"]["ready"] is True
    assert manifest["approval_acceptance"]["ready"] is True
    assert manifest["context_acceptance"]["ready"] is True
    assert manifest["orchestration_acceptance"]["ready"] is True
    assert manifest["coordination_acceptance"]["ready"] is True
    assert manifest["provider_acceptance"]["ready"] is True
    assert manifest["storage_acceptance"]["ready"] is True
    assert manifest["recovery"]["ready"] is True
    assert manifest["resume"]["ready"] is True
    assert manifest["acceptance"]["run_summary"]["tool_call_count"] == 1
    assert manifest["approval_acceptance"]["approval_resume"]["approved_count"] == 1
    assert manifest["context_acceptance"]["context_summary"][
        "prompt_semantic_trimmed_count"
    ] == 1
    assert manifest["orchestration_acceptance"]["capability_discovery"]["counts"][
        "mcp_resource"
    ] == 1
    assert manifest["coordination_acceptance"]["planner"]["status"] == "completed"
    assert manifest["coordination_acceptance"]["handoff"]["decision"]["selected_session"] == "code-reviewer"
    assert manifest["coordination_acceptance"]["agent_tool"]["result"]["status"] == "completed"
    assert manifest["coordination_acceptance"]["trace_eval"]["ok"] is True
    assert manifest["provider_acceptance"]["route_matrix"]["selected"]["vision"] == "vision"
    assert manifest["storage_acceptance"]["preflight"]["ready"] is True
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

    assert "runtime_boundary_audit" in readiness["matched"]["capabilities"]
    assert "approval_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "context_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "orchestration_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "coordination_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "provider_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "storage_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "validation_suite" in readiness["matched"]["capabilities"]
    assert "AgentCoreRuntimeBoundaryReport" in readiness["matched"]["public_api"]
    assert "evaluate_agent_core_runtime_boundary" in readiness["matched"]["public_api"]
    assert "AgentCoreApprovalAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_approval_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreContextAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_context_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreOrchestrationAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_orchestration_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreCoordinationAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_coordination_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_provider_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreStorageAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_storage_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreValidationSuite" in readiness["matched"]["public_api"]
    assert "run_agent_core_validation" in readiness["matched"]["public_api"]
    assert "AgentCoreRuntimeBoundaryReport" in stability["present_stable_api"]
    assert "evaluate_agent_core_runtime_boundary" in stability["present_stable_api"]
    assert "AgentCoreApprovalAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_approval_acceptance" in stability["present_stable_api"]
    assert "AgentCoreContextAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_context_acceptance" in stability["present_stable_api"]
    assert "AgentCoreOrchestrationAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_orchestration_acceptance" in stability["present_stable_api"]
    assert "AgentCoreCoordinationAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_coordination_acceptance" in stability["present_stable_api"]
    assert "AgentCoreProviderAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_provider_acceptance" in stability["present_stable_api"]
    assert "AgentCoreStorageAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_storage_acceptance" in stability["present_stable_api"]
    assert "AgentCoreValidationSuite" in stability["present_stable_api"]
    assert "run_agent_core_validation" in stability["present_stable_api"]


def test_runtime_boundary_audit_reports_current_package_clean() -> None:
    import agent_core

    report = agent_core.evaluate_agent_core_runtime_boundary()
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-runtime-boundary-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["hit_count"] == 0
    assert manifest["scanned_module_count"] > 0


def test_runtime_boundary_audit_blocks_forbidden_runtime_imports(tmp_path) -> None:
    from agent_core.validation import evaluate_agent_core_runtime_boundary

    package = tmp_path / "agent_core"
    package.mkdir()
    (package / "__init__.py").write_text("import openai\n", encoding="utf-8")
    (package / "adapters").mkdir()

    manifest = evaluate_agent_core_runtime_boundary(package_root=package).manifest()

    assert manifest["status"] == "blocked"
    assert manifest["ready"] is False
    assert manifest["hit_count"] == 2
    assert manifest["forbidden_dependency_hits"][0]["name"] == "openai"
    assert manifest["forbidden_package_hits"][0]["name"] == "adapters"
