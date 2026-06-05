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
    assert manifest["repository_boundary"]["ready"] is True
    assert manifest["repository_boundary"]["hit_count"] == 0
    assert manifest["repository_boundary"]["scanned_example_count"] == 2
    assert manifest["readiness"]["ready"] is True
    assert manifest["api_lifecycle"]["ready"] is True
    assert manifest["api_stability"]["ready"] is True
    assert manifest["acceptance"]["ready"] is True
    assert manifest["approval_acceptance"]["ready"] is True
    assert manifest["budget_acceptance"]["ready"] is True
    assert manifest["capability_governance_acceptance"]["ready"] is True
    assert manifest["context_acceptance"]["ready"] is True
    assert manifest["context_window_acceptance"]["ready"] is True
    assert manifest["orchestration_acceptance"]["ready"] is True
    assert manifest["concurrency_acceptance"]["ready"] is True
    assert manifest["coordination_acceptance"]["ready"] is True
    assert manifest["durable_session_acceptance"]["ready"] is True
    assert manifest["event_acceptance"]["ready"] is True
    assert manifest["external_backend_acceptance"]["ready"] is True
    assert manifest["guardrail_acceptance"]["ready"] is True
    assert manifest["interaction_acceptance"]["ready"] is True
    assert manifest["interrupt_acceptance"]["ready"] is True
    assert manifest["lifecycle_acceptance"]["ready"] is True
    assert manifest["native_tool_acceptance"]["ready"] is True
    assert manifest["packaging_acceptance"]["ready"] is True
    assert manifest["packaging_acceptance"]["public_api"]["public_api_count"] <= 40
    assert manifest["packaging_acceptance"]["public_api"]["contract_api_count"] <= 90
    assert manifest["packaging_acceptance"]["public_api"]["public_api_count"] < manifest[
        "packaging_acceptance"
    ]["public_api"]["root_export_count"]
    assert manifest["provider_acceptance"]["ready"] is True
    assert manifest["provider_conformance"]["ready"] is True
    assert manifest["provider_resilience_acceptance"]["ready"] is True
    assert manifest["redaction_acceptance"]["ready"] is True
    assert manifest["trace_export_acceptance"]["ready"] is True
    assert manifest["eval_suite_acceptance"]["ready"] is True
    assert manifest["state_bundle_acceptance"]["ready"] is True
    assert manifest["storage_acceptance"]["ready"] is True
    assert manifest["task_profile_acceptance"]["ready"] is True
    assert manifest["recovery"]["ready"] is True
    assert manifest["resume"]["ready"] is True
    assert manifest["trace_replay_acceptance"]["ready"] is True
    assert manifest["acceptance"]["run_summary"]["tool_call_count"] == 1
    assert manifest["api_lifecycle"]["policy"][
        "allow_mvp_breaking_changes_before_1"
    ] is True
    assert manifest["api_lifecycle"]["unknown_public_api_count"] == 0
    assert manifest["approval_acceptance"]["approval_resume"]["approved_count"] == 1
    assert manifest["budget_acceptance"]["estimated_cost_block"]["provider_request_count"] == 0
    assert manifest["budget_acceptance"]["override_success"]["center"]["usage"]["cost_usd"] == 0.01
    assert manifest["budget_acceptance"]["trace_eval"]["ok"] is True
    assert manifest["capability_governance_acceptance"]["tool_governance"][
        "visible_tool_names"
    ] == ["mcp__fs__read_file", "safe_lookup"]
    assert manifest["capability_governance_acceptance"]["mcp_governance"][
        "inventory_statuses"
    ] == {"broken": "failed", "disabled": "disabled", "fs": "refreshed"}
    assert (
        "danger-admin"
        not in manifest["capability_governance_acceptance"]["skill_governance"][
            "auto_selected"
        ]
    )
    assert manifest["context_acceptance"]["context_summary"][
        "prompt_semantic_trimmed_count"
    ] == 1
    assert manifest["context_window_acceptance"]["ready_report"]["trimmed_count"] == 2
    assert manifest["context_window_acceptance"]["blocked_report"]["status"] == "blocked"
    assert manifest["orchestration_acceptance"]["capability_discovery"]["counts"][
        "mcp_resource"
    ] == 1
    assert manifest["concurrency_acceptance"]["active_snapshot"]["active_run_count"] == 2
    assert manifest["concurrency_acceptance"]["tools"]["alpha_arguments"] == [
        {"owner": "alpha"}
    ]
    assert manifest["concurrency_acceptance"]["event_streams"]["beta"]["session_names"] == [
        "beta"
    ]
    assert manifest["concurrency_acceptance"]["traces"]["record_count"] == 2
    assert manifest["concurrency_acceptance"]["trace_evals"]["alpha"]["ok"] is True
    assert manifest["coordination_acceptance"]["planner"]["status"] == "completed"
    assert manifest["coordination_acceptance"]["handoff"]["decision"]["selected_session"] == "code-reviewer"
    assert manifest["coordination_acceptance"]["agent_tool"]["result"]["status"] == "completed"
    assert manifest["coordination_acceptance"]["trace_eval"]["ok"] is True
    assert manifest["durable_session_acceptance"]["sqlite_roundtrip"]["reopened"][
        "manager_completed_count"
    ] == 1
    assert manifest["durable_session_acceptance"]["markdown_roundtrip"]["reopened"][
        "context_material_count"
    ] == 1
    assert manifest["durable_session_acceptance"]["trace_eval"]["sqlite"]["ok"] is True
    assert manifest["durable_session_acceptance"]["trace_eval"]["markdown"]["ok"] is True
    assert manifest["event_acceptance"]["streaming_run"]["model_stream_event_count"] >= 1
    assert manifest["event_acceptance"]["manager_stream"]["after"]["terminal"] is True
    assert manifest["event_acceptance"]["trace_eval"]["ok"] is True
    assert set(manifest["external_backend_acceptance"]["backend_matrix"]["builtin_kinds"]) == {
        "sqlite",
        "markdown",
    }
    assert set(manifest["external_backend_acceptance"]["backend_matrix"]["external_kinds"]) == {
        "postgres",
        "vector",
        "graph",
        "product",
        "object_storage",
        "custom",
    }
    assert manifest["external_backend_acceptance"]["external_memory"]["call_count"] == 4
    assert set(manifest["external_backend_acceptance"]["external_context_material"]["kinds"]) == {
        "vector",
        "object_storage",
    }
    assert manifest["external_backend_acceptance"]["trace_eval"]["ok"] is True
    assert manifest["guardrail_acceptance"]["preflight"]["preflight_status"] == "blocked"
    assert manifest["guardrail_acceptance"]["policy"]["denied_subjects"] == [
        "tool:delete_target"
    ]
    assert manifest["guardrail_acceptance"]["policy"]["executed_tool_count"] == 0
    assert manifest["guardrail_acceptance"]["structured_output"]["repair_count"] == 1
    assert manifest["guardrail_acceptance"]["trace_evals"]["structured_output"]["ok"] is True
    assert manifest["interaction_acceptance"]["run"]["stream_request_count"] == 2
    assert manifest["interaction_acceptance"]["provider_request"]["first_tool_names"] == [
        "lookup"
    ]
    assert manifest["interaction_acceptance"]["structured_output"]["ok_count"] == 1
    assert manifest["interaction_acceptance"]["trace_summary"][
        "provider_tool_result_execution_count"
    ] == 1
    assert manifest["interaction_acceptance"]["trace_eval"]["ok"] is True
    assert manifest["interrupt_acceptance"]["pre_cancel"]["provider_request_count"] == 0
    assert manifest["interrupt_acceptance"]["provider_timeout"]["status"] == "timeout"
    assert manifest["interrupt_acceptance"]["tool_timeout"]["tool_invocation_count"] == 1
    assert manifest["interrupt_acceptance"]["manager_cancel_reset"]["second_status"] == "completed"
    assert manifest["interrupt_acceptance"]["trace_evals"]["tool_timeout"]["ok"] is True
    assert manifest["lifecycle_acceptance"]["queue"]["before"]["queued_run_count"] == 1
    assert manifest["lifecycle_acceptance"]["queued_cancel"][
        "second_provider_request_count"
    ] == 0
    assert manifest["lifecycle_acceptance"]["active_cancel"]["status"] == "cancelled"
    assert manifest["lifecycle_acceptance"]["timeout"]["status"] == "timeout"
    assert manifest["lifecycle_acceptance"]["trace_eval"]["ok"] is True
    assert manifest["native_tool_acceptance"]["provider_request"]["first_tool_names"] == [
        "lookup"
    ]
    assert manifest["native_tool_acceptance"]["tool_runtime"]["invocation_count"] == 1
    assert manifest["native_tool_acceptance"]["trace_summary"][
        "provider_tool_result_execution_count"
    ] == 1
    assert manifest["native_tool_acceptance"]["trace_eval"]["ok"] is True
    assert manifest["packaging_acceptance"]["project_metadata"]["name"] == "raven-heart"
    assert manifest["packaging_acceptance"]["build_metadata"]["includes_agent_core"] is True
    assert manifest["packaging_acceptance"]["public_api"]["stability_ready"] is True
    assert manifest["provider_acceptance"]["route_matrix"]["selected"]["vision"] == "vision"
    assert manifest["provider_conformance"]["tool_calls"]["tool_call_names"] == ["lookup"]
    assert manifest["provider_conformance"]["json_mode"]["parsed_ok"] is True
    assert manifest["provider_resilience_acceptance"]["retry_completion"][
        "fallback_request_count"
    ] == 1
    assert manifest["provider_resilience_acceptance"]["explicit_provider"][
        "fallback_request_count"
    ] == 0
    assert manifest["provider_resilience_acceptance"]["stream_fallback"]["event_types"] == [
        "delta",
        "usage",
        "message_end",
    ]
    assert manifest["provider_resilience_acceptance"]["trace_evals"]["stream_fallback"][
        "ok"
    ] is True
    assert manifest["redaction_acceptance"]["leak_scan"]["leaked_count"] == 0
    assert manifest["redaction_acceptance"]["redaction"]["redacted_count"] >= 5
    assert manifest["trace_export_acceptance"]["leak_scan"]["leaked_count"] == 0
    assert manifest["trace_export_acceptance"]["trace_eval"]["ok"] is True
    assert manifest["trace_replay_acceptance"]["compatibility_compare"]["ok"] is True
    assert manifest["trace_replay_acceptance"]["drift_detection"]["ok"] is False
    assert manifest["trace_replay_acceptance"]["legacy_replay"]["eval_ok"] is True
    assert manifest["eval_suite_acceptance"]["suite_report"]["status_counts"] == {
        "ready": 2
    }
    assert manifest["eval_suite_acceptance"]["missing_case_report"]["status_counts"] == {
        "missing": 1
    }
    assert manifest["state_bundle_acceptance"]["ready_bundle"]["component_count"] == 5
    assert manifest["state_bundle_acceptance"]["leak_scan"]["leaked_count"] == 0
    assert manifest["storage_acceptance"]["preflight"]["ready"] is True
    assert manifest["task_profile_acceptance"]["matrix"]["completed_profiles"] == [
        "code",
        "ops",
        "security",
    ]
    assert manifest["task_profile_acceptance"]["matrix"]["context_enabled_profiles"] == [
        "code",
        "ops",
        "security",
    ]
    assert manifest["recovery"]["provider_recovery"]["fallback_used"] is True
    assert manifest["resume"]["trace_eval"]["ok"] is True


@pytest.mark.asyncio
async def test_agent_core_validation_suite_blocks_missing_stable_api() -> None:
    import agent_core

    sdk_manifest = agent_core.agent_core_sdk_manifest().manifest()
    sdk_manifest["public_api"] = [
        name for name in sdk_manifest["public_api"] if name != "AgentRunner"
    ]
    sdk_manifest["contract_api"] = [
        name for name in sdk_manifest["contract_api"] if name != "AgentRunner"
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
    sdk_manifest = agent_core.agent_core_sdk_manifest().manifest()

    capability_names = set(readiness["matched"]["capabilities"])
    readiness_api = set(readiness["matched"]["public_api"])
    stable_api = set(stability["present_stable_api"])
    expected_capabilities = {
        "approval_acceptance_harness",
        "budget_acceptance_harness",
        "capability_governance_acceptance_harness",
        "context_acceptance_harness",
        "context_window_report",
        "orchestration_acceptance_harness",
        "packaging_acceptance_harness",
        "provider_contract_profile",
        "task_contract_profile",
        "trace_export_bundle",
        "trace_replay_acceptance_harness",
        "validation_suite",
    }
    expected_contract_api = {
        "AgentCoreAPIContract",
        "AgentCoreProviderContractProfile",
        "AgentCoreRuntimeBoundaryReport",
        "AgentCoreSDKManifest",
        "AgentCoreTaskContractProfile",
        "AgentRunner",
        "AgentSession",
        "AgentTaskContract",
        "ApprovalCenter",
        "ContextWindowBuilder",
        "LLMProviderCenter",
        "LLMProviderPort",
        "MemoryCenter",
        "MCPCenter",
        "PromptIR",
        "ReActExecutor",
        "StorageBackendCatalog",
        "ToolCenter",
        "TraceEvalHarness",
        "TraceExportBuilder",
        "TraceReplayHarness",
        "agent_core_api_contract",
        "agent_core_provider_contract_profile",
        "agent_core_task_contract_profile",
        "evaluate_agent_core_api_lifecycle",
        "evaluate_agent_core_readiness",
        "evaluate_agent_core_runtime_boundary",
        "run_agent_core_validation",
    }
    compatibility_only = {
        "AgentCorePackagingAcceptanceHarness",
        "AgentCoreTraceReplayAcceptanceHarness",
        "AgentCoreValidationSuite",
        "MarkdownTimelineStore",
        "SQLiteTimelineStore",
        "run_agent_core_packaging_acceptance",
        "run_agent_core_trace_replay_acceptance",
    }

    assert expected_capabilities <= capability_names
    assert expected_contract_api <= readiness_api
    assert expected_contract_api <= stable_api
    assert compatibility_only.isdisjoint(readiness_api)
    assert compatibility_only.isdisjoint(stable_api)
    assert compatibility_only <= set(agent_core.__all__)
    assert sdk_manifest["public_api_count"] <= 40
    assert sdk_manifest["contract_api_count"] <= 90
    assert sdk_manifest["root_export_count"] > sdk_manifest["contract_api_count"]


def _legacy_validation_suite_wide_api_contract_assertions() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "runtime_boundary_audit" in readiness["matched"]["capabilities"]
    assert "api_lifecycle_policy" in readiness["matched"]["capabilities"]
    assert "approval_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "budget_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "capability_governance_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "context_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "context_window_report" in readiness["matched"]["capabilities"]
    assert "orchestration_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "concurrency_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "coordination_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "durable_session_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "event_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "external_backend_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "guardrail_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "interaction_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "interrupt_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "lifecycle_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "native_tool_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "packaging_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "provider_contract_profile" in readiness["matched"]["capabilities"]
    assert "task_contract_profile" in readiness["matched"]["capabilities"]
    assert "provider_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "provider_conformance_harness" in readiness["matched"]["capabilities"]
    assert "provider_resilience_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "redaction_contracts" in readiness["matched"]["capabilities"]
    assert "trace_export_bundle" in readiness["matched"]["capabilities"]
    assert "trace_replay_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "eval_suite_runner" in readiness["matched"]["capabilities"]
    assert "state_bundle_contracts" in readiness["matched"]["capabilities"]
    assert "storage_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "task_profile_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "validation_suite" in readiness["matched"]["capabilities"]
    assert "AgentCoreRuntimeBoundaryReport" in readiness["matched"]["public_api"]
    assert "evaluate_agent_core_runtime_boundary" in readiness["matched"]["public_api"]
    assert "AgentCoreAPILifecyclePolicy" in readiness["matched"]["public_api"]
    assert "AgentCoreAPILifecycleReport" in readiness["matched"]["public_api"]
    assert "evaluate_agent_core_api_lifecycle" in readiness["matched"]["public_api"]
    assert "AgentCoreApprovalAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_approval_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreBudgetAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_budget_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreCapabilityGovernanceAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_capability_governance_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreContextAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_context_acceptance" in readiness["matched"]["public_api"]
    assert "ContextWindowBuilder" in readiness["matched"]["public_api"]
    assert "ContextWindowReport" in readiness["matched"]["public_api"]
    assert "build_context_window_report" in readiness["matched"]["public_api"]
    assert "TimelineStorePort" in readiness["matched"]["public_api"]
    assert "TimelineStore" in readiness["matched"]["public_api"]
    assert "SQLiteTimelineStore" in readiness["matched"]["public_api"]
    assert "MarkdownTimelineStore" in readiness["matched"]["public_api"]
    assert "AgentCoreContextWindowAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_context_window_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreOrchestrationAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_orchestration_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreConcurrencyAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_concurrency_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreCoordinationAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_coordination_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreDurableSessionAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "AgentCoreDurableSessionAcceptanceReport" in readiness["matched"]["public_api"]
    assert "run_agent_core_durable_session_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreEventAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_event_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreExternalBackendAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_external_backend_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreGuardrailAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_guardrail_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreInteractionAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_interaction_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreInterruptAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_interrupt_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreLifecycleAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_lifecycle_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreNativeToolAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_native_tool_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCorePackagingAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_packaging_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_provider_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderAPIFormat" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderContractProfile" in readiness["matched"]["public_api"]
    assert "agent_core_provider_contract_profile" in readiness["matched"]["public_api"]
    assert "AgentTaskContract" in readiness["matched"]["public_api"]
    assert "AgentCoreTaskContractProfile" in readiness["matched"]["public_api"]
    assert "agent_core_task_contract_profile" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderConformanceHarness" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderConformanceReport" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderConformanceSpec" in readiness["matched"]["public_api"]
    assert "run_agent_core_provider_conformance" in readiness["matched"]["public_api"]
    assert "AgentCoreProviderResilienceAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_provider_resilience_acceptance" in readiness["matched"]["public_api"]
    assert "RedactionPolicy" in readiness["matched"]["public_api"]
    assert "AgentCoreRedactionAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_redaction_acceptance" in readiness["matched"]["public_api"]
    assert "TraceExportBuilder" in readiness["matched"]["public_api"]
    assert "TraceExportBundle" in readiness["matched"]["public_api"]
    assert "export_trace_bundle" in readiness["matched"]["public_api"]
    assert "AgentCoreTraceExportAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_trace_export_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreTraceReplayAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_trace_replay_acceptance" in readiness["matched"]["public_api"]
    assert "TraceEvalSuiteRunner" in readiness["matched"]["public_api"]
    assert "AgentCoreEvalSuiteAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_eval_suite_acceptance" in readiness["matched"]["public_api"]
    assert "AgentStateBundle" in readiness["matched"]["public_api"]
    assert "AgentStateBundleBuilder" in readiness["matched"]["public_api"]
    assert "build_agent_state_bundle" in readiness["matched"]["public_api"]
    assert "AgentCoreStateBundleAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_state_bundle_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreStorageAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_storage_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreTaskProfileAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "run_agent_core_task_profile_acceptance" in readiness["matched"]["public_api"]
    assert "AgentCoreValidationSuite" in readiness["matched"]["public_api"]
    assert "run_agent_core_validation" in readiness["matched"]["public_api"]
    assert "AgentCoreRuntimeBoundaryReport" in stability["present_stable_api"]
    assert "evaluate_agent_core_runtime_boundary" in stability["present_stable_api"]
    assert "AgentCoreAPILifecyclePolicy" in stability["present_stable_api"]
    assert "AgentCoreAPILifecycleReport" in stability["present_stable_api"]
    assert "evaluate_agent_core_api_lifecycle" in stability["present_stable_api"]
    assert "AgentCoreApprovalAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_approval_acceptance" in stability["present_stable_api"]
    assert "AgentCoreBudgetAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_budget_acceptance" in stability["present_stable_api"]
    assert "AgentCoreCapabilityGovernanceAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_capability_governance_acceptance" in stability["present_stable_api"]
    assert "AgentCoreContextAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_context_acceptance" in stability["present_stable_api"]
    assert "ContextWindowBuilder" in stability["present_stable_api"]
    assert "ContextWindowReport" in stability["present_stable_api"]
    assert "build_context_window_report" in stability["present_stable_api"]
    assert "TimelineStorePort" in stability["present_stable_api"]
    assert "TimelineStore" in stability["present_stable_api"]
    assert "SQLiteTimelineStore" in stability["present_stable_api"]
    assert "MarkdownTimelineStore" in stability["present_stable_api"]
    assert "AgentCoreContextWindowAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_context_window_acceptance" in stability["present_stable_api"]
    assert "AgentCoreOrchestrationAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_orchestration_acceptance" in stability["present_stable_api"]
    assert "AgentCoreConcurrencyAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_concurrency_acceptance" in stability["present_stable_api"]
    assert "AgentCoreCoordinationAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_coordination_acceptance" in stability["present_stable_api"]
    assert "AgentCoreDurableSessionAcceptanceHarness" in stability["present_stable_api"]
    assert "AgentCoreDurableSessionAcceptanceReport" in stability["present_stable_api"]
    assert "run_agent_core_durable_session_acceptance" in stability["present_stable_api"]
    assert "AgentCoreEventAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_event_acceptance" in stability["present_stable_api"]
    assert "AgentCoreExternalBackendAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_external_backend_acceptance" in stability["present_stable_api"]
    assert "AgentCoreGuardrailAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_guardrail_acceptance" in stability["present_stable_api"]
    assert "AgentCoreInteractionAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_interaction_acceptance" in stability["present_stable_api"]
    assert "AgentCoreInterruptAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_interrupt_acceptance" in stability["present_stable_api"]
    assert "AgentCoreLifecycleAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_lifecycle_acceptance" in stability["present_stable_api"]
    assert "AgentCoreNativeToolAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_native_tool_acceptance" in stability["present_stable_api"]
    assert "AgentCorePackagingAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_packaging_acceptance" in stability["present_stable_api"]
    assert "AgentCoreProviderAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_provider_acceptance" in stability["present_stable_api"]
    assert "AgentCoreProviderAPIFormat" in stability["present_stable_api"]
    assert "AgentCoreProviderContractProfile" in stability["present_stable_api"]
    assert "agent_core_provider_contract_profile" in stability["present_stable_api"]
    assert "AgentTaskContract" in stability["present_stable_api"]
    assert "AgentCoreTaskContractProfile" in stability["present_stable_api"]
    assert "agent_core_task_contract_profile" in stability["present_stable_api"]
    assert "AgentCoreProviderConformanceHarness" in stability["present_stable_api"]
    assert "AgentCoreProviderConformanceReport" in stability["present_stable_api"]
    assert "AgentCoreProviderConformanceSpec" in stability["present_stable_api"]
    assert "run_agent_core_provider_conformance" in stability["present_stable_api"]
    assert "AgentCoreProviderResilienceAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_provider_resilience_acceptance" in stability["present_stable_api"]
    assert "RedactionPolicy" in stability["present_stable_api"]
    assert "AgentCoreRedactionAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_redaction_acceptance" in stability["present_stable_api"]
    assert "TraceExportBuilder" in stability["present_stable_api"]
    assert "TraceExportBundle" in stability["present_stable_api"]
    assert "export_trace_bundle" in stability["present_stable_api"]
    assert "AgentCoreTraceExportAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_trace_export_acceptance" in stability["present_stable_api"]
    assert "AgentCoreTraceReplayAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_trace_replay_acceptance" in stability["present_stable_api"]
    assert "TraceEvalSuiteRunner" in stability["present_stable_api"]
    assert "AgentCoreEvalSuiteAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_eval_suite_acceptance" in stability["present_stable_api"]
    assert "AgentStateBundle" in stability["present_stable_api"]
    assert "AgentStateBundleBuilder" in stability["present_stable_api"]
    assert "build_agent_state_bundle" in stability["present_stable_api"]
    assert "AgentCoreStateBundleAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_state_bundle_acceptance" in stability["present_stable_api"]
    assert "AgentCoreStorageAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_storage_acceptance" in stability["present_stable_api"]
    assert "AgentCoreTaskProfileAcceptanceHarness" in stability["present_stable_api"]
    assert "run_agent_core_task_profile_acceptance" in stability["present_stable_api"]
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


def test_repository_boundary_audit_blocks_runtime_dirs_and_adapter_examples(tmp_path) -> None:
    from agent_core.validation import evaluate_agent_core_repository_boundary

    (tmp_path / "agent_core").mkdir()
    (tmp_path / "agent_core" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "adapters").mkdir()
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "real_provider.py").write_text(
        "import urllib.request\napi_key = 'secret'\n",
        encoding="utf-8",
    )

    manifest = evaluate_agent_core_repository_boundary(repository_root=tmp_path).manifest()
    issue_names = {
        hit["name"]
        for hit in (
            manifest["forbidden_directory_hits"]
            + manifest["forbidden_example_hits"]
        )
    }

    assert manifest["status"] == "blocked"
    assert manifest["ready"] is False
    assert manifest["hit_count"] == 3
    assert {"adapters", "import urllib", "api_key"} <= issue_names
