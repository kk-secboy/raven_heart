from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_trace_eval_suite_runner_evaluates_multiple_cases() -> None:
    import agent_core

    store = agent_core.InMemoryRunTraceStore()
    await store.save(_trace("code-run", provider="code-provider", tool="diff"))
    await store.save(_trace("security-run", provider="security-provider", tool="scan"))

    suite = agent_core.TraceEvalSuite(
        name="portable-regression",
        cases=(
            agent_core.TraceEvalCase(
                name="code-case",
                run_id="code-run",
                spec=agent_core.TraceEvalSpec(
                    name="code-case",
                    expected_status="completed",
                    max_provider_calls=1,
                    required_provider_names=("code-provider",),
                    required_tool_names=("diff",),
                    require_journal_ok=True,
                ),
                tags=("code", "regression"),
            ),
            agent_core.TraceEvalCase(
                name="security-case",
                run_id="security-run",
                spec=agent_core.TraceEvalSpec(
                    name="security-case",
                    expected_status="completed",
                    max_provider_calls=1,
                    required_provider_names=("security-provider",),
                    required_tool_names=("scan",),
                    require_journal_ok=True,
                ),
                tags=("security", "regression"),
            ),
        ),
    )

    report = await agent_core.TraceEvalSuiteRunner(trace_store=store).evaluate(
        suite,
        tags=("regression",),
        metadata={"test": "suite"},
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-trace-eval-suite-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["case_count"] == 2
    assert manifest["status_counts"] == {"ready": 2}
    assert [case["case"]["name"] for case in manifest["case_reports"]] == [
        "code-case",
        "security-case",
    ]
    assert manifest["metadata"]["test"] == "suite"


@pytest.mark.asyncio
async def test_trace_eval_suite_runner_reports_missing_trace() -> None:
    import agent_core

    suite = agent_core.TraceEvalSuite(
        name="missing-regression",
        cases=(
            agent_core.TraceEvalCase(
                name="missing-case",
                run_id="missing-run",
                spec=agent_core.TraceEvalSpec(name="missing-case"),
            ),
        ),
    )

    report = await agent_core.TraceEvalSuiteRunner(
        trace_store=agent_core.InMemoryRunTraceStore()
    ).evaluate(suite)
    manifest = report.manifest()
    issue_codes = {issue["code"] for issue in manifest["issues"]}

    assert manifest["status"] == "blocked"
    assert manifest["ready"] is False
    assert manifest["status_counts"] == {"missing": 1}
    assert "trace_missing" in issue_codes
    assert manifest["case_reports"][0]["status"] == "missing"


@pytest.mark.asyncio
async def test_agent_core_eval_suite_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_eval_suite_acceptance(
        metadata={"test": "eval_suite_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-eval-suite-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["suite_report"]["case_count"] == 2
    assert manifest["suite_report"]["status_counts"] == {"ready": 2}
    assert manifest["missing_case_report"]["status_counts"] == {"missing": 1}


def test_eval_suite_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "eval_suite_runner" in readiness["matched"]["capabilities"]
    assert "TraceEvalCase" in readiness["matched"]["public_api"]
    assert "TraceEvalSuite" in readiness["matched"]["public_api"]
    assert "TraceEvalSuiteRunner" in readiness["matched"]["public_api"]
    assert "AgentCoreEvalSuiteAcceptanceHarness" in readiness["matched"]["public_api"]
    assert "AgentCoreEvalSuiteAcceptanceReport" in readiness["matched"]["public_api"]
    assert "run_agent_core_eval_suite_acceptance" in readiness["matched"]["public_api"]
    assert "TraceEvalCase" in stability["present_stable_api"]
    assert "TraceEvalSuite" in stability["present_stable_api"]
    assert "TraceEvalSuiteRunner" in stability["present_stable_api"]
    assert "AgentCoreEvalSuiteAcceptanceHarness" in stability["present_stable_api"]
    assert "AgentCoreEvalSuiteAcceptanceReport" in stability["present_stable_api"]
    assert "run_agent_core_eval_suite_acceptance" in stability["present_stable_api"]


def _trace(run_id: str, *, provider: str, tool: str) -> dict[str, object]:
    return {
        "schema_version": "agent-core-run-trace-bundle/v1",
        "run": {
            "run_id": run_id,
            "status": "completed",
            "iterations": 1,
            "output_bytes": 12,
        },
        "summary": {
            "status": "completed",
            "journal_ok": True,
            "journal_event_count": 2,
            "provider_call_count": 1,
            "tool_replay_record_count": 1,
        },
        "journal_replay": {
            "ok": True,
            "events": [
                {
                    "sequence": 1,
                    "event_type": "run_started",
                    "run_id": run_id,
                    "payload": {"run_id": run_id},
                },
                {
                    "sequence": 2,
                    "event_type": "run_finished",
                    "run_id": run_id,
                    "payload": {"run_id": run_id, "status": "completed"},
                },
            ],
        },
        "provider": {
            "call_count": 1,
            "calls": [
                {
                    "provider_name": provider,
                    "model": f"{provider}-mini",
                    "status": "completed",
                    "usage": {"total_tokens": 8, "cost_usd": 0.0},
                    "metadata": {},
                }
            ],
        },
        "tool_replay": {
            "record_count": 1,
            "records": [
                {
                    "invocation": {
                        "tool_name": tool,
                        "arguments": {},
                    },
                    "result": {
                        "tool_name": tool,
                        "status": "completed",
                        "call_id": f"{run_id}-{tool}",
                        "metadata": {},
                    },
                    "metadata": {},
                }
            ],
        },
    }
