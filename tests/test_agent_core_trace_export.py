from __future__ import annotations

import pytest

from agent_core.evals import DefaultTraceEvaluator, TraceEvalSpec
from agent_core.trace import AgentRunTraceBundle
from agent_core.trace_export import TraceExportBuilder, TraceExportPolicy, export_trace_bundle
from agent_core.trace_export_acceptance import run_agent_core_trace_export_acceptance


def _secret_trace() -> dict[str, object]:
    return AgentRunTraceBundle(
        run_id="export-run",
        status="completed",
        iterations=1,
        output_bytes=2,
        journal_replay={
            "ok": True,
            "event_count": 2,
            "events": [
                {
                    "sequence": 1,
                    "event_type": "run_started",
                    "run_id": "export-run",
                    "payload": {"run_id": "export-run"},
                },
                {
                    "sequence": 2,
                    "event_type": "run_finished",
                    "run_id": "export-run",
                    "payload": {"run_id": "export-run", "status": "completed"},
                },
            ],
        },
        provider={
            "call_count": 1,
            "calls": [
                {
                    "provider_name": "mock",
                    "model": "mock-mini",
                    "status": "completed",
                    "metadata": {
                        "request": {
                            "headers": {
                                "Authorization": "Bearer export-test-token-123456",
                                "X-API-Key": "sk-export-test-secret-1234567890",
                            },
                            "metadata": {"run_id": "export-run"},
                        }
                    },
                }
            ],
        },
        session={
            "name": "sdk-session",
            "metadata": {"operator_password": "password=operator-secret-123456"},
        },
        tool_replay={
            "record_count": 1,
            "records": [
                {
                    "invocation": {
                        "tool_name": "lookup",
                        "arguments": {"secret_note": "token=tool-secret-123456"},
                    },
                    "result": {"status": "completed", "tool_name": "lookup"},
                }
            ],
        },
    ).manifest()


def test_trace_export_bundle_redacts_secrets_and_keeps_eval_shape() -> None:
    bundle = export_trace_bundle({"run_trace": _secret_trace()}).manifest()
    rendered = str(bundle)

    assert bundle["schema_version"] == "agent-core-trace-export-bundle/v1"
    assert bundle["prompt_safe"] is True
    assert bundle["record_count"] == 1
    assert bundle["records"][0]["payload_schema_version"] == "agent-core-run-trace-bundle/v1"
    assert bundle["records"][0]["original_bytes"] > 0
    assert bundle["records"][0]["exported_bytes"] > 0
    assert len(bundle["records"][0]["original_sha256"]) == 64
    assert len(bundle["records"][0]["exported_sha256"]) == 64
    assert bundle["redacted_count"] >= 3
    assert bundle["records"][0]["payload"]["session"]["name"] == "sdk-session"
    assert bundle["records"][0]["payload"]["session"]["metadata"]["operator_password"][
        "redacted"
    ] is True
    assert "sk-export-test-secret-1234567890" not in rendered
    assert "operator-secret-123456" not in rendered
    assert "Bearer export-test-token-123456" not in rendered
    assert "tool-secret-123456" not in rendered

    exported_trace = bundle["records"][0]["payload"]
    report = DefaultTraceEvaluator().evaluate(
        exported_trace,
        TraceEvalSpec(
            expected_status="completed",
            max_provider_calls=1,
            required_event_types=("run_started", "run_finished"),
        ),
    )
    assert report.ok is True


def test_trace_export_builder_accepts_trace_lists() -> None:
    bundle = TraceExportBuilder().export([_secret_trace()]).manifest()

    assert bundle["record_count"] == 1
    assert bundle["records"][0]["name"] == "export-run"
    assert bundle["redaction"]["redacted"] is True


def test_trace_export_policy_can_disable_redaction_for_private_in_process_use() -> None:
    bundle = export_trace_bundle(
        {"run_trace": _secret_trace()},
        policy=TraceExportPolicy(redact=False),
    ).manifest()

    assert bundle["record_count"] == 1
    assert bundle["redacted_count"] == 0
    assert "sk-export-test-secret-1234567890" in str(bundle)


@pytest.mark.asyncio
async def test_agent_core_trace_export_acceptance_gate_passes() -> None:
    report = await run_agent_core_trace_export_acceptance(metadata={"test": "trace_export"})
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-trace-export-acceptance-report/v1"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["leak_scan"]["leaked_count"] == 0
    assert manifest["export_bundle"]["redacted_count"] >= 4
    assert manifest["trace_eval"]["ok"] is True
