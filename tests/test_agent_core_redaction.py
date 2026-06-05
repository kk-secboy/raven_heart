from __future__ import annotations

import pytest


def test_redaction_policy_redacts_sensitive_keys_and_secret_values() -> None:
    import agent_core

    payload = {
        "headers": {"Authorization": "Bearer secret-token-123456789"},
        "metadata": {"api_key": "sk-test-secret-value-123456789"},
        "notes": "token=inline-secret-value-123456",
        "safe": "public",
    }
    result = agent_core.redact_payload(payload).manifest()
    rendered = str(result)

    assert result["schema_version"] == "agent-core-redaction-result/v1"
    assert result["redacted"] is True
    assert result["redacted_count"] == 3
    assert "secret-token-123456789" not in rendered
    assert "sk-test-secret-value-123456789" not in rendered
    assert "inline-secret-value-123456" not in rendered
    assert result["payload"]["safe"] == "public"
    assert all(decision["digest"].startswith("sha256:") for decision in result["decisions"])


def test_redaction_policy_redacts_oversized_strings() -> None:
    import agent_core

    result = agent_core.redact_payload(
        {"body": "x" * 32},
        policy=agent_core.RedactionPolicy(max_string_bytes=8),
    ).manifest()

    assert result["redacted_count"] == 1
    assert result["decisions"][0]["reason"] == "string_too_large"
    assert result["payload"]["body"]["redacted"] is True


@pytest.mark.asyncio
async def test_agent_core_redaction_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_redaction_acceptance(
        metadata={"test": "redaction_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-redaction-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["leak_scan"]["leaked_count"] == 0
    assert manifest["leak_scan"]["decision_count"] >= 5
    assert {"sensitive_key", "secret_value"} <= set(
        manifest["leak_scan"]["decision_reasons"]
    )


def test_redaction_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "redaction_contracts" in readiness["matched"]["capabilities"]
    assert "RedactionPolicy" in agent_core.__all__
    assert "RedactionResult" in agent_core.__all__
    assert "AgentCoreRedactionAcceptanceHarness" in agent_core.__all__
    assert "run_agent_core_redaction_acceptance" in agent_core.__all__
    assert "RedactionPolicy" not in readiness["matched"]["public_api"]
    assert "RedactionPolicy" not in stability["present_stable_api"]
