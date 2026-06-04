from __future__ import annotations

import pytest

from agent_core.state_bundle import (
    AgentStateBundleBuilder,
    AgentStateBundlePolicy,
    build_agent_state_bundle,
)
from agent_core.state_bundle_acceptance import run_agent_core_state_bundle_acceptance


def _components() -> dict[str, dict[str, object]]:
    return {
        "journal": {
            "schema_version": "agent-core-journal/v1",
            "backend": {"role": "journal", "kind": "sqlite"},
            "runs": [{"run_id": "run-1", "status": "completed"}],
            "metadata": {"api_key": "sk-state-test-secret-1234567890"},
        },
        "run_state": {
            "schema_version": "agent-core-run-store/v1",
            "backend": {"role": "run_state", "kind": "markdown"},
            "runs": [{"run_id": "run-1", "status": "completed"}],
        },
        "memory": {
            "schema_version": "agent-core-memory-store/v1",
            "backend": {"role": "memory", "kind": "external"},
            "metadata": {"authorization": "Bearer state-test-token-123456"},
        },
        "trace": {
            "schema_version": "agent-core-run-trace-bundle/v1",
            "backend": {"role": "run_trace", "kind": "markdown"},
            "run": {"run_id": "run-1", "status": "completed"},
        },
    }


def test_agent_state_bundle_redacts_and_orders_components() -> None:
    bundle = build_agent_state_bundle(
        _components(),
        policy=AgentStateBundlePolicy(required_roles=("journal", "run_state", "memory")),
    ).manifest()
    rendered = str(bundle)

    assert bundle["schema_version"] == "agent-core-state-bundle/v1"
    assert bundle["ready"] is True
    assert bundle["component_count"] == 4
    assert bundle["roles"]["journal"] == 1
    assert bundle["roles"]["memory"] == 1
    assert bundle["kinds"]["sqlite"] == 1
    assert bundle["redacted_count"] >= 2
    assert "sk-state-test-secret-1234567890" not in rendered
    assert "state-test-token-123456" not in rendered
    assert [step["role"] for step in bundle["restore_plan"]["steps"][:3]] == [
        "journal",
        "run_state",
        "memory",
    ]
    assert len(bundle["components"][0]["exported_sha256"]) == 64


def test_agent_state_bundle_blocks_missing_required_roles() -> None:
    bundle = AgentStateBundleBuilder(
        policy=AgentStateBundlePolicy(required_roles=("journal", "artifact"))
    ).build({"journal": _components()["journal"]}).manifest()
    issue_codes = {issue["code"] for issue in bundle["issues"]}

    assert bundle["status"] == "blocked"
    assert bundle["ready"] is False
    assert "required_state_role_missing" in issue_codes


@pytest.mark.asyncio
async def test_agent_core_state_bundle_acceptance_gate_passes() -> None:
    report = await run_agent_core_state_bundle_acceptance(metadata={"test": "state_bundle"})
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-state-bundle-acceptance-report/v1"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["ready_bundle"]["ready"] is True
    assert manifest["ready_bundle"]["component_count"] == 5
    assert manifest["blocked_bundle"]["status"] == "blocked"
    assert manifest["leak_scan"]["leaked_count"] == 0
