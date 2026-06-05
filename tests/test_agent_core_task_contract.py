from __future__ import annotations

import pytest


def test_agent_core_task_contract_profile_declares_runtime_neutral_run_shapes() -> None:
    import agent_core

    profile = agent_core.agent_core_task_contract_profile(
        metadata={"test": "task_contract_profile"}
    )
    manifest = profile.manifest()

    assert manifest["schema_version"] == "agent-core-task-contract-profile/v1"
    assert manifest["contract_names"] == [
        "generic_run_request",
        "context_aware_task",
        "memory_context_task",
        "managed_background_task",
        "tool_structured_task",
    ]
    assert manifest["metadata"] == {"test": "task_contract_profile"}
    assert {"generic", "code", "ops", "security", "custom"} <= set(
        manifest["supported_runtime_profiles"]
    )
    assert "AgentRunRequest" in manifest["required_core_contracts"]
    assert "AgentRunPreflightRequirements" in manifest["required_core_contracts"]
    assert "openai" in manifest["forbidden_core_dependencies"]
    assert "fastapi" in manifest["forbidden_core_dependencies"]

    memory_contract = profile.contract("memory_context_task")
    requirements = memory_contract.preflight_requirements(
        metadata={"request_id": "task-1"}
    ).manifest()
    assert requirements["require_memory"] is True
    assert requirements["metadata"]["contract_name"] == "memory_context_task"
    assert requirements["metadata"]["request_id"] == "task-1"
    storage_roles = {
        requirement["role"]: requirement
        for requirement in requirements["storage_backend_requirements"]
    }
    assert set(storage_roles) == {"memory", "context_material"}
    assert {"sqlite", "markdown", "postgres", "vector", "graph", "custom"} <= set(
        storage_roles["memory"]["allowed_kinds"]
    )
    assert storage_roles["memory"]["require_queryable"] is True


@pytest.mark.asyncio
async def test_agent_runner_applies_task_contract_preflight_before_provider_call() -> None:
    import agent_core

    class ProviderShouldNotRun:
        async def complete(self, request):  # pragma: no cover - defensive assertion
            raise AssertionError("provider must not run when task contract preflight blocks")

    contract = agent_core.AgentTaskContract(
        name="requires-review-tool",
        required_tools=("review_diff",),
        require_memory=True,
        metadata={"owner": "sdk-test"},
    )
    session = agent_core.AgentSession(
        profile=agent_core.AgentProfile(
            name="contract-gated",
            model="deterministic",
            capabilities=agent_core.CapabilitySet(memory_enabled=False),
        ),
        provider=ProviderShouldNotRun(),
        tools=agent_core.ToolRegistry(),
    )

    outcome = await agent_core.AgentRunner(session).run(
        agent_core.AgentRunRequest(
            task="review a parser diff",
            task_contract=contract,
            metadata={"case": "contract-preflight"},
        )
    )

    preflight = outcome.preflight_manifest
    assert outcome.result.status == "denied"
    assert preflight["status"] == "blocked"
    assert preflight["blocking_codes"] == ["memory_required", "missing_tool"]
    assert preflight["request"]["requirements"]["metadata"]["contract_name"] == (
        "requires-review-tool"
    )
    assert preflight["request"]["metadata"]["task_contract"]["name"] == "requires-review-tool"
    assert preflight["request"]["metadata"]["request_metadata"] == {
        "case": "contract-preflight"
    }


def test_task_contract_profile_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    sdk_manifest = agent_core.agent_core_sdk_manifest().manifest()
    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()
    capability_names = {capability["name"] for capability in sdk_manifest["capabilities"]}

    assert "task_contract_profile" in capability_names
    assert "task_contract_profile" in readiness["matched"]["capabilities"]
    assert "AgentTaskContract" in readiness["matched"]["public_api"]
    assert "AgentCoreTaskContractProfile" in readiness["matched"]["public_api"]
    assert "agent_core_task_contract_profile" in readiness["matched"]["public_api"]
    assert "AgentTaskContract" in stability["present_stable_api"]
    assert "AgentCoreTaskContractProfile" in stability["present_stable_api"]
    assert "agent_core_task_contract_profile" in stability["present_stable_api"]
