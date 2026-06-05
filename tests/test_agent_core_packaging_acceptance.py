from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_agent_core_packaging_acceptance_gate_passes() -> None:
    import agent_core

    report = await agent_core.run_agent_core_packaging_acceptance(
        metadata={"test": "packaging_acceptance"}
    )
    manifest = report.manifest()

    assert manifest["schema_version"] == "agent-core-packaging-acceptance-report/v1"
    assert manifest["status"] == "ready"
    assert manifest["ready"] is True
    assert manifest["error_count"] == 0
    assert manifest["project_metadata"]["name"] == "raven-heart"
    assert manifest["project_metadata"]["requires_python"] == ">=3.11"
    assert manifest["project_metadata"]["runtime_dependency_count"] == 0
    assert manifest["project_metadata"]["typed_classifier"] is True
    assert "agent" in manifest["project_metadata"]["keywords"]
    assert manifest["build_metadata"]["build_backend"] == "setuptools.build_meta"
    assert manifest["build_metadata"]["includes_agent_core"] is True
    assert manifest["build_metadata"]["includes_py_typed"] is True
    assert manifest["public_api"]["stability_ready"] is True
    assert manifest["public_api"]["root_export_count"] == manifest["public_api"][
        "unique_root_export_count"
    ]
    assert manifest["public_api"]["root_export_count"] == manifest["public_api"][
        "manifest_root_export_count"
    ]
    assert manifest["public_api"]["public_api_count"] == manifest["public_api"][
        "manifest_public_api_count"
    ]
    assert manifest["public_api"]["contract_api_count"] == manifest["public_api"][
        "manifest_contract_api_count"
    ]
    assert manifest["public_api"]["public_api_count"] <= 40
    assert manifest["public_api"]["public_api_count"] < manifest["public_api"][
        "root_export_count"
    ]
    assert manifest["public_api"]["contract_api_count"] >= manifest["public_api"][
        "public_api_count"
    ]
    assert manifest["public_api"]["contract_api_count"] <= 90
    assert manifest["public_api"]["public_api_in_root_exports"] is True
    assert manifest["public_api"]["contract_api_in_root_exports"] is True
    assert all(manifest["public_api"]["sample_imports"].values())
    assert all(manifest["repository_files"]["files"].values())
    assert manifest["repository_files"]["py_typed_exists"] is True
    assert manifest["examples"]["example_count"] == 3
    assert manifest["examples"]["examples"]["minimal_react.py"]["imports_agent_core"] is True
    assert manifest["examples"]["examples"]["memory_and_skills.py"]["has_main_guard"] is True
    assert (
        manifest["examples"]["examples"]["real_provider_conformance.py"][
            "imports_agent_core"
        ]
        is True
    )


def test_packaging_acceptance_is_declared_in_readiness_and_api_contract() -> None:
    import agent_core

    readiness = agent_core.evaluate_agent_core_readiness().manifest()
    stability = agent_core.evaluate_agent_core_api_stability().manifest()

    assert "packaging_acceptance_harness" in readiness["matched"]["capabilities"]
    assert "AgentCorePackagingAcceptanceHarness" in agent_core.__all__
    assert "AgentCorePackagingAcceptanceReport" in agent_core.__all__
    assert "run_agent_core_packaging_acceptance" in agent_core.__all__
    assert "AgentCorePackagingAcceptanceHarness" not in readiness["matched"]["public_api"]
    assert "AgentCorePackagingAcceptanceHarness" not in stability["present_stable_api"]
