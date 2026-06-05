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
    assert manifest["project_metadata"]["forbidden_dependency_hits"] == []
    assert set(manifest["project_metadata"]["optional_dependencies"]) == {"dev"}
    assert manifest["project_metadata"]["typed_classifier"] is True
    assert "agent" in manifest["project_metadata"]["keywords"]
    assert manifest["build_metadata"]["build_backend"] == "setuptools.build_meta"
    assert isinstance(manifest["build_metadata"]["build_backend_available"], bool)
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
    assert manifest["repository_files"]["ci_workflow_exists"] is True
    assert manifest["repository_files"]["ci_runs_pytest"] is True
    assert manifest["repository_files"]["ci_runs_sdk_validation"] is True
    assert manifest["repository_files"]["documentation"]["mojibake_hit_count"] == 0
    assert all(
        item["utf8_valid"]
        for item in manifest["repository_files"]["documentation"]["files"]
    )
    assert manifest["examples"]["example_count"] == 2
    minimal = manifest["examples"]["examples"]["minimal_react.py"]
    memory = manifest["examples"]["examples"]["memory_and_skills.py"]
    assert minimal["imports_agent_core"] is True
    assert memory["has_main_guard"] is True
    assert minimal["run"]["exit_code"] == 0
    assert minimal["run"]["json_valid"] is True
    assert minimal["run"]["summary"]["status"] == "completed"
    assert minimal["run"]["summary"]["tool_calls"] == 1
    assert memory["run"]["exit_code"] == 0
    assert memory["run"]["json_valid"] is True
    assert memory["run"]["summary"]["loaded_skill_count"] == 1
    assert memory["run"]["summary"]["memory_hit_count"] == 2


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


def test_packaging_acceptance_blocks_forbidden_runtime_dependencies() -> None:
    from agent_core.packaging_acceptance import (
        _packaging_acceptance_issues,
        _project_metadata,
    )

    project_metadata = _project_metadata(
        {
            "project": {
                "name": "raven-heart",
                "version": "0.1.0",
                "requires-python": ">=3.11",
                "dependencies": [],
                "optional-dependencies": {
                    "dev": ["pytest", "openai>=1", "fastapi"],
                    "adapter": ["graphiti-core"],
                },
                "classifiers": ["Typing :: Typed"],
            }
        }
    )
    issues = _packaging_acceptance_issues(
        project_metadata=project_metadata,
        build_metadata={
            "build_backend": "setuptools.build_meta",
            "build_backend_available": True,
            "includes_agent_core": True,
            "includes_py_typed": True,
        },
        public_api={
            "stability_ready": True,
            "root_export_count": 1,
            "unique_root_export_count": 1,
            "manifest_root_export_count": 1,
            "public_api_count": 1,
            "manifest_public_api_count": 1,
            "contract_api_count": 1,
            "manifest_contract_api_count": 1,
            "public_api_missing_from_root": [],
            "contract_api_missing_from_root": [],
        },
        repository_files={
            "files": {"README.md": True, "LICENSE": True, "pyproject.toml": True},
            "py_typed_exists": True,
            "ci_workflow_exists": True,
            "ci_runs_pytest": True,
            "ci_runs_sdk_validation": True,
        },
        examples={
            "examples": {
                "minimal_react.py": {
                    "exists": True,
                    "imports_agent_core": True,
                    "has_main_guard": True,
                    "run": {
                        "status": "completed",
                        "exit_code": 0,
                        "json_valid": True,
                    },
                }
            }
        },
    )
    codes = {issue.code for issue in issues}
    hit_names = {hit["name"] for hit in project_metadata["forbidden_dependency_hits"]}

    assert "forbidden_runtime_dependency_declared" in codes
    assert {"openai", "fastapi", "graphiti-core"} <= hit_names


def test_packaging_acceptance_blocks_documentation_mojibake() -> None:
    from agent_core.packaging_acceptance import _packaging_acceptance_issues

    issues = _packaging_acceptance_issues(
        project_metadata={
            "name": "raven-heart",
            "version": "0.1.0",
            "requires_python": ">=3.11",
            "runtime_dependency_count": 0,
            "typed_classifier": True,
            "forbidden_dependency_hits": [],
        },
        build_metadata={
            "build_backend": "setuptools.build_meta",
            "build_backend_available": True,
            "includes_agent_core": True,
            "includes_py_typed": True,
        },
        public_api={
            "stability_ready": True,
            "root_export_count": 1,
            "unique_root_export_count": 1,
            "manifest_root_export_count": 1,
            "public_api_count": 1,
            "manifest_public_api_count": 1,
            "contract_api_count": 1,
            "manifest_contract_api_count": 1,
            "public_api_missing_from_root": [],
            "contract_api_missing_from_root": [],
        },
        repository_files={
            "files": {"README.md": True, "LICENSE": True, "pyproject.toml": True},
            "py_typed_exists": True,
            "ci_workflow_exists": True,
            "ci_runs_pytest": True,
            "ci_runs_sdk_validation": True,
            "documentation": {
                "mojibake_hit_count": 1,
                "mojibake_hits": [
                    {"path": "README.md", "marker": "锛", "line": 7}
                ],
            },
        },
        examples={
            "examples": {
                "minimal_react.py": {
                    "exists": True,
                    "imports_agent_core": True,
                    "has_main_guard": True,
                    "run": {
                        "status": "completed",
                        "exit_code": 0,
                        "json_valid": True,
                    },
                }
            }
        },
    )
    codes = {issue.code for issue in issues}

    assert "documentation_mojibake_detected" in codes
