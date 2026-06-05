"""SDK-level package metadata and public import acceptance checks."""

from __future__ import annotations

import importlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class AgentCorePackagingAcceptanceIssue:
    """One blocking package-readiness acceptance issue."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-packaging-acceptance-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCorePackagingAcceptanceReport:
    """Prompt-safe report proving the SDK can be packaged and imported."""

    status: str
    project_metadata: dict[str, Any] = field(default_factory=dict)
    build_metadata: dict[str, Any] = field(default_factory=dict)
    public_api: dict[str, Any] = field(default_factory=dict)
    repository_files: dict[str, Any] = field(default_factory=dict)
    examples: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCorePackagingAcceptanceIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-packaging-acceptance-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "project_metadata": dict(self.project_metadata),
            "build_metadata": dict(self.build_metadata),
            "public_api": dict(self.public_api),
            "repository_files": dict(self.repository_files),
            "examples": dict(self.examples),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCorePackagingAcceptanceHarness:
    """Run deterministic package metadata and public import checks."""

    project_root: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(self) -> AgentCorePackagingAcceptanceReport:
        root = Path(self.project_root) if self.project_root is not None else _project_root()
        pyproject = _load_pyproject(root)
        project_metadata = _project_metadata(pyproject)
        build_metadata = _build_metadata(pyproject)
        public_api = _public_api_manifest()
        repository_files = _repository_files(root)
        examples = _examples_manifest(root)
        issues = _packaging_acceptance_issues(
            project_metadata=project_metadata,
            build_metadata=build_metadata,
            public_api=public_api,
            repository_files=repository_files,
            examples=examples,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return AgentCorePackagingAcceptanceReport(
            status=status,
            project_metadata=project_metadata,
            build_metadata=build_metadata,
            public_api=public_api,
            repository_files=repository_files,
            examples=examples,
            issues=issues,
            metadata={
                "scenario": "agent_core_packaging_acceptance",
                "project_root": str(root),
                **dict(self.metadata),
            },
        )


async def run_agent_core_packaging_acceptance(
    *,
    project_root: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentCorePackagingAcceptanceReport:
    """Run the default package metadata and public import acceptance checks."""

    return await AgentCorePackagingAcceptanceHarness(
        project_root=project_root,
        metadata=dict(metadata or {}),
    ).run()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_pyproject(root: Path) -> dict[str, Any]:
    path = root / "pyproject.toml"
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _project_metadata(pyproject: dict[str, Any]) -> dict[str, Any]:
    project = pyproject.get("project") if isinstance(pyproject.get("project"), dict) else {}
    optional = (
        project.get("optional-dependencies")
        if isinstance(project.get("optional-dependencies"), dict)
        else {}
    )
    dependencies = project.get("dependencies") if isinstance(project.get("dependencies"), list) else []
    classifiers = project.get("classifiers") if isinstance(project.get("classifiers"), list) else []
    keywords = project.get("keywords") if isinstance(project.get("keywords"), list) else []
    return {
        "schema_version": "agent-core-package-project-metadata/v1",
        "name": str(project.get("name") or ""),
        "version": str(project.get("version") or ""),
        "description_bytes": len(str(project.get("description") or "").encode("utf-8")),
        "readme": str(project.get("readme") or ""),
        "requires_python": str(project.get("requires-python") or ""),
        "runtime_dependency_count": len(dependencies),
        "runtime_dependencies": [str(item) for item in dependencies],
        "optional_dependency_groups": sorted(str(key) for key in optional),
        "dev_dependency_count": len(optional.get("dev") or ()),
        "classifiers": [str(item) for item in classifiers],
        "keywords": [str(item) for item in keywords],
        "typed_classifier": "Typing :: Typed" in {str(item) for item in classifiers},
    }


def _build_metadata(pyproject: dict[str, Any]) -> dict[str, Any]:
    build = (
        pyproject.get("build-system")
        if isinstance(pyproject.get("build-system"), dict)
        else {}
    )
    tool = pyproject.get("tool") if isinstance(pyproject.get("tool"), dict) else {}
    setuptools = tool.get("setuptools") if isinstance(tool.get("setuptools"), dict) else {}
    packages = (
        setuptools.get("packages")
        if isinstance(setuptools.get("packages"), dict)
        else {}
    )
    find = packages.get("find") if isinstance(packages.get("find"), dict) else {}
    include = find.get("include") if isinstance(find.get("include"), list) else []
    package_data = (
        setuptools.get("package-data")
        if isinstance(setuptools.get("package-data"), dict)
        else {}
    )
    agent_core_data = (
        package_data.get("agent_core")
        if isinstance(package_data.get("agent_core"), list)
        else []
    )
    requires = build.get("requires") if isinstance(build.get("requires"), list) else []
    return {
        "schema_version": "agent-core-package-build-metadata/v1",
        "build_backend": str(build.get("build-backend") or ""),
        "build_requires": [str(item) for item in requires],
        "package_find_include": [str(item) for item in include],
        "includes_agent_core": any(str(item) == "agent_core*" for item in include),
        "agent_core_package_data": [str(item) for item in agent_core_data],
        "includes_py_typed": "py.typed" in {str(item) for item in agent_core_data},
    }


def _public_api_manifest() -> dict[str, Any]:
    package = importlib.import_module("agent_core")
    exported = tuple(str(item) for item in getattr(package, "__all__", ()))
    exported_set = set(exported)
    sdk_manifest = package.agent_core_sdk_manifest().manifest()
    stability = package.evaluate_agent_core_api_stability().manifest()
    sample_names = (
        "AgentRunner",
        "AgentSession",
        "LLMProviderCenter",
        "ToolRegistry",
        "MemoryCenter",
        "TraceEvalHarness",
        "run_agent_core_validation",
    )
    public_api = tuple(str(item) for item in sdk_manifest.get("public_api", ()))
    contract_api = tuple(str(item) for item in sdk_manifest.get("contract_api", ()))
    manifest_root_export_count = int(sdk_manifest.get("root_export_count") or 0)
    return {
        "schema_version": "agent-core-package-public-api-smoke/v1",
        "root_export_count": len(exported),
        "unique_root_export_count": len(set(exported)),
        "manifest_root_export_count": manifest_root_export_count,
        "public_api_count": len(public_api),
        "manifest_public_api_count": int(sdk_manifest.get("public_api_count") or 0),
        "contract_api_count": len(contract_api),
        "manifest_contract_api_count": int(sdk_manifest.get("contract_api_count") or 0),
        "public_api_names": list(public_api),
        "public_api_missing_from_root": sorted(set(public_api) - exported_set),
        "contract_api_missing_from_root": sorted(set(contract_api) - exported_set),
        "public_api_in_root_exports": set(public_api) <= exported_set,
        "contract_api_in_root_exports": set(contract_api) <= exported_set,
        "stability_ready": bool(stability.get("ready")),
        "missing_stable_api": list(stability.get("missing_stable_api") or ()),
        "sample_imports": {
            name: getattr(package, name, None) is not None for name in sample_names
        },
    }


def _repository_files(root: Path) -> dict[str, Any]:
    files = {
        name: root.joinpath(name).exists()
        for name in ("README.md", "LICENSE", "pyproject.toml")
    }
    typed_marker = root / "agent_core" / "py.typed"
    return {
        "schema_version": "agent-core-package-repository-files/v1",
        "files": files,
        "readme_bytes": _file_size(root / "README.md"),
        "license_bytes": _file_size(root / "LICENSE"),
        "py_typed_exists": typed_marker.exists(),
        "py_typed_bytes": _file_size(typed_marker),
    }


def _examples_manifest(root: Path) -> dict[str, Any]:
    expected = ("minimal_react.py", "memory_and_skills.py")
    example_root = root / "examples"
    examples = {}
    for name in expected:
        path = example_root / name
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        examples[name] = {
            "exists": path.exists(),
            "bytes": len(text.encode("utf-8")),
            "imports_agent_core": "agent_core" in text,
            "has_main_guard": 'if __name__ == "__main__"' in text,
        }
    return {
        "schema_version": "agent-core-package-examples-smoke/v1",
        "example_count": len([item for item in examples.values() if item["exists"]]),
        "examples": examples,
    }


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _packaging_acceptance_issues(
    *,
    project_metadata: dict[str, Any],
    build_metadata: dict[str, Any],
    public_api: dict[str, Any],
    repository_files: dict[str, Any],
    examples: dict[str, Any],
) -> tuple[AgentCorePackagingAcceptanceIssue, ...]:
    issues: list[AgentCorePackagingAcceptanceIssue] = []
    if project_metadata.get("name") != "raven-heart":
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="project_metadata",
                code="package_name_unexpected",
                message="Package name must remain raven-heart.",
                metadata=dict(project_metadata),
            )
        )
    if not str(project_metadata.get("version") or ""):
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="project_metadata",
                code="package_version_missing",
                message="Package version is missing.",
            )
        )
    if str(project_metadata.get("requires_python") or "") != ">=3.11":
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="project_metadata",
                code="requires_python_unexpected",
                message="Package must declare Python 3.11+ support.",
                metadata=dict(project_metadata),
            )
        )
    if int(project_metadata.get("runtime_dependency_count") or 0) != 0:
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="project_metadata",
                code="runtime_dependencies_present",
                message="SDK package should keep runtime dependencies out of core.",
                metadata=dict(project_metadata),
            )
        )
    if project_metadata.get("typed_classifier") is not True:
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="project_metadata",
                code="typed_classifier_missing",
                message="Package metadata must declare Typing :: Typed.",
                metadata=dict(project_metadata),
            )
        )
    if build_metadata.get("build_backend") != "setuptools.build_meta":
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="build_metadata",
                code="build_backend_missing",
                message="Package build backend must be explicit.",
                metadata=dict(build_metadata),
            )
        )
    if build_metadata.get("includes_agent_core") is not True:
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="build_metadata",
                code="agent_core_package_not_included",
                message="Package discovery must include agent_core.",
                metadata=dict(build_metadata),
            )
        )
    if build_metadata.get("includes_py_typed") is not True:
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="build_metadata",
                code="py_typed_not_packaged",
                message="Package data must include agent_core/py.typed.",
                metadata=dict(build_metadata),
            )
        )
    if public_api.get("stability_ready") is not True:
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="public_api",
                code="stable_public_api_missing",
                message="Package root is missing stable public API names.",
                metadata=dict(public_api),
            )
        )
    if public_api.get("root_export_count") != public_api.get("unique_root_export_count"):
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="public_api",
                code="duplicate_public_exports",
                message="Package root has duplicate public API exports.",
                metadata=dict(public_api),
            )
        )
    if public_api.get("root_export_count") != public_api.get("manifest_root_export_count"):
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="public_api",
                code="manifest_root_export_count_mismatch",
                message="Package manifest root export count does not match __all__.",
                metadata=dict(public_api),
            )
        )
    if public_api.get("public_api_count") != public_api.get("manifest_public_api_count"):
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="public_api",
                code="manifest_public_api_count_mismatch",
                message="Package manifest public API count is inconsistent.",
                metadata=dict(public_api),
            )
        )
    if public_api.get("contract_api_count") != public_api.get("manifest_contract_api_count"):
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="public_api",
                code="manifest_contract_api_count_mismatch",
                message="Package manifest contract API count is inconsistent.",
                metadata=dict(public_api),
            )
        )
    if int(public_api.get("public_api_count") or 0) > 40:
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="public_api",
                code="root_public_api_too_large",
                message="Root public API must stay small; compatibility exports are tracked separately.",
                metadata=dict(public_api),
            )
        )
    if int(public_api.get("contract_api_count") or 0) > 90:
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="public_api",
                code="contract_api_too_large",
                message="Stable contract API must stay focused on runtime integration contracts.",
                metadata=dict(public_api),
            )
        )
    if public_api.get("public_api_missing_from_root"):
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="public_api",
                code="public_api_missing_from_root",
                message="Manifest root public API names must remain importable from agent_core.",
                metadata=dict(public_api),
            )
        )
    if public_api.get("contract_api_missing_from_root"):
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="public_api",
                code="contract_api_missing_from_root",
                message="Stable contract API names must remain importable from agent_core during v0.x compatibility.",
                metadata=dict(public_api),
            )
        )
    if not all((repository_files.get("files") or {}).values()):
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="repository_files",
                code="repository_file_missing",
                message="README, LICENSE, and pyproject.toml must be present.",
                metadata=dict(repository_files),
            )
        )
    if repository_files.get("py_typed_exists") is not True:
        issues.append(
            AgentCorePackagingAcceptanceIssue(
                source="repository_files",
                code="py_typed_missing",
                message="agent_core/py.typed must exist for typed SDK consumers.",
                metadata=dict(repository_files),
            )
        )
    for name, info in (examples.get("examples") or {}).items():
        if not info.get("exists") or not info.get("imports_agent_core") or not info.get("has_main_guard"):
            issues.append(
                AgentCorePackagingAcceptanceIssue(
                    source="examples",
                    code="example_smoke_missing",
                    message=f"Example {name} is missing required SDK smoke structure.",
                    metadata={"name": name, **dict(info)},
                )
            )
    return tuple(issues)
