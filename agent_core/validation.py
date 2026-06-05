"""Aggregate SDK validation suite for host-runtime migration gates."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_core.acceptance import run_agent_core_acceptance
from agent_core.approval_acceptance import run_agent_core_approval_acceptance
from agent_core.budget_acceptance import run_agent_core_budget_acceptance
from agent_core.capability_governance_acceptance import (
    run_agent_core_capability_governance_acceptance,
)
from agent_core.context_acceptance import run_agent_core_context_acceptance
from agent_core.context_window_acceptance import run_agent_core_context_window_acceptance
from agent_core.concurrency_acceptance import run_agent_core_concurrency_acceptance
from agent_core.coordination_acceptance import run_agent_core_coordination_acceptance
from agent_core.durable_session_acceptance import run_agent_core_durable_session_acceptance
from agent_core.event_acceptance import run_agent_core_event_acceptance
from agent_core.eval_suite import run_agent_core_eval_suite_acceptance
from agent_core.external_backend_acceptance import (
    run_agent_core_external_backend_acceptance,
)
from agent_core.guardrail_acceptance import run_agent_core_guardrail_acceptance
from agent_core.interaction_acceptance import run_agent_core_interaction_acceptance
from agent_core.interrupt_acceptance import run_agent_core_interrupt_acceptance
from agent_core.lifecycle_acceptance import run_agent_core_lifecycle_acceptance
from agent_core.manifest import (
    AgentCoreSDKManifest,
    FORBIDDEN_RUNTIME_DEPENDENCIES,
    FORBIDDEN_RUNTIME_PACKAGES,
    agent_core_sdk_manifest,
    evaluate_agent_core_api_lifecycle,
    evaluate_agent_core_api_stability,
    evaluate_agent_core_readiness,
)
from agent_core.native_tool_acceptance import run_agent_core_native_tool_acceptance
from agent_core.orchestration_acceptance import run_agent_core_orchestration_acceptance
from agent_core.packaging_acceptance import run_agent_core_packaging_acceptance
from agent_core.provider_acceptance import run_agent_core_provider_acceptance
from agent_core.provider_conformance import (
    AgentCoreProviderConformanceSpec,
    run_agent_core_provider_conformance,
)
from agent_core.providers import LLMProviderPort
from agent_core.provider_resilience_acceptance import (
    run_agent_core_provider_resilience_acceptance,
)
from agent_core.redaction_acceptance import run_agent_core_redaction_acceptance
from agent_core.recovery import run_agent_core_recovery_acceptance
from agent_core.resume_acceptance import run_agent_core_resume_acceptance
from agent_core.state_bundle_acceptance import run_agent_core_state_bundle_acceptance
from agent_core.storage_acceptance import run_agent_core_storage_acceptance
from agent_core.task_profile_acceptance import run_agent_core_task_profile_acceptance
from agent_core.trace_export_acceptance import run_agent_core_trace_export_acceptance
from agent_core.trace_replay_acceptance import run_agent_core_trace_replay_acceptance


@dataclass(frozen=True)
class AgentCoreValidationIssue:
    """One blocking issue from the aggregate SDK validation suite."""

    source: str
    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-validation-issue/v1",
            "source": self.source,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreRuntimeBoundaryHit:
    """One forbidden runtime dependency or package found inside the SDK package."""

    kind: str
    name: str
    path: str
    line: int = 0

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-runtime-boundary-hit/v1",
            "kind": self.kind,
            "name": self.name,
            "path": self.path,
            "line": self.line,
        }


@dataclass(frozen=True)
class AgentCoreRuntimeBoundaryReport:
    """Prompt-safe audit proving the SDK package has no runtime adapter imports."""

    status: str
    scanned_module_count: int = 0
    scanned_package_count: int = 0
    forbidden_dependency_hits: tuple[AgentCoreRuntimeBoundaryHit, ...] = ()
    forbidden_package_hits: tuple[AgentCoreRuntimeBoundaryHit, ...] = ()
    package_root: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def hit_count(self) -> int:
        return len(self.forbidden_dependency_hits) + len(self.forbidden_package_hits)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-runtime-boundary-report/v1",
            "status": self.status,
            "ready": self.ready,
            "hit_count": self.hit_count,
            "scanned_module_count": self.scanned_module_count,
            "scanned_package_count": self.scanned_package_count,
            "forbidden_dependency_hits": [
                hit.manifest() for hit in self.forbidden_dependency_hits
            ],
            "forbidden_package_hits": [
                hit.manifest() for hit in self.forbidden_package_hits
            ],
            "package_root": self.package_root,
            "metadata": dict(self.metadata),
        }


REPOSITORY_FORBIDDEN_EXAMPLE_MARKERS: tuple[str, ...] = (
    "import urllib",
    "from urllib",
    "import requests",
    "from requests",
    "import aiohttp",
    "from aiohttp",
    "api_key",
    "authorization",
    "bearer ",
    "base_url",
    "chat/completions",
    "openai",
    "anthropic",
)


@dataclass(frozen=True)
class AgentCoreRepositoryBoundaryReport:
    """Prompt-safe audit proving the repository does not ship runtime adapters."""

    status: str
    repository_root: str = ""
    scanned_top_level_dir_count: int = 0
    scanned_example_count: int = 0
    forbidden_directory_hits: tuple[AgentCoreRuntimeBoundaryHit, ...] = ()
    forbidden_example_hits: tuple[AgentCoreRuntimeBoundaryHit, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def hit_count(self) -> int:
        return len(self.forbidden_directory_hits) + len(self.forbidden_example_hits)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-repository-boundary-report/v1",
            "status": self.status,
            "ready": self.ready,
            "hit_count": self.hit_count,
            "repository_root": self.repository_root,
            "scanned_top_level_dir_count": self.scanned_top_level_dir_count,
            "scanned_example_count": self.scanned_example_count,
            "forbidden_directory_hits": [
                hit.manifest() for hit in self.forbidden_directory_hits
            ],
            "forbidden_example_hits": [
                hit.manifest() for hit in self.forbidden_example_hits
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreValidationReport:
    """Prompt-safe aggregate SDK validation report."""

    status: str
    summary: dict[str, Any] = field(default_factory=dict)
    runtime_boundary: dict[str, Any] = field(default_factory=dict)
    repository_boundary: dict[str, Any] = field(default_factory=dict)
    readiness: dict[str, Any] = field(default_factory=dict)
    api_lifecycle: dict[str, Any] = field(default_factory=dict)
    api_stability: dict[str, Any] = field(default_factory=dict)
    acceptance: dict[str, Any] = field(default_factory=dict)
    approval_acceptance: dict[str, Any] = field(default_factory=dict)
    budget_acceptance: dict[str, Any] = field(default_factory=dict)
    capability_governance_acceptance: dict[str, Any] = field(default_factory=dict)
    context_acceptance: dict[str, Any] = field(default_factory=dict)
    context_window_acceptance: dict[str, Any] = field(default_factory=dict)
    orchestration_acceptance: dict[str, Any] = field(default_factory=dict)
    concurrency_acceptance: dict[str, Any] = field(default_factory=dict)
    coordination_acceptance: dict[str, Any] = field(default_factory=dict)
    durable_session_acceptance: dict[str, Any] = field(default_factory=dict)
    event_acceptance: dict[str, Any] = field(default_factory=dict)
    external_backend_acceptance: dict[str, Any] = field(default_factory=dict)
    guardrail_acceptance: dict[str, Any] = field(default_factory=dict)
    interaction_acceptance: dict[str, Any] = field(default_factory=dict)
    interrupt_acceptance: dict[str, Any] = field(default_factory=dict)
    lifecycle_acceptance: dict[str, Any] = field(default_factory=dict)
    native_tool_acceptance: dict[str, Any] = field(default_factory=dict)
    packaging_acceptance: dict[str, Any] = field(default_factory=dict)
    provider_acceptance: dict[str, Any] = field(default_factory=dict)
    provider_conformance: dict[str, Any] = field(default_factory=dict)
    provider_resilience_acceptance: dict[str, Any] = field(default_factory=dict)
    redaction_acceptance: dict[str, Any] = field(default_factory=dict)
    eval_suite_acceptance: dict[str, Any] = field(default_factory=dict)
    state_bundle_acceptance: dict[str, Any] = field(default_factory=dict)
    storage_acceptance: dict[str, Any] = field(default_factory=dict)
    task_profile_acceptance: dict[str, Any] = field(default_factory=dict)
    trace_export_acceptance: dict[str, Any] = field(default_factory=dict)
    trace_replay_acceptance: dict[str, Any] = field(default_factory=dict)
    recovery: dict[str, Any] = field(default_factory=dict)
    resume: dict[str, Any] = field(default_factory=dict)
    issues: tuple[AgentCoreValidationIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-validation-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "issues": [issue.manifest() for issue in self.issues],
            "summary": dict(self.summary),
            "runtime_boundary": dict(self.runtime_boundary),
            "repository_boundary": dict(self.repository_boundary),
            "readiness": dict(self.readiness),
            "api_lifecycle": dict(self.api_lifecycle),
            "api_stability": dict(self.api_stability),
            "acceptance": dict(self.acceptance),
            "approval_acceptance": dict(self.approval_acceptance),
            "budget_acceptance": dict(self.budget_acceptance),
            "capability_governance_acceptance": dict(self.capability_governance_acceptance),
            "context_acceptance": dict(self.context_acceptance),
            "context_window_acceptance": dict(self.context_window_acceptance),
            "orchestration_acceptance": dict(self.orchestration_acceptance),
            "concurrency_acceptance": dict(self.concurrency_acceptance),
            "coordination_acceptance": dict(self.coordination_acceptance),
            "durable_session_acceptance": dict(self.durable_session_acceptance),
            "event_acceptance": dict(self.event_acceptance),
            "external_backend_acceptance": dict(self.external_backend_acceptance),
            "guardrail_acceptance": dict(self.guardrail_acceptance),
            "interaction_acceptance": dict(self.interaction_acceptance),
            "interrupt_acceptance": dict(self.interrupt_acceptance),
            "lifecycle_acceptance": dict(self.lifecycle_acceptance),
            "native_tool_acceptance": dict(self.native_tool_acceptance),
            "packaging_acceptance": dict(self.packaging_acceptance),
            "provider_acceptance": dict(self.provider_acceptance),
            "provider_conformance": dict(self.provider_conformance),
            "provider_resilience_acceptance": dict(self.provider_resilience_acceptance),
            "redaction_acceptance": dict(self.redaction_acceptance),
            "eval_suite_acceptance": dict(self.eval_suite_acceptance),
            "state_bundle_acceptance": dict(self.state_bundle_acceptance),
            "storage_acceptance": dict(self.storage_acceptance),
            "task_profile_acceptance": dict(self.task_profile_acceptance),
            "trace_export_acceptance": dict(self.trace_export_acceptance),
            "trace_replay_acceptance": dict(self.trace_replay_acceptance),
            "recovery": dict(self.recovery),
            "resume": dict(self.resume),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AgentCoreValidationSuite:
    """Run all pure-SDK gates needed before host-runtime adapter migration."""

    metadata: dict[str, Any] = field(default_factory=dict)

    async def run(
        self,
        *,
        sdk_manifest: AgentCoreSDKManifest | dict[str, Any] | None = None,
        package_root: str | Path | None = None,
        provider_conformance_provider: LLMProviderPort | None = None,
        provider_conformance_spec: AgentCoreProviderConformanceSpec | None = None,
    ) -> AgentCoreValidationReport:
        manifest = sdk_manifest or agent_core_sdk_manifest()
        manifest_dict = (
            manifest.manifest()
            if isinstance(manifest, AgentCoreSDKManifest)
            else dict(manifest)
        )
        runtime_boundary = evaluate_agent_core_runtime_boundary(
            manifest,
            package_root=package_root,
            metadata={"validation_gate": "runtime_boundary", **dict(self.metadata)},
        ).manifest()
        repository_boundary = evaluate_agent_core_repository_boundary(
            repository_root=_repository_root_for_package_root(package_root),
            metadata={"validation_gate": "repository_boundary", **dict(self.metadata)},
        ).manifest()
        readiness = evaluate_agent_core_readiness(manifest).manifest()
        api_lifecycle = evaluate_agent_core_api_lifecycle(manifest).manifest()
        api_stability = evaluate_agent_core_api_stability(manifest).manifest()
        acceptance = (
            await run_agent_core_acceptance(
                sdk_manifest=manifest,
                metadata={"validation_gate": "acceptance", **dict(self.metadata)},
            )
        ).manifest()
        approval_acceptance = (
            await run_agent_core_approval_acceptance(
                metadata={"validation_gate": "approval_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        budget_acceptance = (
            await run_agent_core_budget_acceptance(
                metadata={"validation_gate": "budget_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        capability_governance_acceptance = (
            await run_agent_core_capability_governance_acceptance(
                metadata={
                    "validation_gate": "capability_governance_acceptance",
                    **dict(self.metadata),
                }
            )
        ).manifest()
        context_acceptance = (
            await run_agent_core_context_acceptance(
                metadata={"validation_gate": "context_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        context_window_acceptance = (
            await run_agent_core_context_window_acceptance(
                metadata={
                    "validation_gate": "context_window_acceptance",
                    **dict(self.metadata),
                }
            )
        ).manifest()
        orchestration_acceptance = (
            await run_agent_core_orchestration_acceptance(
                metadata={"validation_gate": "orchestration_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        concurrency_acceptance = (
            await run_agent_core_concurrency_acceptance(
                metadata={"validation_gate": "concurrency_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        coordination_acceptance = (
            await run_agent_core_coordination_acceptance(
                metadata={"validation_gate": "coordination_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        durable_session_acceptance = (
            await run_agent_core_durable_session_acceptance(
                metadata={
                    "validation_gate": "durable_session_acceptance",
                    **dict(self.metadata),
                }
            )
        ).manifest()
        event_acceptance = (
            await run_agent_core_event_acceptance(
                metadata={"validation_gate": "event_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        external_backend_acceptance = (
            await run_agent_core_external_backend_acceptance(
                metadata={
                    "validation_gate": "external_backend_acceptance",
                    **dict(self.metadata),
                }
            )
        ).manifest()
        guardrail_acceptance = (
            await run_agent_core_guardrail_acceptance(
                metadata={"validation_gate": "guardrail_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        interaction_acceptance = (
            await run_agent_core_interaction_acceptance(
                metadata={"validation_gate": "interaction_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        interrupt_acceptance = (
            await run_agent_core_interrupt_acceptance(
                metadata={"validation_gate": "interrupt_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        lifecycle_acceptance = (
            await run_agent_core_lifecycle_acceptance(
                metadata={"validation_gate": "lifecycle_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        native_tool_acceptance = (
            await run_agent_core_native_tool_acceptance(
                metadata={"validation_gate": "native_tool_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        provider_acceptance = (
            await run_agent_core_provider_acceptance(
                metadata={"validation_gate": "provider_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        provider_conformance = (
            await run_agent_core_provider_conformance(
                provider=provider_conformance_provider,
                spec=provider_conformance_spec,
                metadata={
                    "validation_gate": "provider_conformance",
                    **dict(self.metadata),
                },
            )
        ).manifest()
        provider_resilience_acceptance = (
            await run_agent_core_provider_resilience_acceptance(
                metadata={
                    "validation_gate": "provider_resilience_acceptance",
                    **dict(self.metadata),
                }
            )
        ).manifest()
        redaction_acceptance = (
            await run_agent_core_redaction_acceptance(
                metadata={"validation_gate": "redaction_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        eval_suite_acceptance = (
            await run_agent_core_eval_suite_acceptance(
                metadata={"validation_gate": "eval_suite_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        state_bundle_acceptance = (
            await run_agent_core_state_bundle_acceptance(
                metadata={"validation_gate": "state_bundle_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        packaging_acceptance = (
            await run_agent_core_packaging_acceptance(
                metadata={"validation_gate": "packaging_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        storage_acceptance = (
            await run_agent_core_storage_acceptance(
                metadata={"validation_gate": "storage_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        task_profile_acceptance = (
            await run_agent_core_task_profile_acceptance(
                metadata={"validation_gate": "task_profile_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        trace_export_acceptance = (
            await run_agent_core_trace_export_acceptance(
                metadata={"validation_gate": "trace_export_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        trace_replay_acceptance = (
            await run_agent_core_trace_replay_acceptance(
                metadata={"validation_gate": "trace_replay_acceptance", **dict(self.metadata)}
            )
        ).manifest()
        recovery = (
            await run_agent_core_recovery_acceptance(
                metadata={"validation_gate": "recovery", **dict(self.metadata)}
            )
        ).manifest()
        resume = (
            await run_agent_core_resume_acceptance(
                metadata={"validation_gate": "resume", **dict(self.metadata)}
            )
        ).manifest()
        issues = _validation_issues(
            runtime_boundary=runtime_boundary,
            repository_boundary=repository_boundary,
            readiness=readiness,
            api_lifecycle=api_lifecycle,
            api_stability=api_stability,
            acceptance=acceptance,
            approval_acceptance=approval_acceptance,
            budget_acceptance=budget_acceptance,
            capability_governance_acceptance=capability_governance_acceptance,
            context_acceptance=context_acceptance,
            context_window_acceptance=context_window_acceptance,
            orchestration_acceptance=orchestration_acceptance,
            concurrency_acceptance=concurrency_acceptance,
            coordination_acceptance=coordination_acceptance,
            durable_session_acceptance=durable_session_acceptance,
            event_acceptance=event_acceptance,
            external_backend_acceptance=external_backend_acceptance,
            guardrail_acceptance=guardrail_acceptance,
            interaction_acceptance=interaction_acceptance,
            interrupt_acceptance=interrupt_acceptance,
            lifecycle_acceptance=lifecycle_acceptance,
            native_tool_acceptance=native_tool_acceptance,
            packaging_acceptance=packaging_acceptance,
            provider_acceptance=provider_acceptance,
            provider_conformance=provider_conformance,
            provider_resilience_acceptance=provider_resilience_acceptance,
            redaction_acceptance=redaction_acceptance,
            eval_suite_acceptance=eval_suite_acceptance,
            state_bundle_acceptance=state_bundle_acceptance,
            storage_acceptance=storage_acceptance,
            task_profile_acceptance=task_profile_acceptance,
            trace_export_acceptance=trace_export_acceptance,
            trace_replay_acceptance=trace_replay_acceptance,
            recovery=recovery,
            resume=resume,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        summary = _validation_summary(
            status=status,
            issues=issues,
            sdk_manifest=manifest_dict,
            runtime_boundary=runtime_boundary,
            repository_boundary=repository_boundary,
            readiness=readiness,
            api_lifecycle=api_lifecycle,
            api_stability=api_stability,
            acceptance=acceptance,
            approval_acceptance=approval_acceptance,
            budget_acceptance=budget_acceptance,
            capability_governance_acceptance=capability_governance_acceptance,
            context_acceptance=context_acceptance,
            context_window_acceptance=context_window_acceptance,
            orchestration_acceptance=orchestration_acceptance,
            concurrency_acceptance=concurrency_acceptance,
            coordination_acceptance=coordination_acceptance,
            durable_session_acceptance=durable_session_acceptance,
            event_acceptance=event_acceptance,
            external_backend_acceptance=external_backend_acceptance,
            guardrail_acceptance=guardrail_acceptance,
            interaction_acceptance=interaction_acceptance,
            interrupt_acceptance=interrupt_acceptance,
            lifecycle_acceptance=lifecycle_acceptance,
            native_tool_acceptance=native_tool_acceptance,
            packaging_acceptance=packaging_acceptance,
            provider_acceptance=provider_acceptance,
            provider_conformance=provider_conformance,
            provider_resilience_acceptance=provider_resilience_acceptance,
            redaction_acceptance=redaction_acceptance,
            eval_suite_acceptance=eval_suite_acceptance,
            state_bundle_acceptance=state_bundle_acceptance,
            storage_acceptance=storage_acceptance,
            task_profile_acceptance=task_profile_acceptance,
            trace_export_acceptance=trace_export_acceptance,
            trace_replay_acceptance=trace_replay_acceptance,
            recovery=recovery,
            resume=resume,
        )
        return AgentCoreValidationReport(
            status=status,
            summary=summary,
            runtime_boundary=runtime_boundary,
            repository_boundary=repository_boundary,
            readiness=readiness,
            api_lifecycle=api_lifecycle,
            api_stability=api_stability,
            acceptance=acceptance,
            approval_acceptance=approval_acceptance,
            budget_acceptance=budget_acceptance,
            capability_governance_acceptance=capability_governance_acceptance,
            context_acceptance=context_acceptance,
            context_window_acceptance=context_window_acceptance,
            orchestration_acceptance=orchestration_acceptance,
            concurrency_acceptance=concurrency_acceptance,
            coordination_acceptance=coordination_acceptance,
            durable_session_acceptance=durable_session_acceptance,
            event_acceptance=event_acceptance,
            external_backend_acceptance=external_backend_acceptance,
            guardrail_acceptance=guardrail_acceptance,
            interaction_acceptance=interaction_acceptance,
            interrupt_acceptance=interrupt_acceptance,
            lifecycle_acceptance=lifecycle_acceptance,
            native_tool_acceptance=native_tool_acceptance,
            packaging_acceptance=packaging_acceptance,
            provider_acceptance=provider_acceptance,
            provider_conformance=provider_conformance,
            provider_resilience_acceptance=provider_resilience_acceptance,
            redaction_acceptance=redaction_acceptance,
            eval_suite_acceptance=eval_suite_acceptance,
            state_bundle_acceptance=state_bundle_acceptance,
            storage_acceptance=storage_acceptance,
            task_profile_acceptance=task_profile_acceptance,
            trace_export_acceptance=trace_export_acceptance,
            trace_replay_acceptance=trace_replay_acceptance,
            recovery=recovery,
            resume=resume,
            issues=issues,
            metadata={"scenario": "agent_core_validation", **dict(self.metadata)},
        )


async def run_agent_core_validation(
    *,
    sdk_manifest: AgentCoreSDKManifest | dict[str, Any] | None = None,
    package_root: str | Path | None = None,
    provider_conformance_provider: LLMProviderPort | None = None,
    provider_conformance_spec: AgentCoreProviderConformanceSpec | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreValidationReport:
    """Run the default aggregate SDK validation suite."""

    return await AgentCoreValidationSuite(metadata=dict(metadata or {})).run(
        sdk_manifest=sdk_manifest,
        package_root=package_root,
        provider_conformance_provider=provider_conformance_provider,
        provider_conformance_spec=provider_conformance_spec,
    )


def evaluate_agent_core_runtime_boundary(
    sdk_manifest: AgentCoreSDKManifest | dict[str, Any] | None = None,
    *,
    package_root: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreRuntimeBoundaryReport:
    """Audit the SDK package for forbidden runtime imports and adapter packages."""

    manifest = (
        (sdk_manifest or agent_core_sdk_manifest()).manifest()
        if isinstance(sdk_manifest, AgentCoreSDKManifest) or sdk_manifest is None
        else dict(sdk_manifest)
    )
    boundary = manifest.get("runtime_boundary") or {}
    forbidden_dependencies = tuple(
        str(item)
        for item in boundary.get("forbidden_dependencies", FORBIDDEN_RUNTIME_DEPENDENCIES)
        or ()
    )
    forbidden_packages = tuple(
        str(item)
        for item in boundary.get("forbidden_packages", FORBIDDEN_RUNTIME_PACKAGES)
        or ()
    )
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parent
    dependency_hits: list[AgentCoreRuntimeBoundaryHit] = []
    scanned_module_count = 0
    for path in sorted(root.rglob("*.py")) if root.exists() else ():
        if "__pycache__" in path.parts:
            continue
        scanned_module_count += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            dependency_hits.append(
                AgentCoreRuntimeBoundaryHit(
                    kind="syntax_error",
                    name="python_syntax",
                    path=_relative_path(path, root),
                    line=int(exc.lineno or 0),
                )
            )
            continue
        for name, line in _import_names(tree):
            if _matches_forbidden_name(name, forbidden_dependencies):
                dependency_hits.append(
                    AgentCoreRuntimeBoundaryHit(
                        kind="forbidden_dependency",
                        name=name,
                        path=_relative_path(path, root),
                        line=line,
                    )
                )

    package_hits: list[AgentCoreRuntimeBoundaryHit] = []
    scanned_package_count = 0
    if root.exists():
        for path in sorted(root.iterdir()):
            if not path.is_dir() or path.name == "__pycache__":
                continue
            scanned_package_count += 1
            if path.name in forbidden_packages:
                package_hits.append(
                    AgentCoreRuntimeBoundaryHit(
                        kind="forbidden_package",
                        name=path.name,
                        path=_relative_path(path, root),
                    )
                )

    status = "blocked" if dependency_hits or package_hits else "ready"
    return AgentCoreRuntimeBoundaryReport(
        status=status,
        scanned_module_count=scanned_module_count,
        scanned_package_count=scanned_package_count,
        forbidden_dependency_hits=tuple(dependency_hits),
        forbidden_package_hits=tuple(package_hits),
        package_root=str(root),
        metadata={
            "sdk_manifest_schema": str(manifest.get("schema_version") or ""),
            "forbidden_dependency_count": len(forbidden_dependencies),
            "forbidden_package_count": len(forbidden_packages),
            **dict(metadata or {}),
        },
    )


def evaluate_agent_core_repository_boundary(
    *,
    repository_root: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentCoreRepositoryBoundaryReport:
    """Audit the repository for concrete runtime/provider adapter artifacts."""

    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[1]
    forbidden_dirs = tuple(FORBIDDEN_RUNTIME_PACKAGES)
    directory_hits: list[AgentCoreRuntimeBoundaryHit] = []
    scanned_top_level_dir_count = 0
    if root.exists():
        for path in sorted(root.iterdir()):
            if not path.is_dir() or path.name in {".git", "__pycache__"}:
                continue
            scanned_top_level_dir_count += 1
            if path.name in forbidden_dirs:
                directory_hits.append(
                    AgentCoreRuntimeBoundaryHit(
                        kind="forbidden_runtime_directory",
                        name=path.name,
                        path=_relative_path(path, root),
                    )
                )

    example_hits: list[AgentCoreRuntimeBoundaryHit] = []
    scanned_example_count = 0
    examples_root = root / "examples"
    if examples_root.exists():
        for path in sorted(examples_root.glob("*.py")):
            scanned_example_count += 1
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            for marker in REPOSITORY_FORBIDDEN_EXAMPLE_MARKERS:
                line = _first_marker_line(text, marker)
                if line:
                    example_hits.append(
                        AgentCoreRuntimeBoundaryHit(
                            kind="forbidden_example_adapter_marker",
                            name=marker,
                            path=_relative_path(path, root),
                            line=line,
                        )
                    )

    status = "blocked" if directory_hits or example_hits else "ready"
    return AgentCoreRepositoryBoundaryReport(
        status=status,
        repository_root=str(root),
        scanned_top_level_dir_count=scanned_top_level_dir_count,
        scanned_example_count=scanned_example_count,
        forbidden_directory_hits=tuple(directory_hits),
        forbidden_example_hits=tuple(example_hits),
        metadata={
            "forbidden_directory_count": len(forbidden_dirs),
            "forbidden_example_marker_count": len(REPOSITORY_FORBIDDEN_EXAMPLE_MARKERS),
            **dict(metadata or {}),
        },
    )


def _repository_root_for_package_root(package_root: str | Path | None) -> Path:
    if package_root is None:
        return Path(__file__).resolve().parents[1]
    root = Path(package_root)
    return root.parent if root.name == "agent_core" else root


def _first_marker_line(text: str, marker: str) -> int:
    marker = marker.lower()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if marker in line:
            return line_number
    return 0


def _validation_issues(
    *,
    runtime_boundary: dict[str, Any],
    repository_boundary: dict[str, Any],
    readiness: dict[str, Any],
    api_lifecycle: dict[str, Any],
    api_stability: dict[str, Any],
    acceptance: dict[str, Any],
    approval_acceptance: dict[str, Any],
    budget_acceptance: dict[str, Any],
    capability_governance_acceptance: dict[str, Any],
    context_acceptance: dict[str, Any],
    context_window_acceptance: dict[str, Any],
    orchestration_acceptance: dict[str, Any],
    concurrency_acceptance: dict[str, Any],
    coordination_acceptance: dict[str, Any],
    durable_session_acceptance: dict[str, Any],
    event_acceptance: dict[str, Any],
    external_backend_acceptance: dict[str, Any],
    guardrail_acceptance: dict[str, Any],
    interaction_acceptance: dict[str, Any],
    interrupt_acceptance: dict[str, Any],
    lifecycle_acceptance: dict[str, Any],
    native_tool_acceptance: dict[str, Any],
    packaging_acceptance: dict[str, Any],
    provider_acceptance: dict[str, Any],
    provider_conformance: dict[str, Any],
    provider_resilience_acceptance: dict[str, Any],
    redaction_acceptance: dict[str, Any],
    eval_suite_acceptance: dict[str, Any],
    state_bundle_acceptance: dict[str, Any],
    storage_acceptance: dict[str, Any],
    task_profile_acceptance: dict[str, Any],
    trace_export_acceptance: dict[str, Any],
    trace_replay_acceptance: dict[str, Any],
    recovery: dict[str, Any],
    resume: dict[str, Any],
) -> tuple[AgentCoreValidationIssue, ...]:
    issues: list[AgentCoreValidationIssue] = []
    _extend_boundary_issues(issues, runtime_boundary)
    _extend_repository_boundary_issues(issues, repository_boundary)
    _extend_report_issues(issues, source="readiness", report=readiness)
    _extend_report_issues(issues, source="api_lifecycle", report=api_lifecycle)
    _extend_api_stability_issues(issues, api_stability)
    _extend_report_issues(issues, source="acceptance", report=acceptance)
    _extend_report_issues(issues, source="approval_acceptance", report=approval_acceptance)
    _extend_report_issues(issues, source="budget_acceptance", report=budget_acceptance)
    _extend_report_issues(
        issues,
        source="capability_governance_acceptance",
        report=capability_governance_acceptance,
    )
    _extend_report_issues(issues, source="context_acceptance", report=context_acceptance)
    _extend_report_issues(
        issues,
        source="context_window_acceptance",
        report=context_window_acceptance,
    )
    _extend_report_issues(
        issues,
        source="orchestration_acceptance",
        report=orchestration_acceptance,
    )
    _extend_report_issues(
        issues,
        source="concurrency_acceptance",
        report=concurrency_acceptance,
    )
    _extend_report_issues(
        issues,
        source="coordination_acceptance",
        report=coordination_acceptance,
    )
    _extend_report_issues(
        issues,
        source="durable_session_acceptance",
        report=durable_session_acceptance,
    )
    _extend_report_issues(
        issues,
        source="event_acceptance",
        report=event_acceptance,
    )
    _extend_report_issues(
        issues,
        source="external_backend_acceptance",
        report=external_backend_acceptance,
    )
    _extend_report_issues(
        issues,
        source="guardrail_acceptance",
        report=guardrail_acceptance,
    )
    _extend_report_issues(
        issues,
        source="interaction_acceptance",
        report=interaction_acceptance,
    )
    _extend_report_issues(
        issues,
        source="interrupt_acceptance",
        report=interrupt_acceptance,
    )
    _extend_report_issues(
        issues,
        source="lifecycle_acceptance",
        report=lifecycle_acceptance,
    )
    _extend_report_issues(
        issues,
        source="native_tool_acceptance",
        report=native_tool_acceptance,
    )
    _extend_report_issues(
        issues,
        source="packaging_acceptance",
        report=packaging_acceptance,
    )
    _extend_report_issues(issues, source="provider_acceptance", report=provider_acceptance)
    _extend_report_issues(issues, source="provider_conformance", report=provider_conformance)
    _extend_report_issues(
        issues,
        source="provider_resilience_acceptance",
        report=provider_resilience_acceptance,
    )
    _extend_report_issues(issues, source="redaction_acceptance", report=redaction_acceptance)
    _extend_report_issues(
        issues,
        source="eval_suite_acceptance",
        report=eval_suite_acceptance,
    )
    _extend_report_issues(
        issues,
        source="state_bundle_acceptance",
        report=state_bundle_acceptance,
    )
    _extend_report_issues(issues, source="storage_acceptance", report=storage_acceptance)
    _extend_report_issues(
        issues,
        source="task_profile_acceptance",
        report=task_profile_acceptance,
    )
    _extend_report_issues(
        issues,
        source="trace_export_acceptance",
        report=trace_export_acceptance,
    )
    _extend_report_issues(
        issues,
        source="trace_replay_acceptance",
        report=trace_replay_acceptance,
    )
    _extend_report_issues(issues, source="recovery", report=recovery)
    _extend_report_issues(issues, source="resume", report=resume)
    return tuple(issues)


def _validation_summary(
    *,
    status: str,
    issues: tuple[AgentCoreValidationIssue, ...],
    sdk_manifest: dict[str, Any],
    runtime_boundary: dict[str, Any],
    repository_boundary: dict[str, Any],
    readiness: dict[str, Any],
    api_lifecycle: dict[str, Any],
    api_stability: dict[str, Any],
    acceptance: dict[str, Any],
    approval_acceptance: dict[str, Any],
    budget_acceptance: dict[str, Any],
    capability_governance_acceptance: dict[str, Any],
    context_acceptance: dict[str, Any],
    context_window_acceptance: dict[str, Any],
    orchestration_acceptance: dict[str, Any],
    concurrency_acceptance: dict[str, Any],
    coordination_acceptance: dict[str, Any],
    durable_session_acceptance: dict[str, Any],
    event_acceptance: dict[str, Any],
    external_backend_acceptance: dict[str, Any],
    guardrail_acceptance: dict[str, Any],
    interaction_acceptance: dict[str, Any],
    interrupt_acceptance: dict[str, Any],
    lifecycle_acceptance: dict[str, Any],
    native_tool_acceptance: dict[str, Any],
    packaging_acceptance: dict[str, Any],
    provider_acceptance: dict[str, Any],
    provider_conformance: dict[str, Any],
    provider_resilience_acceptance: dict[str, Any],
    redaction_acceptance: dict[str, Any],
    eval_suite_acceptance: dict[str, Any],
    state_bundle_acceptance: dict[str, Any],
    storage_acceptance: dict[str, Any],
    task_profile_acceptance: dict[str, Any],
    trace_export_acceptance: dict[str, Any],
    trace_replay_acceptance: dict[str, Any],
    recovery: dict[str, Any],
    resume: dict[str, Any],
) -> dict[str, Any]:
    gates = {
        "runtime_boundary": runtime_boundary,
        "repository_boundary": repository_boundary,
        "readiness": readiness,
        "api_lifecycle": api_lifecycle,
        "api_stability": api_stability,
        "acceptance": acceptance,
        "approval_acceptance": approval_acceptance,
        "budget_acceptance": budget_acceptance,
        "capability_governance_acceptance": capability_governance_acceptance,
        "context_acceptance": context_acceptance,
        "context_window_acceptance": context_window_acceptance,
        "orchestration_acceptance": orchestration_acceptance,
        "concurrency_acceptance": concurrency_acceptance,
        "coordination_acceptance": coordination_acceptance,
        "durable_session_acceptance": durable_session_acceptance,
        "event_acceptance": event_acceptance,
        "external_backend_acceptance": external_backend_acceptance,
        "guardrail_acceptance": guardrail_acceptance,
        "interaction_acceptance": interaction_acceptance,
        "interrupt_acceptance": interrupt_acceptance,
        "lifecycle_acceptance": lifecycle_acceptance,
        "native_tool_acceptance": native_tool_acceptance,
        "packaging_acceptance": packaging_acceptance,
        "provider_acceptance": provider_acceptance,
        "provider_conformance": provider_conformance,
        "provider_resilience_acceptance": provider_resilience_acceptance,
        "redaction_acceptance": redaction_acceptance,
        "eval_suite_acceptance": eval_suite_acceptance,
        "state_bundle_acceptance": state_bundle_acceptance,
        "storage_acceptance": storage_acceptance,
        "task_profile_acceptance": task_profile_acceptance,
        "trace_export_acceptance": trace_export_acceptance,
        "trace_replay_acceptance": trace_replay_acceptance,
        "recovery": recovery,
        "resume": resume,
    }
    ready_gates = tuple(name for name, report in gates.items() if report.get("ready") is True)
    blocked_gates = tuple(name for name, report in gates.items() if report.get("ready") is not True)
    issue_counts: dict[str, int] = {}
    for issue in issues:
        issue_counts[issue.source] = issue_counts.get(issue.source, 0) + 1
    packaging_public_api = packaging_acceptance.get("public_api") or {}
    packaging_project = packaging_acceptance.get("project_metadata") or {}
    packaging_repository = packaging_acceptance.get("repository_files") or {}
    packaging_examples = (packaging_acceptance.get("examples") or {}).get("examples") or {}
    provider_matrix = provider_conformance.get("check_matrix") or {}
    storage_interfaces = sdk_manifest.get("storage_backend_interfaces") or {}
    context_pipeline = sdk_manifest.get("context_pipeline") or {}
    task_profile_matrix = task_profile_acceptance.get("matrix") or {}
    return {
        "schema_version": "agent-core-validation-summary/v1",
        "status": status,
        "ready": status == "ready",
        "gate_count": len(gates),
        "ready_gate_count": len(ready_gates),
        "blocked_gate_count": len(blocked_gates),
        "ready_gates": list(ready_gates),
        "blocked_gates": list(blocked_gates),
        "issue_count": len(issues),
        "error_count": sum(1 for issue in issues if issue.severity == "error"),
        "issue_counts_by_source": dict(sorted(issue_counts.items())),
        "runtime_free": (
            runtime_boundary.get("ready") is True
            and repository_boundary.get("ready") is True
            and int(runtime_boundary.get("hit_count") or 0) == 0
            and int(repository_boundary.get("hit_count") or 0) == 0
        ),
        "public_api_count": int(packaging_public_api.get("public_api_count") or 0),
        "contract_api_count": int(packaging_public_api.get("contract_api_count") or 0),
        "root_export_count": int(packaging_public_api.get("root_export_count") or 0),
        "package_dependency_boundary": {
            "runtime_dependency_count": int(
                packaging_project.get("runtime_dependency_count") or 0
            ),
            "optional_dependency_groups": list(
                packaging_project.get("optional_dependency_groups") or ()
            ),
            "forbidden_dependency_hit_count": len(
                packaging_project.get("forbidden_dependency_hits") or ()
            ),
            "forbidden_dependency_hits": list(
                packaging_project.get("forbidden_dependency_hits") or ()
            ),
        },
        "package_documentation_hygiene": {
            "mojibake_hit_count": int(
                (
                    (packaging_repository.get("documentation") or {}).get(
                        "mojibake_hit_count"
                    )
                    if isinstance(packaging_repository.get("documentation"), dict)
                    else 0
                )
                or 0
            ),
            "mojibake_hits": list(
                (
                    (packaging_repository.get("documentation") or {}).get(
                        "mojibake_hits"
                    )
                    if isinstance(packaging_repository.get("documentation"), dict)
                    else ()
                )
                or ()
            ),
        },
        "examples_executed": {
            name: {
                "exit_code": ((info.get("run") or {}).get("exit_code")),
                "json_valid": bool((info.get("run") or {}).get("json_valid")),
            }
            for name, info in packaging_examples.items()
            if isinstance(info, dict)
        },
        "provider_conformance": {
            "provider_source": provider_matrix.get("provider_source"),
            "required_checks": list(provider_matrix.get("required_checks") or ()),
            "failed_required_checks": list(
                provider_matrix.get("failed_required_checks") or ()
            ),
            "error_issue_codes": list(provider_matrix.get("error_issue_codes") or ()),
        },
        "context_pipeline": {
            "stage_count": int(context_pipeline.get("stage_count") or 0),
            "stage_order": list(context_pipeline.get("stage_order") or ()),
        },
        "storage_backend_interfaces": {
            "role_count": len(storage_interfaces.get("roles") or ()),
            "builtin_kinds": list(storage_interfaces.get("builtin_kinds") or ()),
            "external_kinds": list(storage_interfaces.get("external_kinds") or ()),
            "role_contract_count": len(storage_interfaces.get("role_contracts") or ()),
        },
        "task_profile_coverage": {
            "profile_count": int(task_profile_matrix.get("profile_count") or 0),
            "profile_names": list(task_profile_matrix.get("profile_names") or ()),
            "completed_profiles": list(
                task_profile_matrix.get("completed_profiles") or ()
            ),
            "tool_enabled_profiles": list(
                task_profile_matrix.get("tool_enabled_profiles") or ()
            ),
            "memory_enabled_profiles": list(
                task_profile_matrix.get("memory_enabled_profiles") or ()
            ),
            "context_enabled_profiles": list(
                task_profile_matrix.get("context_enabled_profiles") or ()
            ),
            "skill_names": list(task_profile_matrix.get("skill_names") or ()),
            "provider_names": list(task_profile_matrix.get("provider_names") or ()),
        },
        "migration_readiness": _migration_readiness_summary(
            status=status,
            ready_gates=ready_gates,
            blocked_gates=blocked_gates,
            runtime_free=(
                runtime_boundary.get("ready") is True
                and repository_boundary.get("ready") is True
                and int(runtime_boundary.get("hit_count") or 0) == 0
                and int(repository_boundary.get("hit_count") or 0) == 0
            ),
            packaging_examples=packaging_examples,
            provider_matrix=provider_matrix,
            storage_interfaces=storage_interfaces,
            context_pipeline=context_pipeline,
            task_profile_matrix=task_profile_matrix,
        ),
    }


def _migration_readiness_summary(
    *,
    status: str,
    ready_gates: tuple[str, ...],
    blocked_gates: tuple[str, ...],
    runtime_free: bool,
    packaging_examples: dict[str, Any],
    provider_matrix: dict[str, Any],
    storage_interfaces: dict[str, Any],
    context_pipeline: dict[str, Any],
    task_profile_matrix: dict[str, Any],
) -> dict[str, Any]:
    examples_ok = all(
        isinstance(info, dict)
        and ((info.get("run") or {}).get("exit_code")) == 0
        and bool((info.get("run") or {}).get("json_valid"))
        for info in packaging_examples.values()
    )
    deterministic_provider_ok = (
        provider_matrix.get("provider_source") == "deterministic"
        and not list(provider_matrix.get("failed_required_checks") or ())
        and not list(provider_matrix.get("error_issue_codes") or ())
    )
    live_provider_ok = (
        provider_matrix.get("provider_source") == "external"
        and bool(provider_matrix.get("ready_for_real_provider_smoke"))
    )
    live_provider_status = (
        "passed"
        if live_provider_ok
        else (
            "failed"
            if provider_matrix.get("provider_source") == "external"
            else "requires_external_provider"
        )
    )
    live_provider_gap_status = (
        "covered"
        if live_provider_ok
        else (
            "failed"
            if provider_matrix.get("provider_source") == "external"
            else "not_run_in_sdk_validation"
        )
    )
    provider_conformance_ok = deterministic_provider_ok or live_provider_ok
    storage_contracts_ready = (
        len(storage_interfaces.get("roles") or ()) > 0
        and len(storage_interfaces.get("role_contracts") or ())
        == len(storage_interfaces.get("roles") or ())
        and {"in_memory", "sqlite", "markdown"}
        <= set(str(kind) for kind in storage_interfaces.get("builtin_kinds") or ())
        and {"postgres", "vector", "graph", "product"}
        <= set(str(kind) for kind in storage_interfaces.get("external_kinds") or ())
    )
    context_pipeline_ready = int(context_pipeline.get("stage_count") or 0) >= 7
    expected_task_profiles = {"code", "ops", "security"}
    task_profiles_ready = all(
        expected_task_profiles <= set(task_profile_matrix.get(key) or ())
        for key in (
            "completed_profiles",
            "tool_enabled_profiles",
            "memory_enabled_profiles",
            "context_enabled_profiles",
        )
    )
    sdk_core_usable = (
        status == "ready"
        and runtime_free
        and examples_ok
        and provider_conformance_ok
        and storage_contracts_ready
        and context_pipeline_ready
        and task_profiles_ready
    )
    not_covered = [
        {
            "item": "live_llm_provider_conformance",
            "status": live_provider_gap_status,
            "owner": "runtime_or_provider_adapter",
            "reason": "requires a runtime-supplied LLMProviderPort plus credentials",
        },
        {
            "item": "ravenstorm_runtime_adapter_acceptance",
            "status": "out_of_scope_for_sdk_repo",
            "owner": "ravenstorm_or_adapter_repo",
            "reason": "SDK validation intentionally excludes Raven/OpenAI Agents SDK/Graphiti/FastAPI adapters",
        },
        {
            "item": "production_backend_drivers",
            "status": "out_of_scope_for_sdk_repo",
            "owner": "runtime_or_storage_adapter",
            "reason": "PG, vector, graph, object storage, and product stores implement SDK ports outside core",
        },
        {
            "item": "domain_eval_suites",
            "status": "out_of_scope_for_sdk_repo",
            "owner": "runtime",
            "reason": "code, ops, security, and Raven-specific scoring datasets are product policy",
        },
    ]
    return {
        "schema_version": "agent-core-migration-readiness-summary/v1",
        "sdk_core_status": (
            "usable_for_live_provider_tests" if sdk_core_usable else "blocked"
        ),
        "sdk_core_usable": sdk_core_usable,
        "ready_for_runtime_adapter_work": sdk_core_usable,
        "ready_for_live_provider_conformance": sdk_core_usable,
        "live_provider_conformance_status": live_provider_status,
        "runtime_adapters_in_scope": False,
        "ravenstorm_adapter_in_scope": False,
        "evidence": {
            "ready_gate_count": len(ready_gates),
            "blocked_gate_count": len(blocked_gates),
            "runtime_free": runtime_free,
            "examples_ok": examples_ok,
            "deterministic_provider_ok": deterministic_provider_ok,
            "live_provider_ok": live_provider_ok,
            "provider_conformance_ok": provider_conformance_ok,
            "storage_contracts_ready": storage_contracts_ready,
            "context_pipeline_ready": context_pipeline_ready,
            "task_profiles_ready": task_profiles_ready,
        },
        "blocked_gates": list(blocked_gates),
        "not_covered_by_sdk_validation": not_covered,
        "next_validation_step": (
            "fix_external_provider_conformance_failures"
            if live_provider_status == "failed"
            else (
                "run_agent_core_provider_conformance(provider=runtime_provider)"
                if sdk_core_usable and not live_provider_ok
                else ""
            )
        ),
    }


def _extend_boundary_issues(
    issues: list[AgentCoreValidationIssue],
    report: dict[str, Any],
) -> None:
    if report.get("ready") is True:
        return
    for hit in report.get("forbidden_dependency_hits") or ():
        if not isinstance(hit, dict):
            continue
        issues.append(
            AgentCoreValidationIssue(
                source="runtime_boundary",
                code="forbidden_runtime_dependency",
                message=f"Forbidden runtime dependency in SDK package: {hit.get('name')}",
                metadata=dict(hit),
            )
        )
    for hit in report.get("forbidden_package_hits") or ():
        if not isinstance(hit, dict):
            continue
        issues.append(
            AgentCoreValidationIssue(
                source="runtime_boundary",
                code="forbidden_runtime_package",
                message=f"Forbidden runtime package in SDK package: {hit.get('name')}",
                metadata=dict(hit),
            )
        )
    if not any(issue.source == "runtime_boundary" for issue in issues):
        issues.append(
            AgentCoreValidationIssue(
                source="runtime_boundary",
                code="runtime_boundary_not_ready",
                message="runtime boundary audit is not ready.",
                metadata={"status": report.get("status")},
            )
        )


def _extend_repository_boundary_issues(
    issues: list[AgentCoreValidationIssue],
    report: dict[str, Any],
) -> None:
    if report.get("ready") is True:
        return
    for hit in report.get("forbidden_directory_hits") or ():
        if not isinstance(hit, dict):
            continue
        issues.append(
            AgentCoreValidationIssue(
                source="repository_boundary",
                code="forbidden_runtime_directory",
                message=f"Forbidden runtime adapter directory in SDK repository: {hit.get('name')}",
                metadata=dict(hit),
            )
        )
    for hit in report.get("forbidden_example_hits") or ():
        if not isinstance(hit, dict):
            continue
        issues.append(
            AgentCoreValidationIssue(
                source="repository_boundary",
                code="forbidden_example_adapter_marker",
                message=f"Forbidden runtime/provider adapter marker in SDK example: {hit.get('name')}",
                metadata=dict(hit),
            )
        )
    if not any(issue.source == "repository_boundary" for issue in issues):
        issues.append(
            AgentCoreValidationIssue(
                source="repository_boundary",
                code="repository_boundary_not_ready",
                message="repository boundary audit is not ready.",
                metadata={"status": report.get("status")},
            )
        )


def _extend_report_issues(
    issues: list[AgentCoreValidationIssue],
    *,
    source: str,
    report: dict[str, Any],
) -> None:
    if report.get("ready") is True:
        return
    for issue in report.get("issues") or ():
        if not isinstance(issue, dict):
            continue
        issues.append(
            AgentCoreValidationIssue(
                source=source,
                code=str(issue.get("code") or f"{source}_issue"),
                message=str(issue.get("message") or ""),
                severity=str(issue.get("severity") or "error"),
                metadata=dict(issue.get("metadata") or {}),
            )
        )
    if not any(issue.source == source for issue in issues):
        issues.append(
            AgentCoreValidationIssue(
                source=source,
                code=f"{source}_not_ready",
                message=f"{source} gate is not ready.",
                metadata={"status": report.get("status")},
            )
        )


def _import_names(tree: ast.AST) -> tuple[tuple[str, int], ...]:
    names: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend((alias.name, int(node.lineno)) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append((node.module, int(node.lineno)))
    return tuple(names)


def _matches_forbidden_name(name: str, forbidden: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _extend_api_stability_issues(
    issues: list[AgentCoreValidationIssue],
    report: dict[str, Any],
) -> None:
    if report.get("ready") is True:
        return
    missing = tuple(str(name) for name in report.get("missing_stable_api") or ())
    for name in missing:
        issues.append(
            AgentCoreValidationIssue(
                source="api_stability",
                code="stable_api_missing",
                message=f"Stable API is missing: {name}",
                metadata={"api_name": name},
            )
        )
    if not missing:
        issues.append(
            AgentCoreValidationIssue(
                source="api_stability",
                code="api_stability_not_ready",
                message="API stability gate is not ready.",
                metadata={"status": report.get("status")},
            )
        )
