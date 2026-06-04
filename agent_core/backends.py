"""Provider-neutral storage backend manifests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


StorageBackendKind = Literal[
    "none",
    "in_memory",
    "sqlite",
    "markdown",
    "postgres",
    "object_storage",
    "vector",
    "graph",
    "product",
    "external",
    "custom",
]
StorageBackendRole = Literal[
    "memory",
    "journal",
    "tool_replay",
    "run_trace",
    "run_state",
    "planner_state",
    "artifact",
    "context_material",
    "approval",
    "policy_decision",
    "event_log",
]


@dataclass(frozen=True)
class StorageBackendSpec:
    """Prompt-safe backend metadata for SDK ports and runtime adapters."""

    role: StorageBackendRole
    kind: StorageBackendKind
    name: str = ""
    namespace: str = ""
    durable: bool = False
    inspectable: bool = False
    queryable: bool = False
    transactional: bool = False
    core_builtin: bool = True
    capabilities: tuple[str, ...] = ()
    location: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-storage-backend/v1",
            "role": self.role,
            "kind": self.kind,
            "name": self.name,
            "namespace": self.namespace,
            "durable": self.durable,
            "inspectable": self.inspectable,
            "queryable": self.queryable,
            "transactional": self.transactional,
            "core_builtin": self.core_builtin,
            "capabilities": list(self.capabilities),
            "location": self.location,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StorageBackendRequirement:
    """SDK-level preflight requirement for a store role.

    Runtime adapters can satisfy this with PostgreSQL, object storage, graph,
    vector, or product-specific stores without importing those drivers here.
    """

    role: StorageBackendRole
    allowed_kinds: tuple[StorageBackendKind, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    namespace: str = ""
    require_durable: bool | None = None
    require_queryable: bool | None = None
    require_transactional: bool | None = None
    require_inspectable: bool | None = None
    allow_external: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-storage-backend-requirement/v1",
            "role": self.role,
            "allowed_kinds": list(self.allowed_kinds),
            "required_capabilities": list(self.required_capabilities),
            "namespace": self.namespace,
            "require_durable": self.require_durable,
            "require_queryable": self.require_queryable,
            "require_transactional": self.require_transactional,
            "require_inspectable": self.require_inspectable,
            "allow_external": self.allow_external,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class StorageBackendCandidate:
    """One backend candidate evaluated against a requirement."""

    backend: StorageBackendSpec
    matched: bool
    reason: str = ""
    missing_capabilities: tuple[str, ...] = ()
    score: int = 0

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-storage-backend-candidate/v1",
            "matched": self.matched,
            "reason": self.reason,
            "score": self.score,
            "missing_capabilities": list(self.missing_capabilities),
            "backend": self.backend.manifest(),
        }


@dataclass(frozen=True)
class StorageBackendSelection:
    """Prompt-safe result of selecting a backend for one role."""

    requirement: StorageBackendRequirement
    candidates: tuple[StorageBackendCandidate, ...] = ()
    selected: StorageBackendCandidate | None = None

    @property
    def ready(self) -> bool:
        return self.selected is not None and self.selected.matched

    @property
    def status(self) -> str:
        return "ready" if self.ready else "missing"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-storage-backend-selection/v1",
            "status": self.status,
            "ready": self.ready,
            "requirement": self.requirement.manifest(),
            "selected": self.selected.manifest() if self.selected is not None else {},
            "candidate_count": len(self.candidates),
            "candidates": [candidate.manifest() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class StorageBackendPreflightReport:
    """Prompt-safe multi-role backend readiness report."""

    selections: tuple[StorageBackendSelection, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return bool(self.selections) and all(selection.ready for selection in self.selections)

    @property
    def status(self) -> str:
        return "ready" if self.ready else "blocked"

    @property
    def blocking_count(self) -> int:
        return sum(1 for selection in self.selections if not selection.ready)

    def manifest(self) -> dict[str, Any]:
        selected = tuple(selection for selection in self.selections if selection.ready)
        missing = tuple(selection for selection in self.selections if not selection.ready)
        selected_backends = tuple(
            selection.selected.backend
            for selection in selected
            if selection.selected is not None
        )
        return {
            "schema_version": "agent-core-storage-backend-preflight/v1",
            "status": self.status,
            "ready": self.ready,
            "requirement_count": len(self.selections),
            "selection_count": len(selected),
            "blocking_count": self.blocking_count,
            "required_roles": [selection.requirement.role for selection in self.selections],
            "selected_roles": [backend.role for backend in selected_backends],
            "selected_kinds": [backend.kind for backend in selected_backends],
            "missing_roles": [selection.requirement.role for selection in missing],
            "blocking_reasons": sorted(
                {
                    reason
                    for selection in missing
                    for candidate in selection.candidates
                    for reason in candidate.reason.split(",")
                    if reason and reason != "matched"
                }
            ),
            "selections": [selection.manifest() for selection in self.selections],
            "metadata": dict(self.metadata),
        }


class StorageBackendCatalog:
    """Registry and selector for SDK and runtime-owned data backends."""

    def __init__(self, backends: tuple[StorageBackendSpec | dict[str, Any], ...] = ()) -> None:
        self._backends: list[StorageBackendSpec] = []
        for backend in backends:
            self.register(backend)

    def register(self, backend: StorageBackendSpec | dict[str, Any]) -> StorageBackendSpec:
        spec = _coerce_backend_spec(backend)
        self._backends.append(spec)
        return spec

    def extend(self, backends: tuple[StorageBackendSpec | dict[str, Any], ...]) -> None:
        for backend in backends:
            self.register(backend)

    def all(self) -> tuple[StorageBackendSpec, ...]:
        return tuple(self._backends)

    def for_role(self, role: StorageBackendRole) -> tuple[StorageBackendSpec, ...]:
        return tuple(backend for backend in self._backends if backend.role == role)

    def select(
        self,
        requirement: StorageBackendRequirement | None = None,
        *,
        role: StorageBackendRole | None = None,
        allowed_kinds: tuple[StorageBackendKind, ...] = (),
        required_capabilities: tuple[str, ...] = (),
        namespace: str = "",
        require_durable: bool | None = None,
        require_queryable: bool | None = None,
        require_transactional: bool | None = None,
        require_inspectable: bool | None = None,
        allow_external: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> StorageBackendSelection:
        if requirement is None:
            if role is None:
                raise ValueError("storage backend selection requires a role")
            requirement = StorageBackendRequirement(
                role=role,
                allowed_kinds=allowed_kinds,
                required_capabilities=required_capabilities,
                namespace=namespace,
                require_durable=require_durable,
                require_queryable=require_queryable,
                require_transactional=require_transactional,
                require_inspectable=require_inspectable,
                allow_external=allow_external,
                metadata=dict(metadata or {}),
            )

        candidates = tuple(_evaluate_backend(backend, requirement) for backend in self.for_role(requirement.role))
        selected = _select_best_candidate(candidates)
        return StorageBackendSelection(requirement=requirement, candidates=candidates, selected=selected)

    def preflight(
        self,
        requirements: tuple[StorageBackendRequirement, ...],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> StorageBackendPreflightReport:
        return StorageBackendPreflightReport(
            selections=tuple(self.select(requirement) for requirement in requirements),
            metadata=dict(metadata or {}),
        )

    def manifest(self) -> dict[str, Any]:
        roles: dict[str, int] = {}
        kinds: dict[str, int] = {}
        external_count = 0
        for backend in self._backends:
            roles[backend.role] = roles.get(backend.role, 0) + 1
            kinds[backend.kind] = kinds.get(backend.kind, 0) + 1
            if not backend.core_builtin:
                external_count += 1
        return {
            "schema_version": "agent-core-storage-backend-catalog/v1",
            "backend_count": len(self._backends),
            "external_backend_count": external_count,
            "roles": roles,
            "kinds": kinds,
            "backends": [backend.manifest() for backend in self._backends],
        }


def storage_backend_manifest(
    *,
    role: StorageBackendRole,
    kind: StorageBackendKind,
    name: str = "",
    namespace: str = "",
    durable: bool | None = None,
    inspectable: bool | None = None,
    queryable: bool | None = None,
    transactional: bool | None = None,
    core_builtin: bool = True,
    capabilities: tuple[str, ...] = (),
    location: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    defaults = _backend_defaults(kind)
    return StorageBackendSpec(
        role=role,
        kind=kind,
        name=name,
        namespace=namespace,
        durable=defaults["durable"] if durable is None else durable,
        inspectable=defaults["inspectable"] if inspectable is None else inspectable,
        queryable=defaults["queryable"] if queryable is None else queryable,
        transactional=defaults["transactional"] if transactional is None else transactional,
        core_builtin=core_builtin,
        capabilities=tuple(capabilities),
        location=location,
        metadata=dict(metadata or {}),
    ).manifest()


def _coerce_backend_spec(backend: StorageBackendSpec | dict[str, Any]) -> StorageBackendSpec:
    if isinstance(backend, StorageBackendSpec):
        return backend
    kind = str(backend.get("kind") or "custom")
    defaults = _backend_defaults(kind)  # type: ignore[arg-type]
    return StorageBackendSpec(
        role=str(backend.get("role") or "memory"),  # type: ignore[arg-type]
        kind=kind,  # type: ignore[arg-type]
        name=str(backend.get("name") or ""),
        namespace=str(backend.get("namespace") or ""),
        durable=bool(backend.get("durable", defaults["durable"])),
        inspectable=bool(backend.get("inspectable", defaults["inspectable"])),
        queryable=bool(backend.get("queryable", defaults["queryable"])),
        transactional=bool(backend.get("transactional", defaults["transactional"])),
        core_builtin=bool(backend.get("core_builtin", True)),
        capabilities=tuple(str(item) for item in backend.get("capabilities", ()) or ()),
        location=str(backend.get("location") or ""),
        metadata=dict(backend.get("metadata") or {}),
    )


def _evaluate_backend(
    backend: StorageBackendSpec,
    requirement: StorageBackendRequirement,
) -> StorageBackendCandidate:
    reasons: list[str] = []
    score = 0

    if requirement.allowed_kinds and backend.kind not in requirement.allowed_kinds:
        reasons.append("kind_not_allowed")
    else:
        score += 4

    if requirement.namespace and backend.namespace != requirement.namespace:
        reasons.append("namespace_mismatch")
    elif requirement.namespace:
        score += 3

    if requirement.require_durable is not None and backend.durable is not requirement.require_durable:
        reasons.append("durable_mismatch")
    elif requirement.require_durable is not None:
        score += 2

    if requirement.require_queryable is not None and backend.queryable is not requirement.require_queryable:
        reasons.append("queryable_mismatch")
    elif requirement.require_queryable is not None:
        score += 2

    if requirement.require_transactional is not None and backend.transactional is not requirement.require_transactional:
        reasons.append("transactional_mismatch")
    elif requirement.require_transactional is not None:
        score += 2

    if requirement.require_inspectable is not None and backend.inspectable is not requirement.require_inspectable:
        reasons.append("inspectable_mismatch")
    elif requirement.require_inspectable is not None:
        score += 2

    if not requirement.allow_external and not backend.core_builtin:
        reasons.append("external_backend_not_allowed")

    capabilities = set(backend.capabilities)
    missing_capabilities = tuple(
        capability
        for capability in requirement.required_capabilities
        if capability not in capabilities
    )
    if missing_capabilities:
        reasons.append("missing_capabilities")
    else:
        score += len(requirement.required_capabilities)

    matched = not reasons
    return StorageBackendCandidate(
        backend=backend,
        matched=matched,
        reason="matched" if matched else ",".join(reasons),
        missing_capabilities=missing_capabilities,
        score=score,
    )


def _select_best_candidate(candidates: tuple[StorageBackendCandidate, ...]) -> StorageBackendCandidate | None:
    matched = [candidate for candidate in candidates if candidate.matched]
    if not matched:
        return None
    return sorted(
        matched,
        key=lambda item: (
            item.score,
            item.backend.durable,
            item.backend.transactional,
            item.backend.queryable,
            item.backend.name,
        ),
        reverse=True,
    )[0]


def _backend_defaults(kind: StorageBackendKind) -> dict[str, bool]:
    if kind == "none":
        return {
            "durable": False,
            "inspectable": False,
            "queryable": False,
            "transactional": False,
        }
    if kind == "in_memory":
        return {
            "durable": False,
            "inspectable": False,
            "queryable": True,
            "transactional": False,
        }
    if kind == "sqlite":
        return {
            "durable": True,
            "inspectable": False,
            "queryable": True,
            "transactional": True,
        }
    if kind == "markdown":
        return {
            "durable": True,
            "inspectable": True,
            "queryable": False,
            "transactional": False,
        }
    if kind == "object_storage":
        return {
            "durable": True,
            "inspectable": False,
            "queryable": False,
            "transactional": False,
        }
    if kind in {"postgres", "vector", "graph", "product", "external"}:
        return {
            "durable": True,
            "inspectable": False,
            "queryable": True,
            "transactional": kind in {"postgres", "product", "external"},
        }
    return {
        "durable": False,
        "inspectable": False,
        "queryable": False,
        "transactional": False,
    }
