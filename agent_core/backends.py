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
