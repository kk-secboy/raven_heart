"""Lifecycle hook contracts for runtime-observable agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal, Protocol


LifecycleEventType = Literal["run_starting", "run_completed", "run_failed"]
LifecycleHookStatus = Literal["completed", "failed"]


@dataclass(frozen=True)
class AgentLifecycleEvent:
    """Runtime-facing lifecycle event with prompt-safe manifests."""

    type: LifecycleEventType
    session_name: str = ""
    run_id: str = ""
    task: str = ""
    status: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-lifecycle-event/v1",
            "type": self.type,
            "session_name": self.session_name,
            "run_id": self.run_id,
            "task_bytes": len(self.task.encode("utf-8")),
            "task_sha256": sha256(self.task.encode("utf-8")).hexdigest()
            if self.task
            else "",
            "status": self.status,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class AgentLifecycleHookPort(Protocol):
    async def on_lifecycle_event(self, event: AgentLifecycleEvent) -> None:
        """Handle one lifecycle event."""


@dataclass(frozen=True)
class AgentLifecycleHookRecord:
    hook_name: str
    event_type: LifecycleEventType
    status: LifecycleHookStatus
    event: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-lifecycle-hook-record/v1",
            "hook_name": self.hook_name,
            "event_type": self.event_type,
            "status": self.status,
            "event": dict(self.event),
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class _HookMount:
    name: str
    hook: AgentLifecycleHookPort
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        hook_manifest = getattr(self.hook, "manifest", None)
        return {
            "name": self.name,
            "hook": hook_manifest() if callable(hook_manifest) else {},
            "metadata": dict(self.metadata),
        }


class AgentLifecycleHookCenter:
    """Small hook bus owned by the SDK runner, with auditable failures."""

    def __init__(self, *, fail_fast: bool = False) -> None:
        self.fail_fast = fail_fast
        self._hooks: dict[str, _HookMount] = {}
        self.records: list[AgentLifecycleHookRecord] = []

    def register(
        self,
        name: str,
        hook: AgentLifecycleHookPort,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not name:
            raise ValueError("lifecycle hook name is required")
        self._hooks[name] = _HookMount(name=name, hook=hook, metadata=dict(metadata or {}))

    def unregister(self, name: str) -> bool:
        return self._hooks.pop(name, None) is not None

    async def emit(self, event: AgentLifecycleEvent) -> None:
        for mount in self._ordered_hooks():
            try:
                await mount.hook.on_lifecycle_event(event)
            except Exception as exc:
                record = AgentLifecycleHookRecord(
                    hook_name=mount.name,
                    event_type=event.type,
                    status="failed",
                    event=event.manifest(),
                    error=str(exc),
                    metadata=dict(mount.metadata),
                )
                self.records.append(record)
                if self.fail_fast:
                    raise
                continue
            self.records.append(
                AgentLifecycleHookRecord(
                    hook_name=mount.name,
                    event_type=event.type,
                    status="completed",
                    event=event.manifest(),
                    metadata=dict(mount.metadata),
                )
            )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-lifecycle-hook-center/v1",
            "fail_fast": self.fail_fast,
            "hook_count": len(self._hooks),
            "hooks": [mount.manifest() for mount in self._ordered_hooks()],
            "record_count": len(self.records),
            "records": [record.manifest() for record in self.records],
        }

    def _ordered_hooks(self) -> tuple[_HookMount, ...]:
        return tuple(self._hooks[name] for name in sorted(self._hooks))


class NullLifecycleHooks(AgentLifecycleHookCenter):
    """Default no-op lifecycle hook center."""

    def __init__(self) -> None:
        super().__init__(fail_fast=False)
