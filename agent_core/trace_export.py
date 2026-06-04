"""Prompt-safe trace export bundles for SDK observability and replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from agent_core.redaction import RedactionPolicy, redact_payload


@dataclass(frozen=True)
class TraceExportPolicy:
    """Policy for turning trace manifests into prompt-safe export bundles."""

    name: str = "prompt_safe_trace_export"
    redact: bool = True
    redaction_policy: RedactionPolicy = field(default_factory=lambda: _trace_export_redaction_policy())
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-export-policy/v1",
            "name": self.name,
            "redact": self.redact,
            "redaction_policy": self.redaction_policy.manifest(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceExportRecord:
    """One redacted manifest inside an export bundle."""

    name: str
    payload: dict[str, Any]
    original_bytes: int = 0
    exported_bytes: int = 0
    original_sha256: str = ""
    exported_sha256: str = ""
    schema_version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-export-record/v1",
            "name": self.name,
            "payload_schema_version": self.schema_version,
            "original_bytes": self.original_bytes,
            "exported_bytes": self.exported_bytes,
            "original_sha256": self.original_sha256,
            "exported_sha256": self.exported_sha256,
            "payload": dict(self.payload),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TraceExportBundle:
    """Prompt-safe bundle of trace/eval manifests ready for host export."""

    records: tuple[TraceExportRecord, ...]
    policy: TraceExportPolicy = field(default_factory=TraceExportPolicy)
    redaction: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def record_count(self) -> int:
        return len(self.records)

    @property
    def redacted_count(self) -> int:
        return int(self.redaction.get("redacted_count") or 0)

    def manifest(self) -> dict[str, Any]:
        original_bytes = sum(record.original_bytes for record in self.records)
        exported_bytes = sum(record.exported_bytes for record in self.records)
        return {
            "schema_version": "agent-core-trace-export-bundle/v1",
            "prompt_safe": True,
            "record_count": self.record_count,
            "original_bytes": original_bytes,
            "exported_bytes": exported_bytes,
            "redacted_count": self.redacted_count,
            "records": [record.manifest() for record in self.records],
            "policy": self.policy.manifest(),
            "redaction": dict(self.redaction),
            "metadata": dict(self.metadata),
        }


class TraceExportBuilder:
    """Build prompt-safe trace export bundles without runtime dependencies."""

    def __init__(self, *, policy: TraceExportPolicy | None = None) -> None:
        self.policy = policy or TraceExportPolicy()

    def export(
        self,
        traces: dict[str, dict[str, Any]] | tuple[dict[str, Any], ...] | list[dict[str, Any]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TraceExportBundle:
        raw_records = _normalize_traces(traces)
        payloads = {name: dict(payload) for name, payload in raw_records}
        original_sizes = {
            name: _payload_measurement(payload) for name, payload in raw_records
        }
        redaction = (
            redact_payload(
                payloads,
                policy=self.policy.redaction_policy,
                metadata={"export_policy": self.policy.name},
            )
            if self.policy.redact
            else redact_payload(
                payloads,
                policy=RedactionPolicy(
                    sensitive_key_patterns=(),
                    redact_secret_values=False,
                    max_string_bytes=2**31 - 1,
                ),
                metadata={"export_policy": self.policy.name, "redaction_disabled": True},
            )
        )
        exported_payloads = redaction.payload if isinstance(redaction.payload, dict) else {}
        records: list[TraceExportRecord] = []
        for name, original_payload in raw_records:
            exported_payload = exported_payloads.get(name)
            if not isinstance(exported_payload, dict):
                exported_payload = {}
            exported_bytes, exported_sha = _payload_measurement(exported_payload)
            original_bytes, original_sha = original_sizes[name]
            records.append(
                TraceExportRecord(
                    name=name,
                    payload=exported_payload,
                    original_bytes=original_bytes,
                    exported_bytes=exported_bytes,
                    original_sha256=original_sha,
                    exported_sha256=exported_sha,
                    schema_version=str(original_payload.get("schema_version") or ""),
                )
            )
        return TraceExportBundle(
            records=tuple(records),
            policy=self.policy,
            redaction=_redaction_audit(redaction.manifest()),
            metadata={"scenario": "trace_export", **dict(metadata or {})},
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-trace-export-builder/v1",
            "policy": self.policy.manifest(),
        }


def export_trace_bundle(
    traces: dict[str, dict[str, Any]] | tuple[dict[str, Any], ...] | list[dict[str, Any]],
    *,
    policy: TraceExportPolicy | None = None,
    metadata: dict[str, Any] | None = None,
) -> TraceExportBundle:
    """Build a prompt-safe trace export bundle from trace/eval manifests."""

    return TraceExportBuilder(policy=policy).export(traces, metadata=metadata)


def _trace_export_redaction_policy() -> RedactionPolicy:
    patterns = tuple(
        pattern for pattern in RedactionPolicy().sensitive_key_patterns if pattern != "session"
    )
    return RedactionPolicy(sensitive_key_patterns=patterns)


def _normalize_traces(
    traces: dict[str, dict[str, Any]] | tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> tuple[tuple[str, dict[str, Any]], ...]:
    if isinstance(traces, dict):
        return tuple(
            (str(name), dict(payload))
            for name, payload in traces.items()
            if isinstance(payload, dict)
        )
    records: list[tuple[str, dict[str, Any]]] = []
    for index, payload in enumerate(traces, start=1):
        if not isinstance(payload, dict):
            continue
        run = payload.get("run") if isinstance(payload.get("run"), dict) else {}
        name = str(run.get("run_id") or payload.get("name") or f"trace-{index}")
        records.append((name, dict(payload)))
    return tuple(records)


def _redaction_audit(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "agent-core-trace-export-redaction-audit/v1",
        "redacted": bool(manifest.get("redacted")),
        "redacted_count": int(manifest.get("redacted_count") or 0),
        "decisions": list(manifest.get("decisions") or ()),
        "metadata": dict(manifest.get("metadata") or {}),
    }


def _payload_measurement(payload: dict[str, Any]) -> tuple[int, str]:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return len(raw), hashlib.sha256(raw).hexdigest()
