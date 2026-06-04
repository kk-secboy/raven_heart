"""Prompt-safe redaction contracts for SDK manifests."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


_DEFAULT_SENSITIVE_KEY_PATTERNS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "bearer",
    "client_secret",
    "credential",
    "jwt",
    "password",
    "private_key",
    "secret",
    "session",
    "token",
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)\b("
    r"bearer\s+[a-z0-9._\-]{12,}|"
    r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*[^\s,;]{8,}|"
    r"sk-[a-z0-9_\-]{16,}"
    r")\b"
)


@dataclass(frozen=True)
class RedactionDecision:
    """One prompt-safe redaction audit entry."""

    path: str
    reason: str
    original_type: str
    original_bytes: int = 0
    digest: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-redaction-decision/v1",
            "path": self.path,
            "reason": self.reason,
            "original_type": self.original_type,
            "original_bytes": self.original_bytes,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class RedactionResult:
    """Prompt-safe result of applying a redaction policy."""

    payload: Any
    decisions: tuple[RedactionDecision, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def redacted_count(self) -> int:
        return len(self.decisions)

    @property
    def redacted(self) -> bool:
        return bool(self.decisions)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-redaction-result/v1",
            "redacted": self.redacted,
            "redacted_count": self.redacted_count,
            "payload": self.payload,
            "decisions": [decision.manifest() for decision in self.decisions],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RedactionPolicy:
    """Recursive prompt-safe redaction policy."""

    sensitive_key_patterns: tuple[str, ...] = _DEFAULT_SENSITIVE_KEY_PATTERNS
    redact_secret_values: bool = True
    max_string_bytes: int = 4096
    replacement: str = "[REDACTED]"
    hash_prefix: str = "sha256:"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sensitive_key_patterns",
            tuple(str(item).lower() for item in self.sensitive_key_patterns if str(item)),
        )
        object.__setattr__(self, "max_string_bytes", max(1, int(self.max_string_bytes)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def redact(self, payload: Any, *, metadata: dict[str, Any] | None = None) -> RedactionResult:
        decisions: list[RedactionDecision] = []
        redacted = self._redact_value(payload, path="$", decisions=decisions, key="")
        return RedactionResult(
            payload=redacted,
            decisions=tuple(decisions),
            metadata={"policy": self.manifest(), **dict(metadata or {})},
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-redaction-policy/v1",
            "sensitive_key_patterns": list(self.sensitive_key_patterns),
            "redact_secret_values": self.redact_secret_values,
            "max_string_bytes": self.max_string_bytes,
            "replacement": self.replacement,
            "hash_prefix": self.hash_prefix,
            "metadata": dict(self.metadata),
        }

    def _redact_value(
        self,
        value: Any,
        *,
        path: str,
        decisions: list[RedactionDecision],
        key: str,
    ) -> Any:
        if self._is_sensitive_key(key):
            decisions.append(self._decision(path, "sensitive_key", value))
            return self._replacement_for(value)
        if isinstance(value, dict):
            return {
                str(item_key): self._redact_value(
                    item_value,
                    path=f"{path}.{item_key}",
                    decisions=decisions,
                    key=str(item_key),
                )
                for item_key, item_value in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                self._redact_value(
                    item,
                    path=f"{path}[{index}]",
                    decisions=decisions,
                    key="",
                )
                for index, item in enumerate(value)
            ]
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            if len(encoded) > self.max_string_bytes:
                decisions.append(self._decision(path, "string_too_large", value))
                return self._replacement_for(value)
            if self.redact_secret_values and _SECRET_VALUE_RE.search(value):
                decisions.append(self._decision(path, "secret_value", value))
                return self._replacement_for(value)
        return value

    def _is_sensitive_key(self, key: str) -> bool:
        lowered = str(key or "").lower().replace("-", "_")
        return any(pattern in lowered for pattern in self.sensitive_key_patterns)

    def _replacement_for(self, value: Any) -> dict[str, Any]:
        return {
            "redacted": True,
            "replacement": self.replacement,
            "original_type": type(value).__name__,
            "original_bytes": _value_bytes(value),
            "digest": self.hash_prefix + _value_digest(value),
        }

    def _decision(self, path: str, reason: str, value: Any) -> RedactionDecision:
        return RedactionDecision(
            path=path,
            reason=reason,
            original_type=type(value).__name__,
            original_bytes=_value_bytes(value),
            digest=self.hash_prefix + _value_digest(value),
        )


def redact_payload(
    payload: Any,
    *,
    policy: RedactionPolicy | None = None,
    metadata: dict[str, Any] | None = None,
) -> RedactionResult:
    """Redact a payload using the default prompt-safe SDK policy."""

    return (policy or RedactionPolicy()).redact(payload, metadata=metadata)


def _value_bytes(value: Any) -> int:
    return len(str(value).encode("utf-8"))


def _value_digest(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
