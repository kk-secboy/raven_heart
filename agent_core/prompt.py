"""Prompt IR with Yaklang-style stable/dynamic bucket boundaries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PromptBucketRole(StrEnum):
    HIGH_STATIC = "high_static"
    FROZEN = "frozen"
    SEMI_DYNAMIC_1 = "semi_dynamic_1"
    SEMI_DYNAMIC_2 = "semi_dynamic_2"
    TIMELINE_OPEN = "timeline_open"
    DYNAMIC = "dynamic"


PROMPT_BUCKET_ORDER = (
    PromptBucketRole.HIGH_STATIC,
    PromptBucketRole.FROZEN,
    PromptBucketRole.SEMI_DYNAMIC_1,
    PromptBucketRole.SEMI_DYNAMIC_2,
    PromptBucketRole.TIMELINE_OPEN,
    PromptBucketRole.DYNAMIC,
)


@dataclass(frozen=True)
class CacheHint:
    cacheable: bool
    policy: str
    provider_marker: str = ""


def default_cache_hint(role: PromptBucketRole) -> CacheHint:
    if role in {PromptBucketRole.HIGH_STATIC, PromptBucketRole.FROZEN}:
        return CacheHint(True, "stable_cacheable_prefix", "cache_prefix")
    if role in {PromptBucketRole.SEMI_DYNAMIC_1, PromptBucketRole.SEMI_DYNAMIC_2}:
        return CacheHint(True, "semi_dynamic_refresh_on_task_change", "cache_semistatic")
    return CacheHint(False, "dynamic_refresh_each_turn", "")


def estimate_tokens(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 4) if text else 0


@dataclass(frozen=True)
class PromptBucket:
    role: PromptBucketRole
    content: str = ""
    cache_hint: CacheHint | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bytes(self) -> int:
        return len(self.content.encode("utf-8"))

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.content)

    @property
    def sha256(self) -> str:
        if not self.content:
            return ""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def render(self) -> str:
        body = self.content.strip()
        if not body:
            return ""
        return f'<prompt_materials role="{self.role.value}">\n{body}\n</prompt_materials>'

    def manifest(self) -> dict[str, Any]:
        hint = self.cache_hint or default_cache_hint(self.role)
        return {
            "role": self.role.value,
            "included": bool(self.content),
            "bytes": self.bytes,
            "estimated_tokens": self.estimated_tokens,
            "sha256": self.sha256,
            "cache_hint": {
                "cacheable": hint.cacheable,
                "policy": hint.policy,
                "provider_marker": hint.provider_marker,
            },
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PromptIR:
    buckets: tuple[PromptBucket, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_parts(
        cls,
        *,
        high_static: str = "",
        frozen: str = "",
        semi_dynamic_1: str = "",
        semi_dynamic_2: str = "",
        timeline_open: str = "",
        dynamic: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "PromptIR":
        content_by_role = {
            PromptBucketRole.HIGH_STATIC: high_static,
            PromptBucketRole.FROZEN: frozen,
            PromptBucketRole.SEMI_DYNAMIC_1: semi_dynamic_1,
            PromptBucketRole.SEMI_DYNAMIC_2: semi_dynamic_2,
            PromptBucketRole.TIMELINE_OPEN: timeline_open,
            PromptBucketRole.DYNAMIC: dynamic,
        }
        return cls(
            buckets=tuple(
                PromptBucket(role=role, content=content_by_role[role])
                for role in PROMPT_BUCKET_ORDER
            ),
            metadata=dict(metadata or {}),
        )

    def bucket(self, role: PromptBucketRole) -> PromptBucket:
        for bucket in self.buckets:
            if bucket.role == role:
                return bucket
        return PromptBucket(role=role)

    def render(self) -> str:
        return "\n\n".join(part for bucket in self.ordered_buckets() if (part := bucket.render()))

    def ordered_buckets(self) -> tuple[PromptBucket, ...]:
        by_role = {bucket.role: bucket for bucket in self.buckets}
        return tuple(by_role.get(role, PromptBucket(role=role)) for role in PROMPT_BUCKET_ORDER)

    def manifest(self) -> dict[str, Any]:
        rendered = self.render()
        bucket_manifests = [bucket.manifest() for bucket in self.ordered_buckets()]
        return {
            "schema_version": "agent-core-prompt-ir/v1",
            "prompt_bytes": len(rendered.encode("utf-8")),
            "prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest()
            if rendered
            else "",
            "buckets": bucket_manifests,
            "buckets_by_role": {item["role"]: item for item in bucket_manifests},
            "metadata": dict(self.metadata),
        }


class PromptAssembler:
    """Small helper for building PromptIR incrementally."""

    def __init__(self) -> None:
        self._parts: dict[PromptBucketRole, list[str]] = {role: [] for role in PROMPT_BUCKET_ORDER}

    def add(self, role: PromptBucketRole, content: str) -> "PromptAssembler":
        text = str(content or "").strip()
        if text:
            self._parts[role].append(text)
        return self

    def build(self, metadata: dict[str, Any] | None = None) -> PromptIR:
        return PromptIR(
            buckets=tuple(
                PromptBucket(role=role, content="\n\n".join(self._parts[role]))
                for role in PROMPT_BUCKET_ORDER
            ),
            metadata=dict(metadata or {}),
        )

