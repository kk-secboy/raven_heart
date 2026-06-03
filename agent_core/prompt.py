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

DEFAULT_PROMPT_TRIM_ORDER = (
    PromptBucketRole.TIMELINE_OPEN,
    PromptBucketRole.SEMI_DYNAMIC_1,
    PromptBucketRole.SEMI_DYNAMIC_2,
    PromptBucketRole.FROZEN,
    PromptBucketRole.DYNAMIC,
    PromptBucketRole.HIGH_STATIC,
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
class PromptTrimRule:
    role: PromptBucketRole
    order: int
    min_keep_bytes: int = 0
    preserve_head_ratio: float = 0.33
    protected: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_keep_bytes", max(0, int(self.min_keep_bytes)))
        ratio = min(0.9, max(0.1, float(self.preserve_head_ratio)))
        object.__setattr__(self, "preserve_head_ratio", ratio)

    def manifest(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "order": self.order,
            "min_keep_bytes": self.min_keep_bytes,
            "preserve_head_ratio": self.preserve_head_ratio,
            "protected": self.protected,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PromptTrimStep:
    role: PromptBucketRole
    original_bytes: int
    final_bytes: int
    overage_before: int
    protected: bool = False
    reason: str = ""

    def manifest(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "original_bytes": self.original_bytes,
            "final_bytes": self.final_bytes,
            "removed_bytes": max(0, self.original_bytes - self.final_bytes),
            "overage_before": self.overage_before,
            "protected": self.protected,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PromptTrimPlan:
    target_bytes: int
    original_bytes: int
    rules: tuple[PromptTrimRule, ...]
    marker: str = "\n[...trimmed...]\n"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-prompt-trim-plan/v1",
            "target_bytes": self.target_bytes,
            "original_bytes": self.original_bytes,
            "marker_bytes": len(self.marker.encode("utf-8")),
            "rules": [rule.manifest() for rule in self.rules],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PromptTrimResult:
    plan: PromptTrimPlan
    steps: tuple[PromptTrimStep, ...]
    final_bytes: int
    converged: bool

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-prompt-trim/v1",
            "target_bytes": self.plan.target_bytes,
            "original_bytes": self.plan.original_bytes,
            "final_bytes": self.final_bytes,
            "converged": self.converged,
            "plan": self.plan.manifest(),
            "steps": [step.manifest() for step in self.steps],
            "trimmed_roles": [
                {
                    "role": step.role.value,
                    "original_bytes": step.original_bytes,
                    "trimmed_bytes": step.final_bytes,
                }
                for step in self.steps
                if step.final_bytes < step.original_bytes
            ],
        }


def _default_min_keep_bytes(role: PromptBucketRole) -> int:
    if role == PromptBucketRole.HIGH_STATIC:
        return 4096
    if role == PromptBucketRole.DYNAMIC:
        return 256
    if role == PromptBucketRole.FROZEN:
        return 512
    return 0


def _default_preserve_head_ratio(role: PromptBucketRole) -> float:
    if role == PromptBucketRole.TIMELINE_OPEN:
        return 0.25
    if role == PromptBucketRole.DYNAMIC:
        return 0.1
    return 0.33


def _default_trim_reason(role: PromptBucketRole) -> str:
    reasons = {
        PromptBucketRole.TIMELINE_OPEN: "trim volatile timeline context first",
        PromptBucketRole.SEMI_DYNAMIC_1: "trim recall, skills, and semi-dynamic context after timeline",
        PromptBucketRole.SEMI_DYNAMIC_2: "trim task schema/examples after recall context",
        PromptBucketRole.FROZEN: "trim capability catalog only after dynamic context",
        PromptBucketRole.DYNAMIC: "preserve current task as long as possible",
        PromptBucketRole.HIGH_STATIC: "stable system rules are protected",
    }
    return reasons[role]


DEFAULT_PROMPT_TRIM_RULES = tuple(
    PromptTrimRule(
        role=role,
        order=index,
        min_keep_bytes=_default_min_keep_bytes(role),
        preserve_head_ratio=_default_preserve_head_ratio(role),
        protected=role == PromptBucketRole.HIGH_STATIC,
        reason=_default_trim_reason(role),
    )
    for index, role in enumerate(DEFAULT_PROMPT_TRIM_ORDER)
)


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

    def with_content(self, content: str, metadata: dict[str, Any] | None = None) -> "PromptBucket":
        return PromptBucket(
            role=self.role,
            content=content,
            cache_hint=self.cache_hint,
            metadata={**self.metadata, **dict(metadata or {})},
        )

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

    def trim_to_budget(
        self,
        max_bytes: int,
        *,
        trim_order: tuple[PromptBucketRole, ...] = DEFAULT_PROMPT_TRIM_ORDER,
        marker: str = "\n[...trimmed...]\n",
        rules: tuple[PromptTrimRule, ...] | None = None,
    ) -> "PromptIR":
        """Return a semantically trimmed prompt while preserving bucket order.

        The algorithm trims lower-priority dynamic context first, then gradually
        moves toward stable buckets only when the prompt is still over budget.
        It is deterministic and provider-neutral, so runtimes can use it before
        calling any concrete LLM client.
        """

        target = max(1, int(max_bytes))
        original_rendered = self.render()
        original_bytes = len(original_rendered.encode("utf-8"))
        if original_bytes <= target:
            return self

        plan = self.trim_plan(
            max_bytes,
            trim_order=trim_order,
            marker=marker,
            rules=rules,
        )
        by_role = {bucket.role: bucket for bucket in self.ordered_buckets()}
        steps: list[PromptTrimStep] = []
        for rule in plan.rules:
            role = rule.role
            current = PromptIR(
                buckets=tuple(by_role.get(item, PromptBucket(item)) for item in PROMPT_BUCKET_ORDER),
                metadata=dict(self.metadata),
            )
            current_bytes = len(current.render().encode("utf-8"))
            if current_bytes <= target:
                break
            bucket = by_role.get(role, PromptBucket(role))
            if not bucket.content:
                continue
            if rule.protected:
                steps.append(
                    PromptTrimStep(
                        role=role,
                        original_bytes=bucket.bytes,
                        final_bytes=bucket.bytes,
                        overage_before=current_bytes - target,
                        protected=True,
                        reason=rule.reason,
                    )
                )
                continue
            overage = current_bytes - target
            new_content = _trim_bucket_content(
                bucket.content,
                overage,
                marker=marker,
                min_keep_bytes=rule.min_keep_bytes,
                preserve_head_ratio=rule.preserve_head_ratio,
            )
            if new_content == bucket.content:
                continue
            new_bytes = len(new_content.encode("utf-8"))
            by_role[role] = bucket.with_content(
                new_content,
                {
                    "trimmed": True,
                    "original_bytes": bucket.bytes,
                    "trimmed_bytes": new_bytes,
                    "trim_reason": rule.reason,
                    "trim_min_keep_bytes": rule.min_keep_bytes,
                },
            )
            steps.append(
                PromptTrimStep(
                    role=role,
                    original_bytes=bucket.bytes,
                    final_bytes=new_bytes,
                    overage_before=overage,
                    reason=rule.reason,
                )
            )

        final_bytes = _rendered_bytes_by_role(by_role)
        trim_result = PromptTrimResult(
            plan=plan,
            steps=tuple(steps),
            final_bytes=final_bytes,
            converged=final_bytes <= target,
        )
        result = PromptIR(
            buckets=tuple(by_role.get(role, PromptBucket(role)) for role in PROMPT_BUCKET_ORDER),
            metadata={
                **self.metadata,
                "trim": trim_result.manifest(),
            },
        )
        return result

    def trim_plan(
        self,
        max_bytes: int,
        *,
        trim_order: tuple[PromptBucketRole, ...] = DEFAULT_PROMPT_TRIM_ORDER,
        marker: str = "\n[...trimmed...]\n",
        rules: tuple[PromptTrimRule, ...] | None = None,
    ) -> PromptTrimPlan:
        target = max(1, int(max_bytes))
        if rules is None:
            default_by_role = {rule.role: rule for rule in DEFAULT_PROMPT_TRIM_RULES}
            rules = tuple(
                _trim_rule_for_role(default_by_role, role, index)
                for index, role in enumerate(trim_order)
            )
        ordered = tuple(sorted(rules, key=lambda item: (item.order, item.role.value)))
        return PromptTrimPlan(
            target_bytes=target,
            original_bytes=len(self.render().encode("utf-8")),
            rules=ordered,
            marker=marker,
            metadata={
                "bucket_roles": [bucket.role.value for bucket in self.ordered_buckets()],
                "trim_order": [rule.role.value for rule in ordered],
            },
        )


def _trim_rule_for_role(
    defaults: dict[PromptBucketRole, PromptTrimRule],
    role: PromptBucketRole,
    index: int,
) -> PromptTrimRule:
    default = defaults.get(role)
    if default is None:
        return _fallback_trim_rule(role, index)
    return PromptTrimRule(
        role=role,
        order=index,
        min_keep_bytes=default.min_keep_bytes,
        preserve_head_ratio=default.preserve_head_ratio,
        protected=default.protected,
        reason=default.reason,
    )


def _fallback_trim_rule(role: PromptBucketRole, index: int) -> PromptTrimRule:
    return PromptTrimRule(
        role=role,
        order=index,
        min_keep_bytes=_default_min_keep_bytes(role),
        preserve_head_ratio=_default_preserve_head_ratio(role),
        protected=role == PromptBucketRole.HIGH_STATIC,
        reason=_default_trim_reason(role),
    )


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


def _rendered_bytes_by_role(by_role: dict[PromptBucketRole, PromptBucket]) -> int:
    rendered = "\n\n".join(
        part
        for bucket in tuple(by_role.get(role, PromptBucket(role)) for role in PROMPT_BUCKET_ORDER)
        if (part := bucket.render())
    )
    return len(rendered.encode("utf-8"))


def _trim_bucket_content(
    content: str,
    overage: int,
    *,
    marker: str,
    min_keep_bytes: int = 0,
    preserve_head_ratio: float = 0.33,
) -> str:
    text = content.strip()
    if not text:
        return ""
    marker_bytes = len(marker.encode("utf-8"))
    current_bytes = len(text.encode("utf-8"))
    target_bytes = current_bytes - max(1, overage) - marker_bytes
    if min_keep_bytes and current_bytes > min_keep_bytes:
        target_bytes = max(target_bytes, min_keep_bytes)
    if target_bytes <= 0:
        return ""
    if target_bytes >= current_bytes:
        return text
    head_budget = max(1, int(target_bytes * preserve_head_ratio))
    tail_budget = max(1, target_bytes - head_budget)
    head = _take_utf8_prefix(text, head_budget).rstrip()
    tail = _take_utf8_suffix(text, tail_budget).lstrip()
    trimmed = (head + marker + tail).strip()
    return trimmed if trimmed != marker.strip() else ""


def _take_utf8_prefix(text: str, max_bytes: int) -> str:
    used = 0
    parts: list[str] = []
    for char in text:
        char_bytes = len(char.encode("utf-8"))
        if parts and used + char_bytes > max_bytes:
            break
        if not parts and char_bytes > max_bytes:
            break
        parts.append(char)
        used += char_bytes
    return "".join(parts)


def _take_utf8_suffix(text: str, max_bytes: int) -> str:
    used = 0
    parts: list[str] = []
    for char in reversed(text):
        char_bytes = len(char.encode("utf-8"))
        if parts and used + char_bytes > max_bytes:
            break
        if not parts and char_bytes > max_bytes:
            break
        parts.append(char)
        used += char_bytes
    return "".join(reversed(parts))

