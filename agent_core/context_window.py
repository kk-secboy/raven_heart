"""Unified context window audit reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContextWindowPolicy:
    """Policy for evaluating a prompt context window."""

    max_prompt_bytes: int | None = None
    max_excluded_contexts: int | None = None
    max_trimmed_contexts: int | None = None
    required_sources: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-context-window-policy/v1",
            "max_prompt_bytes": self.max_prompt_bytes,
            "max_excluded_contexts": self.max_excluded_contexts,
            "max_trimmed_contexts": self.max_trimmed_contexts,
            "required_sources": list(self.required_sources),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ContextWindowEntry:
    """One context material/injection decision after selection and trimming."""

    name: str
    source: str = ""
    target: str = ""
    status: str = ""
    selected: bool = False
    included: bool = False
    trimmed: bool = False
    rank: int = 0
    score: float = 0.0
    original_bytes: int = 0
    final_bytes: int = 0
    sha256: str = ""
    reason: str = ""
    stages: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-context-window-entry/v1",
            "name": self.name,
            "source": self.source,
            "target": self.target,
            "status": self.status,
            "selected": self.selected,
            "included": self.included,
            "trimmed": self.trimmed,
            "rank": self.rank,
            "score": self.score,
            "original_bytes": self.original_bytes,
            "final_bytes": self.final_bytes,
            "sha256": self.sha256,
            "reason": self.reason,
            "stages": list(self.stages),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ContextWindowIssue:
    """One context window policy issue."""

    code: str
    message: str
    severity: str = "error"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-context-window-issue/v1",
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ContextWindowReport:
    """Prompt-safe context window report derived from prompt/trace manifests."""

    status: str
    entries: tuple[ContextWindowEntry, ...] = ()
    policy: ContextWindowPolicy = field(default_factory=ContextWindowPolicy)
    prompt_bytes: int = 0
    prompt_sha256: str = ""
    bucket_bytes: dict[str, int] = field(default_factory=dict)
    prompt_budget: dict[str, Any] = field(default_factory=dict)
    prompt_bucket_budget: dict[str, Any] = field(default_factory=dict)
    prompt_semantic_trim: dict[str, Any] = field(default_factory=dict)
    prompt_trim: dict[str, Any] = field(default_factory=dict)
    issues: tuple[ContextWindowIssue, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    def manifest(self) -> dict[str, Any]:
        selected = tuple(entry for entry in self.entries if entry.selected)
        included = tuple(entry for entry in self.entries if entry.included)
        trimmed = tuple(entry for entry in self.entries if entry.trimmed)
        excluded = tuple(entry for entry in self.entries if not entry.included)
        return {
            "schema_version": "agent-core-context-window-report/v1",
            "status": self.status,
            "ready": self.ready,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "prompt_bytes": self.prompt_bytes,
            "prompt_sha256": self.prompt_sha256,
            "bucket_bytes": dict(self.bucket_bytes),
            "entry_count": len(self.entries),
            "selected_count": len(selected),
            "included_count": len(included),
            "trimmed_count": len(trimmed),
            "excluded_count": len(excluded),
            "sources": _count_entries(self.entries, "source"),
            "targets": _count_entries(self.entries, "target"),
            "statuses": _count_entries(self.entries, "status"),
            "entries": [entry.manifest() for entry in self.entries],
            "policy": self.policy.manifest(),
            "prompt_budget": dict(self.prompt_budget),
            "prompt_bucket_budget": dict(self.prompt_bucket_budget),
            "prompt_semantic_trim": dict(self.prompt_semantic_trim),
            "prompt_trim": dict(self.prompt_trim),
            "issues": [issue.manifest() for issue in self.issues],
            "metadata": dict(self.metadata),
        }


class ContextWindowBuilder:
    """Build a unified context window report from trace or prompt manifests."""

    def __init__(self, *, policy: ContextWindowPolicy | None = None) -> None:
        self.policy = policy or ContextWindowPolicy()

    def from_trace(
        self,
        trace: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ContextWindowReport:
        prompt = trace.get("prompt") if isinstance(trace.get("prompt"), dict) else {}
        return self.from_prompt(
            prompt,
            context_material_selection=trace.get("context_material_selection"),
            context_injections=trace.get("context_injections"),
            prompt_budget=trace.get("prompt_budget"),
            prompt_bucket_budget=trace.get("prompt_bucket_budget"),
            prompt_semantic_trim=trace.get("prompt_semantic_trim"),
            prompt_trim=trace.get("prompt", {}).get("metadata", {}).get("trim")
            if isinstance(trace.get("prompt"), dict)
            and isinstance(trace.get("prompt", {}).get("metadata"), dict)
            else {},
            metadata={"source": "trace", **dict(metadata or {})},
        )

    def from_prompt(
        self,
        prompt: dict[str, Any],
        *,
        context_material_selection: Any = None,
        context_injections: Any = None,
        prompt_budget: Any = None,
        prompt_bucket_budget: Any = None,
        prompt_semantic_trim: Any = None,
        prompt_trim: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> ContextWindowReport:
        selection = _safe_dict(context_material_selection)
        injections = _safe_dict(context_injections)
        prompt_metadata = _safe_dict(prompt.get("metadata"))
        selection = selection or _safe_dict(prompt_metadata.get("context_material_selection"))
        injections = injections or {
            "injections": list(prompt_metadata.get("context_injections") or ())
        }
        bucket_budget = _safe_dict(prompt_bucket_budget) or _safe_dict(
            prompt_metadata.get("bucket_budget")
        )
        semantic_trim = _safe_dict(prompt_semantic_trim) or _safe_dict(
            prompt_metadata.get("semantic_trim")
        )
        trim = _safe_dict(prompt_trim) or _safe_dict(prompt_metadata.get("trim"))
        budget = _safe_dict(prompt_budget) or _safe_dict(prompt_metadata.get("prompt_budget"))
        entries = _context_window_entries(selection=selection, injections=injections)
        prompt_bytes = int(prompt.get("prompt_bytes") or 0)
        bucket_bytes = _bucket_bytes(prompt)
        issues = _context_window_issues(
            entries=entries,
            policy=self.policy,
            prompt_bytes=prompt_bytes,
        )
        status = "blocked" if any(issue.severity == "error" for issue in issues) else "ready"
        return ContextWindowReport(
            status=status,
            entries=entries,
            policy=self.policy,
            prompt_bytes=prompt_bytes,
            prompt_sha256=str(prompt.get("prompt_sha256") or ""),
            bucket_bytes=bucket_bytes,
            prompt_budget=budget,
            prompt_bucket_budget=bucket_budget,
            prompt_semantic_trim=semantic_trim,
            prompt_trim=trim,
            issues=issues,
            metadata={"scenario": "context_window", **dict(metadata or {})},
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-context-window-builder/v1",
            "policy": self.policy.manifest(),
        }


def build_context_window_report(
    trace_or_prompt: dict[str, Any],
    *,
    policy: ContextWindowPolicy | None = None,
    metadata: dict[str, Any] | None = None,
) -> ContextWindowReport:
    """Build a context window report from a run trace or prompt manifest."""

    builder = ContextWindowBuilder(policy=policy)
    if trace_or_prompt.get("schema_version") == "agent-core-run-trace-bundle/v1":
        return builder.from_trace(trace_or_prompt, metadata=metadata)
    return builder.from_prompt(trace_or_prompt, metadata=metadata)


def _context_window_entries(
    *,
    selection: dict[str, Any],
    injections: dict[str, Any],
) -> tuple[ContextWindowEntry, ...]:
    by_name: dict[str, dict[str, Any]] = {}
    for item in _dict_items(selection.get("selections")):
        name = str(item.get("name") or "")
        if not name:
            continue
        by_name[name] = {
            "name": name,
            "source": str(item.get("metadata", {}).get("source") or item.get("role") or ""),
            "target": str(item.get("target") or ""),
            "status": str(item.get("status") or ""),
            "selected": bool(item.get("selected")),
            "included": False,
            "trimmed": False,
            "rank": _safe_int(item.get("rank")),
            "score": _safe_float(item.get("score")),
            "original_bytes": _safe_int(item.get("bytes")),
            "final_bytes": 0,
            "sha256": str(item.get("sha256") or ""),
            "reason": str(item.get("reason") or ""),
            "stages": ["selected" if item.get("selected") is True else "dropped"],
            "metadata": dict(item.get("metadata") or {})
            if isinstance(item.get("metadata"), dict)
            else {},
        }
    for item in _dict_items(injections.get("injections")):
        name = str(item.get("name") or "")
        if not name:
            continue
        current = by_name.setdefault(
            name,
            {
                "name": name,
                "source": str(item.get("source") or ""),
                "target": str(item.get("target") or ""),
                "status": "",
                "selected": False,
                "included": False,
                "trimmed": False,
                "rank": 0,
                "score": 0.0,
                "original_bytes": 0,
                "final_bytes": 0,
                "sha256": str(item.get("sha256") or ""),
                "reason": "",
                "stages": [],
                "metadata": {},
            },
        )
        current["source"] = current.get("source") or str(item.get("source") or "")
        current["target"] = current.get("target") or str(item.get("target") or "")
        current["status"] = str(item.get("status") or current.get("status") or "")
        current["included"] = bool(item.get("included"))
        current["trimmed"] = bool(item.get("trimmed"))
        current["original_bytes"] = max(
            _safe_int(current.get("original_bytes")),
            _safe_int(item.get("original_bytes") or item.get("bytes")),
        )
        current["final_bytes"] = _safe_int(item.get("final_bytes"))
        current["sha256"] = current.get("sha256") or str(item.get("sha256") or "")
        reason = str(item.get("excluded_reason") or "")
        current["reason"] = reason or str(current.get("reason") or "")
        stages = list(current.get("stages") or ())
        stages.append("injected" if item.get("included") is True else "excluded")
        if item.get("trimmed") is True:
            stages.append("trimmed")
        current["stages"] = sorted(set(stages))
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            current["metadata"] = {**dict(current.get("metadata") or {}), **metadata}
    entries = tuple(ContextWindowEntry(**item) for item in by_name.values())
    return tuple(sorted(entries, key=_entry_sort_key))


def _context_window_issues(
    *,
    entries: tuple[ContextWindowEntry, ...],
    policy: ContextWindowPolicy,
    prompt_bytes: int,
) -> tuple[ContextWindowIssue, ...]:
    issues: list[ContextWindowIssue] = []
    if policy.max_prompt_bytes is not None and prompt_bytes > policy.max_prompt_bytes:
        issues.append(
            ContextWindowIssue(
                code="prompt_bytes_exceeded",
                message="Prompt context window exceeds the configured byte budget.",
                metadata={"prompt_bytes": prompt_bytes, "max_prompt_bytes": policy.max_prompt_bytes},
            )
        )
    excluded_count = sum(1 for entry in entries if not entry.included)
    if policy.max_excluded_contexts is not None and excluded_count > policy.max_excluded_contexts:
        issues.append(
            ContextWindowIssue(
                code="excluded_context_count_exceeded",
                message="Context window excluded too many context entries.",
                metadata={
                    "excluded_count": excluded_count,
                    "max_excluded_contexts": policy.max_excluded_contexts,
                },
            )
        )
    trimmed_count = sum(1 for entry in entries if entry.trimmed)
    if policy.max_trimmed_contexts is not None and trimmed_count > policy.max_trimmed_contexts:
        issues.append(
            ContextWindowIssue(
                code="trimmed_context_count_exceeded",
                message="Context window trimmed too many context entries.",
                metadata={
                    "trimmed_count": trimmed_count,
                    "max_trimmed_contexts": policy.max_trimmed_contexts,
                },
            )
        )
    included_sources = {entry.source for entry in entries if entry.included and entry.source}
    for source in policy.required_sources:
        if source not in included_sources:
            issues.append(
                ContextWindowIssue(
                    code="required_context_source_missing",
                    message=f"Required context source is missing from included context: {source}",
                    metadata={"source": source, "included_sources": sorted(included_sources)},
                )
            )
    return tuple(issues)


def _bucket_bytes(prompt: dict[str, Any]) -> dict[str, int]:
    buckets = prompt.get("buckets")
    if not isinstance(buckets, list):
        return {}
    return {
        str(item.get("role") or ""): _safe_int(item.get("bytes"))
        for item in buckets
        if isinstance(item, dict) and item.get("role")
    }


def _dict_items(value: Any) -> tuple[dict[str, Any], ...]:
    if isinstance(value, dict):
        return (dict(value),)
    if isinstance(value, (list, tuple)):
        return tuple(dict(item) for item in value if isinstance(item, dict))
    return ()


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _count_entries(entries: tuple[ContextWindowEntry, ...], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        value = str(getattr(entry, field) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _entry_sort_key(entry: ContextWindowEntry) -> tuple[int, str, str]:
    return (entry.rank if entry.rank else 9999, entry.target, entry.name)
