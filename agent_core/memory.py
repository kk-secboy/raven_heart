"""Memory and retrieval protocol types."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from agent_core.backends import StorageBackendKind, storage_backend_manifest
from agent_core.embeddings import EmbeddingProviderPort, rank_semantic_documents
from agent_core.search import SearchDocument, rank_documents


MemoryBackendKind = Literal[
    "in_memory",
    "sqlite",
    "markdown",
    "postgres",
    "vector",
    "graph",
    "product",
    "external",
    "custom",
]
MemoryQueryMode = Literal["keyword", "semantic", "vector", "graph", "hybrid"]
MemoryRecallIntent = Literal["generic", "fact_check", "advice", "emotional", "brainstorm"]
MemoryInjectionRoute = Literal[
    "must_aware",
    "action_tips",
    "reliability_warning",
    "emotional_context",
    "connection_links",
    "context",
]
CORE_PACT_SCORE_KEYS = ("c", "o", "r", "e", "p", "a", "t")
CORE_PACT_METADATA_KEYS = {
    "c": ("c_score", "connectivity", "connection", "c"),
    "o": ("o_score", "origin", "reliability", "trust", "confidence", "o"),
    "r": ("r_score", "relevance", "r"),
    "e": ("e_score", "emotion", "emotional", "e"),
    "p": ("p_score", "preference", "constraint", "p"),
    "a": ("a_score", "actionability", "actionable", "a"),
    "t": ("t_score", "timeliness", "freshness", "recent", "t"),
}


@dataclass(frozen=True)
class MemoryQuery:
    query: str
    limit: int = 5
    filters: dict[str, Any] = field(default_factory=dict)
    mode: MemoryQueryMode = "hybrid"
    namespace: str = ""
    vector: tuple[float, ...] = ()
    entities: tuple[str, ...] = ()
    min_score: float | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-memory-query/v1",
            "query": self.query,
            "limit": self.limit,
            "filters": dict(self.filters),
            "mode": self.mode,
            "namespace": self.namespace,
            "has_vector": bool(self.vector),
            "vector_dimensions": len(self.vector),
            "entities": list(self.entities),
            "min_score": self.min_score,
        }


@dataclass(frozen=True)
class MemoryRoute:
    store: str = ""
    stores: tuple[str, ...] = ()
    namespace: str = ""
    tags: tuple[str, ...] = ()
    mode: MemoryQueryMode | None = None

    @classmethod
    def from_query(cls, query: MemoryQuery) -> "MemoryRoute":
        store = str(query.filters.get("store") or "")
        stores_filter = query.filters.get("stores") or ()
        if isinstance(stores_filter, str):
            stores = (stores_filter,)
        else:
            stores = tuple(str(item) for item in stores_filter)
        tags_filter = query.filters.get("tags") or query.filters.get("tag") or ()
        if isinstance(tags_filter, str):
            tags = (tags_filter,)
        else:
            tags = tuple(str(item) for item in tags_filter)
        return cls(
            store=store,
            stores=stores,
            namespace=str(query.namespace or query.filters.get("namespace") or ""),
            tags=tags,
            mode=query.mode,
        )

    def requested_store_names(self) -> tuple[str, ...]:
        names = []
        if self.store:
            names.append(self.store)
        names.extend(self.stores)
        return tuple(dict.fromkeys(names))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-memory-route/v1",
            "store": self.store,
            "stores": list(self.stores),
            "namespace": self.namespace,
            "tags": list(self.tags),
            "mode": self.mode,
        }


@dataclass(frozen=True)
class MemorySearchPlan:
    query: MemoryQuery
    route: MemoryRoute
    selected_stores: tuple["MemoryStoreSpec", ...] = ()
    backend_filters: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-memory-search-plan/v1",
            "query": self.query.manifest(),
            "route": self.route.manifest(),
            "selected_store_count": len(self.selected_stores),
            "selected_stores": [store.manifest() for store in self.selected_stores],
            "backend_filters": dict(self.backend_filters),
        }



@dataclass(frozen=True)
class MemoryHit:
    content: str
    score: float = 0.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-memory-hit/v1",
            "content_bytes": len(self.content.encode("utf-8")),
            "content_sha256": sha256(self.content.encode("utf-8")).hexdigest()
            if self.content
            else "",
            "score": self.score,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryInjectionCandidate:
    hit: MemoryHit
    route: MemoryInjectionRoute
    utility: float
    scores: dict[str, float] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-memory-injection-candidate/v1",
            "route": self.route,
            "utility": round(self.utility, 6),
            "score": self.hit.score,
            "source": self.hit.source,
            "content_bytes": len(self.hit.content.encode("utf-8")),
            "content_sha256": sha256(self.hit.content.encode("utf-8")).hexdigest()
            if self.hit.content
            else "",
            "scores": {key: round(value, 6) for key, value in self.scores.items()},
            "metadata": dict(self.hit.metadata),
        }


@dataclass(frozen=True)
class MemoryInjectionResult:
    intent: MemoryRecallIntent
    query: str = ""
    candidates: tuple[MemoryInjectionCandidate, ...] = ()
    selected: tuple[MemoryInjectionCandidate, ...] = ()
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-memory-injection-result/v1",
            "intent": self.intent,
            "query": self.query,
            "candidate_count": len(self.candidates),
            "selected_count": len(self.selected),
            "routes": [item.route for item in self.selected],
            "content_bytes": len(self.content.encode("utf-8")),
            "content_sha256": sha256(self.content.encode("utf-8")).hexdigest()
            if self.content
            else "",
            "candidates": [item.manifest() for item in self.candidates],
            "selected": [item.manifest() for item in self.selected],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryEntityProfile:
    """Prompt-safe memory entity profile generated at SDK write time."""

    tags: tuple[str, ...] = ()
    potential_questions: tuple[str, ...] = ()
    scores: dict[str, float] = field(default_factory=dict)
    core_pact_vector: tuple[float, ...] = ()
    strategy: str = "deterministic_core_pact_profile"

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-memory-entity-profile/v1",
            "tags": list(self.tags),
            "potential_questions": list(self.potential_questions),
            "scores": {key: round(float(value), 6) for key, value in self.scores.items()},
            "core_pact_vector": [round(float(value), 6) for value in self.core_pact_vector],
            "strategy": self.strategy,
        }


@dataclass(frozen=True)
class MemoryWrite:
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-memory-write/v1",
            "content_bytes": len(self.content.encode("utf-8")),
            "content_sha256": sha256(self.content.encode("utf-8")).hexdigest()
            if self.content
            else "",
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryTriageRequest:
    content: str
    source: str = "timeline_diff"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-memory-triage-request/v1",
            "content_bytes": len(self.content.encode("utf-8")),
            "content_sha256": sha256(self.content.encode("utf-8")).hexdigest()
            if self.content
            else "",
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryTriageDecision:
    should_write: bool
    item: MemoryWrite | None = None
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-memory-triage-decision/v1",
            "should_write": self.should_write,
            "reason": self.reason,
            "item": self.item.manifest() if self.item is not None else {},
            "metadata": dict(self.metadata),
        }


class MemoryTriagePort(Protocol):
    async def triage(self, request: MemoryTriageRequest) -> MemoryTriageDecision:
        """Decide whether a flushed timeline diff should enter long-term memory."""


class DeterministicMemoryTriage:
    """Default dependency-free triage for loop memory flushes."""

    def __init__(self, *, min_content_bytes: int = 12, max_content_bytes: int = 24 * 1024) -> None:
        self.min_content_bytes = max(0, int(min_content_bytes))
        self.max_content_bytes = max(512, int(max_content_bytes))

    async def triage(self, request: MemoryTriageRequest) -> MemoryTriageDecision:
        content = request.content.strip()
        if len(content.encode("utf-8")) < self.min_content_bytes:
            return MemoryTriageDecision(False, reason="content_too_small")
        raw = content.encode("utf-8")
        metadata = dict(request.metadata)
        if len(raw) > self.max_content_bytes:
            content = raw[: self.max_content_bytes].decode("utf-8", errors="ignore").rstrip()
            metadata["triage_compacted"] = True
            metadata["original_content_bytes"] = len(raw)
            metadata["original_content_sha256"] = sha256(raw).hexdigest()
        return MemoryTriageDecision(
            True,
            item=MemoryWrite(content=content, source=request.source, metadata=metadata),
            reason="deterministic_relevant_diff",
            metadata={"strategy": "deterministic_size_gate"},
        )


@dataclass(frozen=True)
class MemoryFlushSignal:
    content: str
    run_id: str = ""
    turn_id: str = ""
    iteration: int = 0
    status: str = ""
    is_done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bytes(self) -> int:
        return len(self.content.encode("utf-8"))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-memory-flush-signal/v1",
            "content_bytes": self.bytes,
            "content_sha256": sha256(self.content.encode("utf-8")).hexdigest()
            if self.content
            else "",
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "iteration": self.iteration,
            "status": self.status,
            "is_done": self.is_done,
            "metadata": dict(self.metadata),
        }


class MemoryFlushBuffer:
    """Internal Yaklang-style memory flush buffer over original timeline diffs."""

    def __init__(
        self,
        *,
        triage: MemoryTriagePort | None = None,
        byte_threshold: int = 16 * 1024,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.triage = triage or DeterministicMemoryTriage()
        self.byte_threshold = max(1, int(byte_threshold))
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self._signals: list[MemoryFlushSignal] = []
        self._first_at: datetime | None = None
        self._seen_hashes: set[str] = set()
        self.last_manifest: dict[str, Any] = {
            "schema_version": "agent-core-memory-flush-buffer/v1",
            "pending_signal_count": 0,
            "flushed": False,
        }

    async def observe(self, signal: MemoryFlushSignal) -> tuple[MemoryWrite, ...]:
        if not signal.content.strip():
            return ()
        if not self._signals:
            self._first_at = datetime.now(UTC)
        self._signals.append(signal)
        if not self._should_flush(signal):
            self.last_manifest = self._manifest(flushed=False, decisions=())
            return ()
        return await self.flush(reason="done" if signal.is_done else "threshold_or_timeout")

    async def flush(self, *, reason: str = "manual") -> tuple[MemoryWrite, ...]:
        signals = tuple(self._signals)
        self._signals = []
        self._first_at = None
        if not signals:
            self.last_manifest = self._manifest(flushed=False, decisions=(), reason=reason)
            return ()
        content = "\n\n".join(signal.content.strip() for signal in signals if signal.content.strip())
        content_hash = sha256(content.encode("utf-8")).hexdigest() if content else ""
        if content_hash in self._seen_hashes:
            self.last_manifest = self._manifest(
                flushed=True,
                decisions=(),
                reason="duplicate_flush",
                signals=signals,
            )
            return ()
        self._seen_hashes.add(content_hash)
        metadata = {
            "flush_reason": reason,
            "signal_count": len(signals),
            "signals": [signal.manifest() for signal in signals],
            "run_id": signals[-1].run_id,
            "turn_id": signals[-1].turn_id,
            "iteration": signals[-1].iteration,
            "status": signals[-1].status,
        }
        decision = await self.triage.triage(
            MemoryTriageRequest(content=content, source="timeline_diff", metadata=metadata)
        )
        self.last_manifest = self._manifest(
            flushed=True,
            decisions=(decision,),
            reason=reason,
            signals=signals,
        )
        if not decision.should_write or decision.item is None:
            return ()
        return (decision.item,)

    def _should_flush(self, signal: MemoryFlushSignal) -> bool:
        if signal.is_done:
            return True
        if sum(item.bytes for item in self._signals) >= self.byte_threshold:
            return True
        if self._first_at is None or self.timeout_seconds <= 0:
            return False
        age = (datetime.now(UTC) - self._first_at).total_seconds()
        return age >= self.timeout_seconds

    def _manifest(
        self,
        *,
        flushed: bool,
        decisions: tuple[MemoryTriageDecision, ...],
        reason: str = "",
        signals: tuple[MemoryFlushSignal, ...] | None = None,
    ) -> dict[str, Any]:
        pending = tuple(self._signals)
        flushed_signals = tuple(signals or ())
        return {
            "schema_version": "agent-core-memory-flush-buffer/v1",
            "flushed": flushed,
            "reason": reason,
            "pending_signal_count": len(pending),
            "pending_bytes": sum(signal.bytes for signal in pending),
            "flushed_signal_count": len(flushed_signals),
            "flushed_bytes": sum(signal.bytes for signal in flushed_signals),
            "decisions": [decision.manifest() for decision in decisions],
            "byte_threshold": self.byte_threshold,
            "timeout_seconds": self.timeout_seconds,
        }


def infer_memory_recall_intent(
    query: str,
    *,
    topics: tuple[str, ...] = (),
    keywords: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> MemoryRecallIntent:
    """Infer how recalled memories should be used in the next prompt."""

    metadata = dict(metadata or {})
    override = str(metadata.get("memory_intent") or metadata.get("intent") or "").strip().lower()
    if override in {"generic", "fact_check", "advice", "emotional", "brainstorm"}:
        return cast(MemoryRecallIntent, override)
    text = " ".join((query, " ".join(topics), " ".join(keywords))).lower()
    if _contains_any(
        text,
        (
            "verify",
            "validate",
            "confirm",
            "evidence",
            "exact",
            "fact",
            "whether",
            "prove",
            "source",
        ),
    ):
        return "fact_check"
    if _contains_any(text, ("should", "how to", "fix", "plan", "recommend", "next", "action", "execute")):
        return "advice"
    if _contains_any(text, ("idea", "brainstorm", "explore", "alternatives", "options")):
        return "brainstorm"
    if _contains_any(text, ("feel", "frustrated", "tone", "emotional", "concern", "worry")):
        return "emotional"
    return "generic"


def build_memory_injection(
    hits: tuple[MemoryHit, ...],
    *,
    query: str = "",
    intent: MemoryRecallIntent = "generic",
    max_total: int = 12,
    max_per_route: int = 4,
    max_content_chars: int = 600,
    min_utility: float = 0.0,
) -> MemoryInjectionResult:
    candidates = tuple(
        sorted(
            (_memory_injection_candidate(hit, query=query, intent=intent) for hit in hits),
            key=lambda item: (-item.utility, item.route, item.hit.source),
        )
    )
    selected: list[MemoryInjectionCandidate] = []
    route_counts: dict[str, int] = {}
    for route in _memory_route_order(intent):
        for candidate in candidates:
            if candidate.route != route:
                continue
            if candidate.utility < min_utility:
                continue
            if len(selected) >= max(1, int(max_total)):
                break
            if route_counts.get(route, 0) >= max(1, int(max_per_route)):
                continue
            selected.append(candidate)
            route_counts[route] = route_counts.get(route, 0) + 1
        if len(selected) >= max(1, int(max_total)):
            break
    content = _render_memory_injection(tuple(selected), max_content_chars=max_content_chars)
    return MemoryInjectionResult(
        intent=intent,
        query=query,
        candidates=candidates,
        selected=tuple(selected),
        content=content,
        metadata={
            "strategy": "deterministic_core_pact_route_rerank",
            "max_total": max_total,
            "max_per_route": max_per_route,
            "max_content_chars": max_content_chars,
            "min_utility": min_utility,
        },
    )


def build_memory_entity_profile(
    item: MemoryWrite | MemoryRecord,
    *,
    query_hint: str = "",
) -> MemoryEntityProfile:
    content = str(item.content or "").strip()
    metadata = dict(item.metadata)
    existing = metadata.get("memory_entity_profile")
    if isinstance(existing, dict):
        scores = _core_pact_scores_from_metadata(existing) or _core_pact_scores_from_metadata(metadata)
        vector = _core_pact_vector_from_metadata(existing) or _core_pact_vector(scores)
        tags = _string_tuple(existing.get("tags")) or _string_tuple(metadata.get("tags")) or _infer_memory_tags(content, metadata)
        questions = (
            _string_tuple(existing.get("potential_questions"))
            or _string_tuple(metadata.get("potential_questions"))
            or _infer_potential_questions(content, tags)
        )
        return MemoryEntityProfile(
            tags=tags,
            potential_questions=questions,
            scores=scores,
            core_pact_vector=vector,
            strategy=str(existing.get("strategy") or "metadata_existing_profile"),
        )
    tags = _string_tuple(metadata.get("tags")) or _infer_memory_tags(content, metadata)
    questions = _string_tuple(metadata.get("potential_questions")) or _infer_potential_questions(content, tags)
    scores = _infer_core_pact_scores(content, metadata, query_hint=query_hint)
    return MemoryEntityProfile(
        tags=tags,
        potential_questions=questions,
        scores=scores,
        core_pact_vector=_core_pact_vector(scores),
    )


def enrich_memory_write(item: MemoryWrite) -> MemoryWrite:
    profile = build_memory_entity_profile(item)
    metadata = _metadata_with_memory_profile(dict(item.metadata), profile)
    return MemoryWrite(content=item.content, source=item.source, metadata=metadata)


def _memory_injection_candidate(
    hit: MemoryHit,
    *,
    query: str,
    intent: MemoryRecallIntent,
) -> MemoryInjectionCandidate:
    scores = _memory_signal_scores(hit, query=query)
    weights = _memory_intent_weights(intent)
    similarity = _clamp01(hit.score)
    utility = similarity * weights["sim"]
    for key in ("c", "o", "r", "e", "p", "a", "t"):
        utility += scores[key] * weights.get(key, 0.0)
    route = _memory_injection_route(scores, intent)
    return MemoryInjectionCandidate(
        hit=hit,
        route=route,
        utility=_clamp01(utility),
        scores={**scores, "sim": similarity},
    )


def _memory_signal_scores(hit: MemoryHit, *, query: str) -> dict[str, float]:
    metadata = dict(hit.metadata)
    content = hit.content.lower()
    kind = str(metadata.get("kind") or metadata.get("entity_type") or "").lower()
    source = str(hit.source or metadata.get("source") or "").lower()
    score = _clamp01(hit.score)
    terms = _memory_terms(query)
    overlap = _term_overlap(content, terms)
    relevance = max(score, overlap)
    actionability = _core_pact_score(metadata, "a")
    if actionability == 0.0 and _contains_any(
        " ".join((content, kind)),
        ("should", "must", "use ", "run ", "check ", "prefer", "step", "guidance", "playbook"),
    ):
        actionability = 0.82
    preference = _core_pact_score(metadata, "p")
    if preference == 0.0 and _contains_any(" ".join((content, kind)), ("prefer", "must", "always", "never", "policy")):
        preference = 0.78
    origin = _core_pact_score(metadata, "o")
    if origin == 0.0:
        origin = 0.35 if _contains_any(" ".join((content, source, kind)), ("unverified", "unknown", "guess")) else 0.72
    emotional = _core_pact_score(metadata, "e")
    connectivity = _core_pact_score(metadata, "c")
    if connectivity == 0.0 and len(terms) > 1 and overlap > 0:
        connectivity = min(1.0, overlap + 0.25)
    timeliness = _core_pact_score(metadata, "t")
    if timeliness == 0.0:
        timeliness = 0.55
    return {
        "c": _clamp01(connectivity),
        "o": _clamp01(origin),
        "r": max(_core_pact_score(metadata, "r"), _clamp01(relevance)),
        "e": _clamp01(emotional),
        "p": _clamp01(preference),
        "a": _clamp01(actionability),
        "t": _clamp01(timeliness),
    }


def _memory_injection_route(
    scores: dict[str, float],
    intent: MemoryRecallIntent,
) -> MemoryInjectionRoute:
    candidates: dict[MemoryInjectionRoute, bool] = {
        "action_tips": scores["a"] >= 0.78 and scores["r"] >= 0.35,
        "must_aware": (scores["r"] >= 0.70 and scores["p"] >= 0.55)
        or (scores["p"] >= 0.82 and scores["r"] >= 0.35),
        "reliability_warning": scores["r"] >= 0.62 and scores["o"] <= 0.42,
        "emotional_context": scores["e"] >= 0.55 and (intent == "emotional" or scores["r"] >= 0.45),
        "connection_links": scores["c"] >= 0.70 and scores["r"] >= 0.30,
    }
    for route in _memory_route_order(intent):
        if route != "context" and candidates.get(route):
            return route
    return "context"


def _memory_route_order(intent: MemoryRecallIntent) -> tuple[MemoryInjectionRoute, ...]:
    if intent == "advice":
        return (
            "action_tips",
            "must_aware",
            "reliability_warning",
            "context",
            "connection_links",
            "emotional_context",
        )
    if intent == "fact_check":
        return (
            "reliability_warning",
            "must_aware",
            "context",
            "action_tips",
            "connection_links",
            "emotional_context",
        )
    if intent == "emotional":
        return (
            "emotional_context",
            "must_aware",
            "context",
            "reliability_warning",
            "action_tips",
            "connection_links",
        )
    if intent == "brainstorm":
        return (
            "connection_links",
            "context",
            "must_aware",
            "action_tips",
            "reliability_warning",
            "emotional_context",
        )
    return (
        "must_aware",
        "action_tips",
        "reliability_warning",
        "context",
        "connection_links",
        "emotional_context",
    )


def _memory_intent_weights(intent: MemoryRecallIntent) -> dict[str, float]:
    if intent == "fact_check":
        return {"sim": 0.45, "o": 0.32, "r": 0.22, "t": 0.16}
    if intent == "advice":
        return {"sim": 0.42, "a": 0.30, "p": 0.22, "r": 0.18}
    if intent == "emotional":
        return {"sim": 0.35, "e": 0.38, "p": 0.24, "r": 0.12}
    if intent == "brainstorm":
        return {"sim": 0.35, "c": 0.34, "r": 0.24, "a": 0.10}
    return {"sim": 0.40, "r": 0.20, "c": 0.14, "t": 0.12, "a": 0.12, "p": 0.08, "o": 0.08}


def _render_memory_injection(
    candidates: tuple[MemoryInjectionCandidate, ...],
    *,
    max_content_chars: int,
) -> str:
    if not candidates:
        return ""
    by_route: dict[str, list[MemoryInjectionCandidate]] = {}
    for candidate in candidates:
        by_route.setdefault(candidate.route, []).append(candidate)
    lines = ["[memory]", "### Retrieved Memories (Contextual)"]
    for route in ("must_aware", "action_tips", "reliability_warning", "context", "connection_links", "emotional_context"):
        items = by_route.get(route) or []
        if not items:
            continue
        lines.append(f"[ {route} ]")
        for candidate in items:
            hit = candidate.hit
            source = hit.source or "memory"
            content = _shrink_memory_content(hit.content, max_content_chars)
            scores = candidate.scores
            lines.append(
                "- "
                f"{source} "
                f"(u={candidate.utility:.3f} sim={scores.get('sim', 0.0):.3f} "
                f"R={scores.get('r', 0.0):.2f} O={scores.get('o', 0.0):.2f} "
                f"A={scores.get('a', 0.0):.2f} P={scores.get('p', 0.0):.2f}): "
                f"{content}"
            )
        lines.append("")
    return "\n".join(lines).strip()


def _metadata_with_memory_profile(
    metadata: dict[str, Any],
    profile: MemoryEntityProfile,
) -> dict[str, Any]:
    enriched = dict(metadata)
    manifest = profile.manifest()
    enriched["memory_entity_profile"] = manifest
    enriched.setdefault("tags", list(profile.tags))
    enriched.setdefault("potential_questions", list(profile.potential_questions))
    enriched["core_pact_scores"] = dict(manifest["scores"])
    enriched["core_pact_vector"] = list(manifest["core_pact_vector"])
    for key, value in profile.scores.items():
        enriched.setdefault(f"{key}_score", float(value))
    return enriched


def _infer_core_pact_scores(
    content: str,
    metadata: dict[str, Any],
    *,
    query_hint: str = "",
) -> dict[str, float]:
    existing = _core_pact_scores_from_metadata(metadata)
    if existing:
        return existing
    text = " ".join((content, str(metadata.get("kind") or ""), str(metadata.get("source") or ""))).lower()
    tags = _string_tuple(metadata.get("tags"))
    questions = _string_tuple(metadata.get("potential_questions"))
    query_terms = _memory_terms(query_hint)
    relevance = _term_overlap(content, query_terms) if query_terms else 0.55
    actionability = 0.78 if _contains_any(
        text,
        ("should", "must", "use ", "run ", "check ", "prefer", "step", "guidance", "playbook", "fix"),
    ) else 0.42
    preference = 0.76 if _contains_any(text, ("prefer", "always", "never", "policy", "constraint", "must")) else 0.36
    origin = 0.32 if _contains_any(text, ("unverified", "unknown", "guess", "maybe")) else 0.68
    emotion = 0.70 if _contains_any(text, ("frustrated", "worried", "angry", "blocked")) else 0.20
    connectivity = min(0.9, 0.30 + 0.08 * len(tags) + 0.05 * len(questions))
    timeliness = 0.78 if _contains_any(text, ("now", "current", "latest", "today", "recent")) else 0.55
    return {
        "c": _clamp01(connectivity),
        "o": _clamp01(origin),
        "r": _clamp01(relevance),
        "e": _clamp01(emotion),
        "p": _clamp01(preference),
        "a": _clamp01(actionability),
        "t": _clamp01(timeliness),
    }


def _infer_memory_tags(content: str, metadata: dict[str, Any], *, limit: int = 8) -> tuple[str, ...]:
    explicit = _string_tuple(metadata.get("tags"))
    if explicit:
        return explicit[:limit]
    candidates = [
        str(metadata.get("kind") or ""),
        str(metadata.get("entity_type") or ""),
        str(metadata.get("scope") or ""),
        str(metadata.get("bucket") or ""),
        *_memory_terms(content),
    ]
    stop = {
        "this",
        "that",
        "with",
        "from",
        "have",
        "about",
        "before",
        "after",
        "should",
        "must",
        "memory",
    }
    tags: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        tag = _normalize_memory_text(candidate).replace(" ", "_")
        if not tag or len(tag) < 3 or tag in stop or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag[:48])
        if len(tags) >= limit:
            break
    return tuple(tags)


def _infer_potential_questions(
    content: str,
    tags: tuple[str, ...],
    *,
    limit: int = 4,
) -> tuple[str, ...]:
    collapsed = " ".join(str(content or "").split())
    if not collapsed:
        return ()
    subject = ", ".join(tags[:3]) if tags else _shrink_memory_content(collapsed, 64)
    questions = [
        f"What prior context is relevant to {subject}?",
        f"When should this memory affect actions about {subject}?",
    ]
    if _contains_any(collapsed.lower(), ("must", "should", "prefer", "always", "never")):
        questions.append(f"What constraints or preferences apply to {subject}?")
    if _contains_any(collapsed.lower(), ("unverified", "evidence", "confirm", "verify")):
        questions.append(f"What evidence should be verified for {subject}?")
    return tuple(dict.fromkeys(questions))[:limit]


def _core_pact_score(metadata: dict[str, Any], key: str) -> float:
    scores = _core_pact_scores_from_metadata(metadata)
    if key in scores:
        return _clamp01(scores[key])
    return _metadata_score(metadata, CORE_PACT_METADATA_KEYS[key])


def _core_pact_scores_from_metadata(metadata: dict[str, Any]) -> dict[str, float]:
    nested = metadata.get("core_pact_scores")
    if not isinstance(nested, dict):
        nested = metadata.get("scores")
    scores: dict[str, float] = {}
    if isinstance(nested, dict):
        for key in CORE_PACT_SCORE_KEYS:
            if key in nested:
                scores[key] = _clamp01(nested.get(key))
            elif f"{key}_score" in nested:
                scores[key] = _clamp01(nested.get(f"{key}_score"))
    for key in CORE_PACT_SCORE_KEYS:
        if key not in scores:
            value = _metadata_score(metadata, CORE_PACT_METADATA_KEYS[key])
            if value:
                scores[key] = value
    if not scores:
        return {}
    return {key: _clamp01(scores.get(key, 0.0)) for key in CORE_PACT_SCORE_KEYS}


def _core_pact_vector_from_metadata(metadata: dict[str, Any]) -> tuple[float, ...]:
    raw = metadata.get("core_pact_vector")
    if raw is None:
        raw = metadata.get("vector")
    values = _float_tuple(raw)
    if len(values) == len(CORE_PACT_SCORE_KEYS):
        return tuple(_clamp01(value) for value in values)
    return ()


def _core_pact_vector(scores: dict[str, float]) -> tuple[float, ...]:
    return tuple(_clamp01(scores.get(key, 0.0)) for key in CORE_PACT_SCORE_KEYS)


def _metadata_score(metadata: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        try:
            return _clamp01(float(value))
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            if text in {"high", "strong", "trusted", "verified"}:
                return 0.9
            if text in {"medium", "normal", "partial"}:
                return 0.55
            if text in {"low", "weak", "unverified", "unknown"}:
                return 0.25
    return 0.0


def _memory_terms(text: str) -> frozenset[str]:
    return frozenset(
        token
        for token in re.findall(r"[a-zA-Z0-9_/-]{3,}", str(text or "").lower())
        if token
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    try:
        return tuple(str(item).strip() for item in value or () if str(item).strip())
    except TypeError:
        return ()


def _float_tuple(value: Any) -> tuple[float, ...]:
    try:
        return tuple(float(item) for item in value or ())
    except (TypeError, ValueError):
        return ()


def _term_overlap(text: str, query_terms: frozenset[str]) -> float:
    if not query_terms:
        return 0.0
    terms = _memory_terms(text)
    if not terms:
        return 0.0
    return min(1.0, len(terms & query_terms) / max(1, len(query_terms)))


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    text = str(text or "").lower()
    return any(needle in text for needle in needles)


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric < 0:
        return 0.0
    if numeric > 1:
        return 1.0
    return numeric


def _shrink_memory_content(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    limit = max(32, int(limit))
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


class MemoryPort(Protocol):
    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        """Search long-term memory."""

    async def write(self, item: MemoryWrite) -> None:
        """Write one memory item."""


class NullMemory:
    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        return ()

    async def write(self, item: MemoryWrite) -> None:
        return None


@dataclass(frozen=True)
class MemoryStoreSpec:
    name: str
    priority: int = 0
    readable: bool = True
    writable: bool = True
    backend_kind: MemoryBackendKind = "custom"
    core_builtin: bool = True
    location: str = ""
    namespaces: tuple[str, ...] = ()
    supports_keyword: bool = True
    supports_semantic: bool = True
    supports_vector: bool = False
    supports_graph: bool = False
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "priority": self.priority,
            "readable": self.readable,
            "writable": self.writable,
            "backend_kind": self.backend_kind,
            "core_builtin": self.core_builtin,
            "location": self.location,
            "backend": storage_backend_manifest(
                role="memory",
                kind=cast(StorageBackendKind, self.backend_kind),
                name=self.name,
                namespace=",".join(self.namespaces),
                capabilities=tuple(
                    name
                    for name, enabled in {
                        "keyword": self.supports_keyword,
                        "semantic": self.supports_semantic,
                        "vector": self.supports_vector,
                        "graph": self.supports_graph,
                    }.items()
                    if enabled
                ),
                location=self.location,
                core_builtin=self.core_builtin,
                metadata={"tags": list(self.tags)},
            ),
            "namespaces": list(self.namespaces),
            "capabilities": {
                "keyword": self.supports_keyword,
                "semantic": self.supports_semantic,
                "vector": self.supports_vector,
                "graph": self.supports_graph,
            },
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class _MemoryStoreMount:
    spec: MemoryStoreSpec
    store: MemoryPort


@dataclass(frozen=True)
class ExternalMemoryCallRecord:
    operation: Literal["search", "write"]
    status: Literal["completed", "failed"]
    store_name: str
    backend_kind: MemoryBackendKind
    query: dict[str, Any] = field(default_factory=dict)
    write: dict[str, Any] = field(default_factory=dict)
    hit_count: int = 0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-external-memory-call-record/v1",
            "operation": self.operation,
            "status": self.status,
            "store_name": self.store_name,
            "backend_kind": self.backend_kind,
            "query": dict(self.query),
            "write": dict(self.write),
            "hit_count": self.hit_count,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class MemoryStoreNotFoundError(KeyError):
    pass


class MemoryGovernanceDeniedError(PermissionError):
    def __init__(self, decision: "MemoryGovernanceDecision") -> None:
        super().__init__(decision.reason)
        self.decision = decision


@dataclass(frozen=True)
class MemoryGovernanceDecision:
    allowed: bool
    decision: Literal["allow", "deny", "rewrite"]
    reason: str
    risk_level: Literal["low", "medium", "high"] = "low"
    store: str = ""
    item: MemoryWrite | None = None
    leaks: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "decision": self.decision,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "store": self.store,
            "leaks": list(self.leaks),
            "metadata": dict(self.metadata),
        }

    def manifest(self) -> dict[str, Any]:
        item = self.item
        item_manifest = {}
        if item is not None:
            item_manifest = {
                "content_bytes": len(item.content.encode("utf-8")),
                "content_sha256": sha256(item.content.encode("utf-8")).hexdigest(),
                "source": item.source,
                "metadata": dict(item.metadata),
            }
        return {
            "schema_version": "agent-core-memory-governance-decision/v1",
            "allowed": self.allowed,
            "decision": self.decision,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "store": self.store,
            "item": item_manifest,
            "leak_count": len(self.leaks),
            "leak_sha256s": [_stable_hash(leak) for leak in self.leaks],
            "metadata": dict(self.metadata),
        }


class MemoryGovernancePort(Protocol):
    async def authorize_write(
        self,
        item: MemoryWrite,
        *,
        store: MemoryStoreSpec,
    ) -> MemoryGovernanceDecision:
        """Authorize and optionally rewrite one memory write."""


class AllowAllMemoryGovernance(MemoryGovernancePort):
    async def authorize_write(
        self,
        item: MemoryWrite,
        *,
        store: MemoryStoreSpec,
    ) -> MemoryGovernanceDecision:
        return MemoryGovernanceDecision(
            allowed=True,
            decision="allow",
            reason="memory_governance_allow_all",
            risk_level=_risk_level_for_scope(item.metadata.get("scope")),
            store=store.name,
            item=item,
        )


class RuleBasedMemoryGovernance(MemoryGovernancePort):
    """Provider-neutral write gate for long-term memory."""

    def __init__(
        self,
        *,
        default_scope: str = "project",
        allowed_global_buckets: tuple[str, ...] = (),
        max_content_chars: int = 8000,
        deny_secret_leaks_all_scopes: bool = False,
    ) -> None:
        self.default_scope = default_scope
        self.allowed_global_buckets = set(allowed_global_buckets)
        self.max_content_chars = max(1, max_content_chars)
        self.deny_secret_leaks_all_scopes = deny_secret_leaks_all_scopes

    async def authorize_write(
        self,
        item: MemoryWrite,
        *,
        store: MemoryStoreSpec,
    ) -> MemoryGovernanceDecision:
        content = item.content.strip()
        if not content:
            return self._deny(
                reason="empty_memory_write",
                store=store,
                scope=item.metadata.get("scope"),
            )

        metadata = dict(item.metadata)
        scope = str(metadata.get("scope") or self.default_scope).strip() or "project"
        metadata["scope"] = scope
        bucket = str(metadata.get("bucket") or _bucket_subject(scope) or "").strip()
        if bucket:
            metadata["bucket"] = bucket

        leaks = _scan_memory_leaks(content, item.source, metadata)
        protected_scope = scope.startswith(("global", "user"))
        if leaks and (protected_scope or self.deny_secret_leaks_all_scopes):
            return self._deny(
                reason="memory_target_or_secret_leak",
                store=store,
                scope=scope,
                bucket=bucket,
                leaks=tuple(leaks[:12]),
            )

        if scope.startswith("global"):
            if not bucket:
                return self._deny(
                    reason="invalid_global_memory_bucket",
                    store=store,
                    scope=scope,
                    bucket=bucket,
                )
            if self.allowed_global_buckets and bucket not in self.allowed_global_buckets:
                return self._deny(
                    reason="global_memory_bucket_not_allowed",
                    store=store,
                    scope=scope,
                    bucket=bucket,
                )

        truncated = False
        if len(content) > self.max_content_chars:
            content = content[: self.max_content_chars].rstrip()
            metadata["memory_truncated"] = True
            truncated = True

        profile = _memory_profile(
            content=content,
            source=item.source,
            scope=scope,
            bucket=bucket,
            metadata=metadata,
        )
        metadata["memory_fingerprint"] = profile["fingerprint"]
        metadata["memory_profile"] = profile
        rewritten = MemoryWrite(content=content, source=item.source, metadata=metadata)
        decision = "rewrite" if truncated or rewritten != item else "allow"
        return MemoryGovernanceDecision(
            allowed=True,
            decision=decision,
            reason="memory_write_policy_allow" if decision == "allow" else "memory_write_policy_rewrite",
            risk_level=_risk_level_for_scope(scope),
            store=store.name,
            item=rewritten,
            metadata={
                "scope": scope,
                "bucket": bucket,
                "fingerprint": profile["fingerprint"],
                "truncated": truncated,
            },
        )

    def _deny(
        self,
        *,
        reason: str,
        store: MemoryStoreSpec,
        scope: Any,
        bucket: str = "",
        leaks: tuple[str, ...] = (),
    ) -> MemoryGovernanceDecision:
        return MemoryGovernanceDecision(
            allowed=False,
            decision="deny",
            reason=reason,
            risk_level=_risk_level_for_scope(scope),
            store=store.name,
            leaks=leaks,
            metadata={"scope": str(scope or self.default_scope), "bucket": bucket},
        )


class MemoryCenter(MemoryPort):
    """Fan-out memory router for core-managed and runtime-managed stores."""

    def __init__(
        self,
        *,
        default_store: str = "",
        governance: MemoryGovernancePort | None = None,
    ) -> None:
        self.default_store = default_store
        self.governance = governance
        self._stores: dict[str, _MemoryStoreMount] = {}
        self._governance_decisions: list[dict[str, Any]] = []

    def register(
        self,
        name: str,
        store: MemoryPort,
        *,
        priority: int = 0,
        readable: bool = True,
        writable: bool = True,
        backend_kind: MemoryBackendKind = "custom",
        namespaces: tuple[str, ...] = (),
        supports_keyword: bool = True,
        supports_semantic: bool = True,
        supports_vector: bool = False,
        supports_graph: bool = False,
        core_builtin: bool = True,
        location: str = "",
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not name:
            raise ValueError("memory store name is required")
        resolved_backend_kind = (
            _infer_memory_backend_kind(store) if backend_kind == "custom" else backend_kind
        )
        self._stores[name] = _MemoryStoreMount(
            spec=MemoryStoreSpec(
                name=name,
                priority=priority,
                readable=readable,
                writable=writable,
                backend_kind=resolved_backend_kind,
                core_builtin=core_builtin,
                location=location,
                namespaces=tuple(namespaces),
                supports_keyword=supports_keyword,
                supports_semantic=supports_semantic,
                supports_vector=supports_vector or _store_supports_strategy_vector(store),
                supports_graph=supports_graph,
                tags=tuple(tags),
                metadata=dict(metadata or {}),
            ),
            store=store,
        )

    def register_spec(self, spec: MemoryStoreSpec, store: MemoryPort) -> None:
        if not spec.name:
            raise ValueError("memory store name is required")
        self._stores[spec.name] = _MemoryStoreMount(spec=spec, store=store)

    def get(self, name: str) -> MemoryPort:
        mount = self._stores.get(name)
        if mount is None:
            raise MemoryStoreNotFoundError(name)
        return mount.store

    def specs(self) -> tuple[MemoryStoreSpec, ...]:
        return tuple(mount.spec for mount in self._ordered_mounts())

    def search_stores(self, query: str = "", *, tag: str = "") -> tuple[MemoryStoreSpec, ...]:
        terms = tuple(part.casefold() for part in query.split() if part)
        specs: list[MemoryStoreSpec] = []
        for mount in self._ordered_mounts():
            spec = mount.spec
            if tag and tag not in spec.tags:
                continue
            haystack = " ".join((spec.name, *spec.tags, *map(str, spec.metadata.values()))).casefold()
            if terms and not all(term in haystack for term in terms):
                continue
            specs.append(spec)
        return tuple(specs)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-memory-center/v1",
            "default_store": self.default_store,
            "stores": [spec.manifest() for spec in self.specs()],
            "governance": {
                "schema_version": "agent-core-memory-governance-trace/v1",
                "enabled": self.governance is not None,
                "decision_count": len(self._governance_decisions),
                "decisions": [dict(item) for item in self._governance_decisions],
            },
        }

    def plan_search(self, query: MemoryQuery) -> MemorySearchPlan:
        route = MemoryRoute.from_query(query)
        backend_filters = {
            key: value
            for key, value in query.filters.items()
            if key not in {"store", "stores", "tag", "tags"}
        }
        mounts = self._select_read_mounts(route=route)
        return MemorySearchPlan(
            query=query,
            route=route,
            selected_stores=tuple(mount.spec for mount in mounts),
            backend_filters=backend_filters,
        )

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        plan = self.plan_search(query)
        mounts = tuple(self._stores[spec.name] for spec in plan.selected_stores)
        hits: list[tuple[int, int, MemoryHit]] = []
        for mount in mounts:
            store_query = MemoryQuery(
                query=query.query,
                limit=query.limit,
                filters=dict(plan.backend_filters),
                mode=query.mode,
                namespace=query.namespace,
                vector=tuple(query.vector),
                entities=tuple(query.entities),
                min_score=query.min_score,
            )
            for index, hit in enumerate(await mount.store.search(store_query)):
                if query.min_score is not None and hit.score < query.min_score:
                    continue
                hits.append(
                    (
                        mount.spec.priority,
                        index,
                        MemoryHit(
                            content=hit.content,
                            score=hit.score,
                            source=hit.source,
                            metadata={**hit.metadata, "store": mount.spec.name},
                        ),
                    )
                )
        hits.sort(key=lambda item: (-item[2].score, -item[0], item[1], item[2].source))
        return tuple(hit for _, _, hit in hits[: query.limit])

    async def write(self, item: MemoryWrite) -> None:
        store_name = str(item.metadata.get("store") or self.default_store or "")
        mount = self._select_write_mount(store_name)
        metadata = {key: value for key, value in item.metadata.items() if key != "store"}
        write_item = MemoryWrite(
            content=item.content,
            source=item.source,
            metadata=metadata,
        )
        if self.governance is not None:
            decision = await self.governance.authorize_write(write_item, store=mount.spec)
            self._record_governance_decision(decision)
            if not decision.allowed:
                raise MemoryGovernanceDeniedError(decision)
            write_item = decision.item or write_item
        await mount.store.write(write_item)

    def _record_governance_decision(self, decision: MemoryGovernanceDecision) -> None:
        manifest = decision.manifest()
        manifest["sequence"] = len(self._governance_decisions) + 1
        self._governance_decisions.append(manifest)

    def _ordered_mounts(self) -> tuple[_MemoryStoreMount, ...]:
        return tuple(
            sorted(
                self._stores.values(),
                key=lambda mount: (-mount.spec.priority, mount.spec.name),
            )
        )

    def _select_read_mounts(
        self,
        *,
        route: MemoryRoute,
    ) -> tuple[_MemoryStoreMount, ...]:
        names = set(route.requested_store_names())

        mounts = tuple(
            mount
            for mount in self._ordered_mounts()
            if mount.spec.readable
            and (not names or mount.spec.name in names)
            and _store_supports_route(mount.spec, route)
        )
        if names and len(mounts) != len(names):
            missing = sorted(names - {mount.spec.name for mount in mounts})
            raise MemoryStoreNotFoundError(",".join(missing))
        return mounts

    def _select_write_mount(self, store_name: str) -> _MemoryStoreMount:
        if store_name:
            mount = self._stores.get(store_name)
            if mount is None or not mount.spec.writable:
                raise MemoryStoreNotFoundError(store_name)
            return mount
        for mount in self._ordered_mounts():
            if mount.spec.writable:
                return mount
        raise MemoryStoreNotFoundError("<writable>")


@dataclass(frozen=True)
class MemoryRecord:
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    @property
    def entity_profile(self) -> MemoryEntityProfile:
        return build_memory_entity_profile(self)

    @property
    def core_pact_vector(self) -> tuple[float, ...]:
        return self.entity_profile.core_pact_vector

    def hit(self, *, score: float) -> MemoryHit:
        return MemoryHit(
            content=self.content,
            score=score,
            source=self.source,
            metadata={**self.metadata, "created_at": self.created_at},
        )


class InMemoryMemoryStore(MemoryPort):
    def __init__(
        self,
        records: tuple[MemoryRecord, ...] = (),
        *,
        embedding_provider: EmbeddingProviderPort | None = None,
        embedding_model: str = "",
        embedding_dimensions: int = 0,
    ) -> None:
        self.records: list[MemoryRecord] = list(records)
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        records = [record for record in self.records if _matches_filters(record, query.filters)]
        if self.embedding_provider is not None and query.mode in {"semantic", "hybrid"}:
            return await _rank_memory_records_semantic(
                query.query,
                records,
                provider=self.embedding_provider,
                model=self.embedding_model,
                dimensions=self.embedding_dimensions or len(query.vector),
                limit=query.limit,
                query_vector=query.vector,
            )
        return _rank_memory_records(query.query, records, limit=query.limit, query_vector=query.vector)

    async def write(self, item: MemoryWrite) -> None:
        item = enrich_memory_write(item)
        self.records.append(
            MemoryRecord(
                content=item.content,
                source=item.source,
                metadata=dict(item.metadata),
                created_at=_utc_now_iso(),
            )
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-in-memory-memory-store/v1",
            "backend_kind": "in_memory",
            "backend": storage_backend_manifest(
                role="memory",
                kind="in_memory",
                capabilities=("keyword", "semantic", "vector"),
            ),
            "record_count": len(self.records),
            "semantic_ranking": self.embedding_provider is not None,
            "strategy_vector_index": True,
        }


class ExternalMemoryStore(MemoryPort):
    """Runtime-owned memory adapter wrapper.

    This class deliberately does not open database connections. It gives PG,
    vector DB, graph, product-memory, or other runtime adapters the same
    prompt-safe manifest and `MemoryPort` shape as SDK-local stores.
    """

    def __init__(
        self,
        adapter: MemoryPort,
        *,
        name: str,
        backend_kind: MemoryBackendKind = "external",
        priority: int = 0,
        readable: bool = True,
        writable: bool = True,
        namespaces: tuple[str, ...] = (),
        supports_keyword: bool = True,
        supports_semantic: bool = True,
        supports_vector: bool = False,
        supports_graph: bool = False,
        tags: tuple[str, ...] = (),
        location: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not name:
            raise ValueError("external memory store name is required")
        self.adapter = adapter
        self.calls: list[ExternalMemoryCallRecord] = []
        self._spec = MemoryStoreSpec(
            name=name,
            priority=priority,
            readable=readable,
            writable=writable,
            backend_kind=backend_kind,
            core_builtin=False,
            location=location,
            namespaces=tuple(namespaces),
            supports_keyword=supports_keyword,
            supports_semantic=supports_semantic,
            supports_vector=supports_vector,
            supports_graph=supports_graph,
            tags=tuple(tags),
            metadata=dict(metadata or {}),
        )

    @property
    def spec(self) -> MemoryStoreSpec:
        return self._spec

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        try:
            hits = await self.adapter.search(query)
        except Exception as exc:
            self._record_call(
                ExternalMemoryCallRecord(
                    operation="search",
                    status="failed",
                    store_name=self._spec.name,
                    backend_kind=self._spec.backend_kind,
                    query=_memory_query_audit_manifest(query),
                    error=str(exc),
                )
            )
            raise
        self._record_call(
            ExternalMemoryCallRecord(
                operation="search",
                status="completed",
                store_name=self._spec.name,
                backend_kind=self._spec.backend_kind,
                query=_memory_query_audit_manifest(query),
                hit_count=len(hits),
            )
        )
        return hits

    async def write(self, item: MemoryWrite) -> None:
        try:
            await self.adapter.write(item)
        except Exception as exc:
            self._record_call(
                ExternalMemoryCallRecord(
                    operation="write",
                    status="failed",
                    store_name=self._spec.name,
                    backend_kind=self._spec.backend_kind,
                    write=item.manifest(),
                    error=str(exc),
                )
            )
            raise
        self._record_call(
            ExternalMemoryCallRecord(
                operation="write",
                status="completed",
                store_name=self._spec.name,
                backend_kind=self._spec.backend_kind,
                write=item.manifest(),
            )
        )

    def _record_call(self, record: ExternalMemoryCallRecord) -> None:
        self.calls.append(record)

    def manifest(self) -> dict[str, Any]:
        adapter_manifest = getattr(self.adapter, "manifest", None)
        return {
            "schema_version": "agent-core-external-memory-store/v1",
            "backend_kind": self._spec.backend_kind,
            "store": self._spec.manifest(),
            "call_count": len(self.calls),
            "calls": [call.manifest() for call in self.calls],
            "adapter": adapter_manifest() if callable(adapter_manifest) else {},
        }


class SQLiteMemoryStore(MemoryPort):
    """Small SQLite-backed memory store for core and lightweight runtimes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        records = [record for record in self._records() if _matches_filters(record, query.filters)]
        return _rank_memory_records(
            query.query,
            records,
            limit=query.limit,
            query_vector=query.vector,
        )

    async def write(self, item: MemoryWrite) -> None:
        item = enrich_memory_write(item)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT INTO memories(content, source, metadata_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    item.content,
                    item.source,
                    json.dumps(item.metadata, ensure_ascii=False, sort_keys=True),
                    _utc_now_iso(),
                ),
            )
            conn.commit()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-sqlite-memory-store/v1",
            "backend_kind": "sqlite",
            "backend": storage_backend_manifest(
                role="memory",
                kind="sqlite",
                location=str(self.path),
                capabilities=("keyword", "semantic", "vector"),
            ),
            "path": str(self.path),
            "record_count": len(self._records()),
            "strategy_vector_index": True,
        }

    def _init(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _records(self) -> tuple[MemoryRecord, ...]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                """
                SELECT content, source, metadata_json, created_at
                FROM memories
                ORDER BY id ASC
                """
            ).fetchall()
        records: list[MemoryRecord] = []
        for content, source, metadata_json, created_at in rows:
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                metadata = {}
            records.append(
                MemoryRecord(
                    content=str(content),
                    source=str(source),
                    metadata=metadata if isinstance(metadata, dict) else {},
                    created_at=str(created_at),
                )
            )
        return tuple(records)


class MarkdownMemoryStore(MemoryPort):
    """Markdown-backed memory store.

    Files written by this store are parsed as individual memory entries. Other
    markdown files under the root are indexed as whole-document memory records.
    """

    def __init__(self, root: str | Path, *, filename: str = "memories.md") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / filename

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        records = [record for record in self._records() if _matches_filters(record, query.filters)]
        return _rank_memory_records(
            query.query,
            records,
            limit=query.limit,
            query_vector=query.vector,
        )

    async def write(self, item: MemoryWrite) -> None:
        item = enrich_memory_write(item)
        payload = {
            "source": item.source,
            "metadata": item.metadata,
            "created_at": _utc_now_iso(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n<!-- memory-entry "
                + json.dumps(payload, ensure_ascii=False, sort_keys=True)
                + " -->\n"
            )
            handle.write(item.content.strip() + "\n")
            handle.write("<!-- /memory-entry -->\n")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-markdown-memory-store/v1",
            "backend_kind": "markdown",
            "backend": storage_backend_manifest(
                role="memory",
                kind="markdown",
                location=str(self.path),
                capabilities=("keyword", "semantic", "vector"),
            ),
            "root": str(self.root),
            "path": str(self.path),
            "record_count": len(self._records()),
            "strategy_vector_index": True,
        }

    def _records(self) -> tuple[MemoryRecord, ...]:
        records: list[MemoryRecord] = []
        parsed_paths: set[Path] = set()
        for path in sorted(self.root.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            entries = _parse_markdown_entries(text)
            if entries:
                parsed_paths.add(path)
                for entry in entries:
                    records.append(
                        MemoryRecord(
                            content=entry["content"],
                            source=str(entry.get("source") or path),
                            metadata=dict(entry.get("metadata") or {}),
                            created_at=str(entry.get("created_at") or ""),
                        )
                    )
        for path in sorted(self.root.rglob("*.md")):
            if path in parsed_paths:
                continue
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            records.append(
                MemoryRecord(
                    content=text,
                    source=str(path),
                    metadata={"path": str(path)},
                    created_at="",
                )
            )
        return tuple(records)


_MARKDOWN_ENTRY_RE = re.compile(
    r"<!--\s*memory-entry\s+({.*?})\s*-->\s*(.*?)\s*<!--\s*/memory-entry\s*-->",
    re.DOTALL,
)

_MEMORY_LEAK_PATTERNS = (
    re.compile(r"https?://[^\s)<>'\"]+", re.IGNORECASE),
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    re.compile(
        r"\b[a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*"
        r"\.(?:com|net|org|io|cn|co|me|edu|gov|info|biz|cc|tv|app|dev|cloud|"
        r"xyz|top|tech|sh|us|uk|jp|de|fr|local|localhost)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\blocalhost\b", re.IGNORECASE),
    re.compile(r"\bflag\{[^}\s]{4,}\}", re.IGNORECASE),
    re.compile(
        r"\b(?:password|passwd|pwd|secret|token|apikey|api_key|credential)s?\s*[:=]\s*[^\s,;]{4,}",
        re.IGNORECASE,
    ),
)


def _parse_markdown_entries(text: str) -> tuple[dict[str, Any], ...]:
    entries: list[dict[str, Any]] = []
    for match in _MARKDOWN_ENTRY_RE.finditer(text):
        try:
            metadata = json.loads(match.group(1))
        except json.JSONDecodeError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        entries.append(
            {
                "content": match.group(2).strip(),
                "source": metadata.get("source", ""),
                "metadata": metadata.get("metadata", {}),
                "created_at": metadata.get("created_at", ""),
            }
        )
    return tuple(entries)


def _rank_memory_records(
    query: str,
    records: list[MemoryRecord],
    *,
    limit: int,
    query_vector: tuple[float, ...] = (),
) -> tuple[MemoryHit, ...]:
    if not records:
        return ()
    documents = _memory_search_documents(records)
    keyword_scores = _keyword_scores(query, documents)
    strategy_scores = _strategy_vector_scores(query, records, query_vector=query_vector)
    if not query.strip() and not query_vector:
        return tuple(record.hit(score=0.0) for record in records[:limit])
    scored: list[tuple[float, str, str, MemoryRecord]] = []
    for record in records:
        keyword_score = keyword_scores.get(id(record), 0.0)
        strategy_score = strategy_scores.get(id(record), 0.0)
        score = (keyword_score * 0.72) + (strategy_score * 0.28)
        if score <= 0:
            continue
        scored.append((score, record.created_at, record.source, record))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return tuple(record.hit(score=_clamp01(score)) for score, _, _, record in scored[:limit])


async def _rank_memory_records_semantic(
    query: str,
    records: list[MemoryRecord],
    *,
    provider: EmbeddingProviderPort,
    model: str = "",
    dimensions: int = 0,
    limit: int,
    query_vector: tuple[float, ...] = (),
) -> tuple[MemoryHit, ...]:
    if not records:
        return ()
    documents = _memory_search_documents(records)
    ranked = await rank_semantic_documents(
        query,
        documents,
        provider=provider,
        model=model,
        dimensions=dimensions,
        limit=0,
    )
    if not query.strip() and not query_vector:
        return tuple(record.hit(score=0.0) for record in records[:limit])
    if not ranked:
        return _rank_memory_records(query, records, limit=limit, query_vector=query_vector)
    semantic_scores = {id(hit.item): _clamp01(hit.score) for hit in ranked}
    keyword_scores = _keyword_scores(query, documents)
    strategy_scores = _strategy_vector_scores(query, records, query_vector=query_vector)
    scored: list[tuple[float, str, str, MemoryRecord]] = []
    for record in records:
        semantic_score = semantic_scores.get(id(record), 0.0)
        keyword_score = keyword_scores.get(id(record), 0.0)
        strategy_score = strategy_scores.get(id(record), 0.0)
        score = (semantic_score * 0.56) + (keyword_score * 0.26) + (strategy_score * 0.18)
        if score <= 0:
            continue
        scored.append((score, record.created_at, record.source, record))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return tuple(record.hit(score=_clamp01(score)) for score, _, _, record in scored[:limit])


def _memory_search_documents(records: list[MemoryRecord]) -> tuple[SearchDocument[MemoryRecord], ...]:
    return tuple(
        SearchDocument(
            item=record,
            text=_memory_record_search_text(record),
            name=record.source,
        )
        for record in records
    )


def _memory_record_search_text(record: MemoryRecord) -> str:
    profile = record.entity_profile
    metadata_text = _memory_record_metadata_search_text(record.metadata)
    return " ".join(
        (
            record.content,
            record.source,
            metadata_text,
            " ".join(profile.tags),
            " ".join(profile.potential_questions),
        )
    )


def _memory_record_metadata_search_text(metadata: dict[str, Any]) -> str:
    ignored = {
        "memory_entity_profile",
        "core_pact_scores",
        "core_pact_vector",
        "signals",
        "timeline_diff",
        "metadata",
    }
    parts: list[str] = []
    for key, value in metadata.items():
        if key in ignored:
            continue
        if key.endswith("_manifest") or key.endswith("_json"):
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                parts.append(text[:160])
            continue
        if isinstance(value, bool | int | float):
            parts.append(str(value))
            continue
        if isinstance(value, (list, tuple, set)):
            scalars = [
                str(item).strip()
                for item in value
                if isinstance(item, str | bool | int | float) and str(item).strip()
            ]
            parts.extend(item[:80] for item in scalars[:8])
    return " ".join(parts)


def _keyword_scores(
    query: str,
    documents: tuple[SearchDocument[MemoryRecord], ...],
) -> dict[int, float]:
    if not query.strip() or not documents:
        return {}
    ranked = rank_documents(query, documents, limit=0)
    return {id(record): max(0.001, 1.0 / (index + 1)) for index, record in enumerate(ranked)}


def _strategy_vector_scores(
    query: str,
    records: list[MemoryRecord],
    *,
    query_vector: tuple[float, ...] = (),
) -> dict[int, float]:
    vector = _resolve_query_core_pact_vector(query, query_vector)
    if not vector:
        return {}
    scores: dict[int, float] = {}
    for record in records:
        score = _cosine_score(vector, record.core_pact_vector)
        if score > 0:
            scores[id(record)] = score
    return scores


def _resolve_query_core_pact_vector(query: str, query_vector: tuple[float, ...]) -> tuple[float, ...]:
    if len(query_vector) == len(CORE_PACT_SCORE_KEYS):
        return tuple(_clamp01(value) for value in query_vector)
    if not query.strip():
        return ()
    return _core_pact_vector(_infer_core_pact_scores(query, {}, query_hint=query))


def _cosine_score(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return _clamp01(numerator / (left_norm * right_norm))


def _matches_filters(record: MemoryRecord, filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if key == "source":
            actual = record.source
        else:
            actual = record.metadata.get(key)
        if actual != expected:
            return False
    return True


def _store_supports_route(spec: MemoryStoreSpec, route: MemoryRoute) -> bool:
    if route.tags and not all(tag in spec.tags for tag in route.tags):
        return False
    if route.namespace and spec.namespaces and route.namespace not in spec.namespaces:
        return False
    mode = route.mode or "hybrid"
    if mode == "keyword":
        return spec.supports_keyword
    if mode == "semantic":
        return spec.supports_semantic
    if mode == "vector":
        return spec.supports_vector or spec.backend_kind in {"vector", "product"}
    if mode == "graph":
        return spec.supports_graph or spec.backend_kind in {"graph", "product"}
    if mode == "hybrid":
        return spec.supports_keyword or spec.supports_semantic or spec.supports_vector or spec.supports_graph
    return True


def _store_supports_strategy_vector(store: MemoryPort) -> bool:
    if isinstance(store, (InMemoryMemoryStore, SQLiteMemoryStore, MarkdownMemoryStore)):
        return True
    manifest = getattr(store, "manifest", None)
    if not callable(manifest):
        return False
    try:
        value = manifest()
    except Exception:
        return False
    return bool(isinstance(value, dict) and value.get("strategy_vector_index"))


def _memory_query_audit_manifest(query: MemoryQuery) -> dict[str, Any]:
    return {
        "schema_version": "agent-core-memory-query-audit/v1",
        "query_bytes": len(query.query.encode("utf-8")),
        "query_sha256": sha256(query.query.encode("utf-8")).hexdigest()
        if query.query
        else "",
        "limit": query.limit,
        "filters": dict(query.filters),
        "mode": query.mode,
        "namespace": query.namespace,
        "has_vector": bool(query.vector),
        "vector_dimensions": len(query.vector),
        "entities": list(query.entities),
        "min_score": query.min_score,
    }


def _infer_memory_backend_kind(store: MemoryPort) -> MemoryBackendKind:
    if isinstance(store, InMemoryMemoryStore):
        return "in_memory"
    if isinstance(store, SQLiteMemoryStore):
        return "sqlite"
    if isinstance(store, MarkdownMemoryStore):
        return "markdown"
    manifest = getattr(store, "manifest", None)
    if callable(manifest):
        try:
            value = manifest()
        except Exception:
            value = {}
        if isinstance(value, dict):
            backend_kind = str(value.get("backend_kind") or "")
            if backend_kind in {
                "in_memory",
                "sqlite",
                "markdown",
                "postgres",
                "vector",
                "graph",
                "product",
                "custom",
            }:
                return backend_kind  # type: ignore[return-value]
    return "custom"


def _risk_level_for_scope(scope: Any) -> Literal["low", "medium", "high"]:
    text = str(scope or "").strip().lower()
    if text.startswith("global"):
        return "high"
    if text.startswith("user"):
        return "medium"
    return "low"


def _bucket_subject(scope: str) -> str:
    text = str(scope or "").strip().lower()
    if text.startswith("global:product:"):
        slug = text.split(":", 2)[2].strip()
        return f"global_product_{slug}" if slug and slug != "*" else ""
    if text.startswith("global:skill:"):
        slug = text.split(":", 2)[2].strip()
        return f"global_skill_{slug}" if slug and slug != "*" else ""
    return ""


def _scan_memory_leaks(content: str, source: str, metadata: dict[str, Any]) -> tuple[str, ...]:
    values = [content, source]
    for value in metadata.values():
        if isinstance(value, (str, int, float)):
            values.append(str(value))
        elif isinstance(value, list):
            values.extend(str(item) for item in value if isinstance(item, (str, int, float)))
    text = "\n".join(values)
    hits: list[str] = []
    seen: set[str] = set()
    for pattern in _MEMORY_LEAK_PATTERNS:
        for match in pattern.findall(text):
            value = str(match)
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            hits.append(value)
    return tuple(hits)


def _memory_profile(
    *,
    content: str,
    source: str,
    scope: str,
    bucket: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    summary_key = _normalize_memory_text(content)
    entity_type = _normalize_memory_text(metadata.get("entity_type") or metadata.get("kind") or "memory")
    evidence_state = _normalize_memory_text(
        metadata.get("evidence_state")
        or metadata.get("status")
        or metadata.get("verification_state")
        or ""
    )
    fingerprint = _stable_hash([scope, bucket, source, entity_type, summary_key])
    return {
        "fingerprint": fingerprint,
        "scope": scope,
        "bucket": bucket,
        "source": source,
        "entity_type": entity_type,
        "summary_key": summary_key[:1000],
        "evidence_state": evidence_state,
    }


def _normalize_memory_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9_ .:/-]+", "", text)
    return text.strip()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()

