"""Knowledge and midterm timeline recall primitives for the loop runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_core.context import ContextInjection, ContextMaterial, ContextMaterialQuery, ContextMaterialStorePort
from agent_core.mcp import MCPCenter, MCPContextMaterialRequest
from agent_core.prompt import PromptBucketRole
from agent_core.search import SearchDocument, rank_documents
from agent_core.timeline import TimelineStorePort


@dataclass(frozen=True)
class KnowledgeRecallRequest:
    query: str = ""
    topics: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    limit: int = 5
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_query(self) -> str:
        return " ".join(
            part
            for part in (self.query, " ".join(self.topics), " ".join(self.keywords))
            if str(part).strip()
        ).strip()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-knowledge-recall-request/v1",
            "query": self.query,
            "topics": list(self.topics),
            "keywords": list(self.keywords),
            "limit": self.limit,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class KnowledgeRecallResult:
    request: KnowledgeRecallRequest
    materials: tuple[ContextMaterial, ...] = ()
    session_summary: str = ""
    fallback_hints: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def injections(self) -> tuple[ContextInjection, ...]:
        if not self.materials and not self.session_summary and not self.fallback_hints:
            return ()
        return (
            ContextInjection(
                name="knowledge_recall",
                content=self.render_prompt(),
                target=PromptBucketRole.TIMELINE_OPEN,
                source="knowledge",
                priority=78,
                metadata=self.manifest(),
            ),
        )

    def render_prompt(self) -> str:
        lines = ["[knowledge_recall]"]
        if self.session_summary:
            lines.extend(("[accumulated_search_summary]", self.session_summary))
        if self.fallback_hints:
            lines.append("[fallback_hints]")
            lines.extend(f"- {hint}" for hint in self.fallback_hints)
        if self.materials:
            lines.append("[latest_hits]")
        for material in self.materials:
            source = str(material.metadata.get("source") or material.metadata.get("kind") or "knowledge")
            lines.append(f"- {material.name} source={source}: {material.content}")
        return "\n".join(lines)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-knowledge-recall-result/v1",
            "enabled": True,
            "query": self.request.normalized_query(),
            "request": self.request.manifest(),
            "hit_count": len(self.materials),
            "hits": [material.manifest() for material in self.materials],
            "session_summary_bytes": len(self.session_summary.encode("utf-8")),
            "fallback_hints": list(self.fallback_hints),
            "metadata": dict(self.metadata),
        }


class KnowledgeRecallPort(Protocol):
    async def recall(self, request: KnowledgeRecallRequest) -> KnowledgeRecallResult:
        """Recall prompt-safe knowledge materials for the next turn."""


class NullKnowledgeRecall:
    async def recall(self, request: KnowledgeRecallRequest) -> KnowledgeRecallResult:
        return KnowledgeRecallResult(
            request=request,
            metadata={"enabled": False, "reason": "knowledge_recall_unavailable"},
        )


class DefaultKnowledgeRecall:
    """Default keyword/BM25 recall over context material and MCP knowledge sources."""

    def __init__(
        self,
        *,
        store: ContextMaterialStorePort | None = None,
        mcp: MCPCenter | None = None,
        max_summary_bytes: int = 10 * 1024,
    ) -> None:
        self.store = store
        self.mcp = mcp
        self.max_summary_bytes = max(1024, int(max_summary_bytes))
        self._session_summary = ""
        self._search_count = 0
        self._failure_count = 0

    async def recall(self, request: KnowledgeRecallRequest) -> KnowledgeRecallResult:
        query_text = request.normalized_query()
        if not query_text:
            return KnowledgeRecallResult(
                request=request,
                metadata={"enabled": False, "reason": "empty_query"},
            )
        materials: list[ContextMaterial] = []
        store_manifest: dict[str, Any] = {}
        mcp_manifest: dict[str, Any] = {}
        limit = max(1, int(request.limit or 5))
        expanded_queries = _knowledge_search_queries(request)
        errors: list[str] = []
        self._search_count += 1
        if self.store is not None:
            store_hits = 0
            store_queries = []
            for candidate_query in expanded_queries:
                remaining = limit - len(materials)
                if remaining <= 0:
                    break
                query_limit = max(1, min(limit, remaining))
                query = ContextMaterialQuery(
                    query=candidate_query,
                    limit=query_limit,
                    mode="hybrid",
                )
                store_queries.append(query.manifest())
                plan_search = getattr(self.store, "plan_search", None)
                if callable(plan_search) and not store_manifest.get("plan"):
                    plan = plan_search(query)
                    manifest = getattr(plan, "manifest", None)
                    if callable(manifest):
                        store_manifest["plan"] = manifest()
                try:
                    found = await self.store.search(query)
                except Exception as exc:
                    errors.append(f"context_material_store search failed for '{candidate_query}': {exc}")
                    continue
                materials.extend(found)
                store_hits += len(found)
            store_manifest["hit_count"] = store_hits
            store_manifest["queries"] = store_queries
        if self.mcp is not None and len(materials) < limit:
            remaining = max(1, limit - len(materials))
            try:
                result = await self.mcp.context_materials(
                    MCPContextMaterialRequest(
                        query=query_text,
                        limit=remaining,
                        role="knowledge",
                        priority=65,
                        metadata={"source": "knowledge_recall"},
                    )
                )
            except Exception as exc:
                errors.append(f"mcp context material search failed: {exc}")
            else:
                materials.extend(result.materials)
                mcp_manifest = result.manifest()
        ranked = _rank_materials(query_text, _dedupe_materials(tuple(materials)), limit=limit)
        fallback_hints = _knowledge_fallback_hints(request, ranked, errors=tuple(errors))
        if errors:
            self._failure_count += 1
        if ranked:
            self._session_summary = _merge_knowledge_summary(
                self._session_summary,
                query=query_text,
                materials=ranked,
                max_bytes=self.max_summary_bytes,
            )
        return KnowledgeRecallResult(
            request=request,
            materials=ranked,
            session_summary=self._session_summary,
            fallback_hints=fallback_hints,
            metadata={
                "strategy": "default_keyword_bm25_stateful",
                "search_count": self._search_count,
                "failure_count": self._failure_count,
                "expanded_queries": list(expanded_queries),
                "context_material_store": store_manifest,
                "mcp_context_materials": mcp_manifest,
                "errors": errors,
                "summary_budget_bytes": self.max_summary_bytes,
            },
        )


@dataclass(frozen=True)
class MidtermTimelineRecallRequest:
    query: str = ""
    topics: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    limit: int = 6
    exclude_recent: int = 3
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_query(self) -> str:
        return " ".join(
            part
            for part in (self.query, " ".join(self.topics), " ".join(self.keywords))
            if str(part).strip()
        ).strip()

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-midterm-timeline-recall-request/v1",
            "query": self.query,
            "topics": list(self.topics),
            "keywords": list(self.keywords),
            "limit": self.limit,
            "exclude_recent": self.exclude_recent,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MidtermTimelineRecallResult:
    request: MidtermTimelineRecallRequest
    entries: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def injections(self) -> tuple[ContextInjection, ...]:
        if not self.entries:
            return ()
        return (
            ContextInjection(
                name="midterm_timeline_recall",
                content=self.render_prompt(),
                target=PromptBucketRole.TIMELINE_OPEN,
                source="timeline",
                priority=74,
                metadata=self.manifest(),
            ),
        )

    def render_prompt(self) -> str:
        return "[midterm_timeline_recall]\n" + "\n".join(f"- {entry}" for entry in self.entries)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-midterm-timeline-recall-result/v1",
            "enabled": True,
            "query": self.request.normalized_query(),
            "request": self.request.manifest(),
            "hit_count": len(self.entries),
            "entries": list(self.entries),
            "metadata": dict(self.metadata),
        }


class MidtermTimelineRecallPort(Protocol):
    async def recall(self, request: MidtermTimelineRecallRequest) -> MidtermTimelineRecallResult:
        """Recall older raw timeline facts or prompt archive refs for the next turn."""


class NullMidtermTimelineRecall:
    async def recall(self, request: MidtermTimelineRecallRequest) -> MidtermTimelineRecallResult:
        return MidtermTimelineRecallResult(
            request=request,
            metadata={"enabled": False, "reason": "midterm_timeline_recall_unavailable"},
        )


class DefaultMidtermTimelineRecall:
    """Default BM25 recall over the raw timeline fact stream and archive refs."""

    def __init__(self, timeline: TimelineStorePort) -> None:
        self.timeline = timeline

    async def recall(self, request: MidtermTimelineRecallRequest) -> MidtermTimelineRecallResult:
        query_text = request.normalized_query()
        if not query_text:
            return MidtermTimelineRecallResult(
                request=request,
                metadata={"enabled": False, "reason": "empty_query"},
            )
        active = tuple(item for item in self.timeline.items if not item.deleted)
        recent = max(0, int(request.exclude_recent))
        candidates = active[:-recent] if recent and len(active) > recent else active
        documents = tuple(
            SearchDocument(
                item=item,
                text=" ".join((item.kind, item.content, str(item.metadata))),
                name=item.item_id,
            )
            for item in candidates
        )
        ranked = rank_documents(query_text, documents, limit=max(1, int(request.limit or 6)))
        entries = tuple(item.render() for item in ranked)
        archive_refs = tuple(getattr(self.timeline, "archive_ref_records", ()) or ())
        if archive_refs and len(entries) < max(1, int(request.limit or 6)):
            remaining = max(0, int(request.limit or 6) - len(entries))
            archive_documents = tuple(
                SearchDocument(
                    item=ref,
                    text=" ".join((ref.archive_id, ref.reason, ref.summary_preview)),
                    name=ref.archive_id,
                )
                for ref in archive_refs
            )
            archive_hits = rank_documents(query_text, archive_documents, limit=remaining)
            entries = (
                *entries,
                *(f"[archive_ref] {ref.render()}" for ref in archive_hits),
            )
        return MidtermTimelineRecallResult(
            request=request,
            entries=entries,
            metadata={
                "strategy": "default_timeline_bm25",
                "raw_candidate_count": len(candidates),
                "archive_ref_count": len(archive_refs),
            },
        )


def _rank_materials(
    query: str,
    materials: tuple[ContextMaterial, ...],
    *,
    limit: int,
) -> tuple[ContextMaterial, ...]:
    if not materials:
        return ()
    documents = tuple(
        SearchDocument(
            item=material,
            text=" ".join(
                (
                    material.name,
                    material.role,
                    material.content,
                    " ".join(str(value) for value in material.metadata.values()),
                )
            ),
            priority=material.priority,
            name=material.name,
        )
        for material in materials
    )
    return tuple(rank_documents(query, documents, limit=limit))


def _knowledge_search_queries(request: KnowledgeRecallRequest) -> tuple[str, ...]:
    candidates: list[str] = []
    base = request.normalized_query()
    if base:
        candidates.append(base)
    topics = tuple(str(item).strip() for item in request.topics if str(item).strip())
    keywords = tuple(str(item).strip() for item in request.keywords if str(item).strip())
    if topics:
        candidates.append(" ".join(topics))
    if keywords:
        candidates.append(" ".join(keywords))
        candidates.extend(keywords[:3])
    query_rewrite = request.metadata.get("query_rewrite")
    if query_rewrite:
        candidates.append(str(query_rewrite))
    return _dedupe_texts(tuple(candidates)) or ((base,) if base else ())


def _dedupe_materials(materials: tuple[ContextMaterial, ...]) -> tuple[ContextMaterial, ...]:
    deduped: list[ContextMaterial] = []
    seen: set[tuple[str, str]] = set()
    for material in materials:
        key = (material.name, material.sha256)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(material)
    return tuple(deduped)


def _merge_knowledge_summary(
    existing: str,
    *,
    query: str,
    materials: tuple[ContextMaterial, ...],
    max_bytes: int,
) -> str:
    lines = [line for line in (existing.strip(), _knowledge_round_summary(query, materials)) if line]
    merged = "\n\n".join(lines).strip()
    if len(merged.encode("utf-8")) <= max_bytes:
        return merged
    return _compress_text_for_query(merged, query=query, max_bytes=max_bytes)


def _knowledge_round_summary(query: str, materials: tuple[ContextMaterial, ...]) -> str:
    if not materials:
        return ""
    lines = [f"=== query: {_shrink_text(query, 180)} ==="]
    for material in materials:
        source = str(material.metadata.get("source") or material.metadata.get("store") or material.role or "knowledge")
        lines.append(f"- {material.name} source={source}: {_shrink_text(material.content, 700)}")
    return "\n".join(lines)


def _knowledge_fallback_hints(
    request: KnowledgeRecallRequest,
    materials: tuple[ContextMaterial, ...],
    *,
    errors: tuple[str, ...],
) -> tuple[str, ...]:
    if materials:
        return ()
    query = request.normalized_query()
    hints = []
    if errors:
        hints.append(
            "Knowledge search failed for the current semantic query; rewrite the query or use another source."
        )
    else:
        hints.append(
            "Knowledge search returned no results for the current semantic query; try a narrower keyword query or a broader semantic rewrite."
        )
    if query:
        hints.append(f"Current knowledge query: {_shrink_text(query, 180)}")
    hints.append(
        "If the knowledge base still has no relevant facts, fall back to web/search/tool evidence instead of repeating the same knowledge query."
    )
    return tuple(hints)


def _compress_text_for_query(text: str, *, query: str, max_bytes: int) -> str:
    query_terms = _terms(query)
    units = [unit.strip() for unit in text.splitlines() if unit.strip()]
    scored = sorted(
        (
            (_unit_score(unit, query_terms, index, len(units)), index, unit)
            for index, unit in enumerate(units)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    selected: list[tuple[int, str]] = []
    used = len("[...knowledge summary compacted...]\n".encode("utf-8"))
    for _, index, unit in scored:
        unit_bytes = len((unit + "\n").encode("utf-8"))
        if used + unit_bytes > max_bytes:
            continue
        selected.append((index, unit))
        used += unit_bytes
    selected.sort(key=lambda item: item[0])
    if not selected and units:
        budget = max(64, max_bytes - 40)
        return "[...knowledge summary compacted...]\n" + _shrink_to_bytes(units[0], budget)
    return "[...knowledge summary compacted...]\n" + "\n".join(unit for _, unit in selected)


def _unit_score(unit: str, query_terms: frozenset[str], index: int, total: int) -> float:
    terms = _terms(unit)
    overlap = len(terms & query_terms) / max(1, len(query_terms)) if query_terms else 0.0
    recency = index / max(1, total - 1) if total > 1 else 1.0
    heading = 0.25 if unit.startswith("===") else 0.0
    return overlap * 2.0 + recency * 0.25 + heading


def _terms(text: str) -> frozenset[str]:
    import re

    return frozenset(
        token
        for token in re.findall(r"[a-zA-Z0-9_/-]{3,}", str(text or "").lower())
        if token
    )


def _dedupe_texts(values: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = " ".join(str(value or "").split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return tuple(result)


def _shrink_text(text: str, limit: int) -> str:
    collapsed = " ".join(str(text or "").split())
    limit = max(16, int(limit))
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."


def _shrink_to_bytes(text: str, max_bytes: int) -> str:
    raw = str(text or "").encode("utf-8")
    max_bytes = max(1, int(max_bytes))
    if len(raw) <= max_bytes:
        return str(text or "")
    return raw[: max_bytes - 3].decode("utf-8", errors="ignore").rstrip() + "..."
