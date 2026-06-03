"""Memory and retrieval protocol types."""

from __future__ import annotations

import json
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


@dataclass(frozen=True)
class MemoryWrite:
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


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
                supports_vector=supports_vector,
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
            )
        return _rank_memory_records(query.query, records, limit=query.limit)

    async def write(self, item: MemoryWrite) -> None:
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
            "backend": storage_backend_manifest(role="memory", kind="in_memory"),
            "record_count": len(self.records),
            "semantic_ranking": self.embedding_provider is not None,
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
        return await self.adapter.search(query)

    async def write(self, item: MemoryWrite) -> None:
        await self.adapter.write(item)

    def manifest(self) -> dict[str, Any]:
        adapter_manifest = getattr(self.adapter, "manifest", None)
        return {
            "schema_version": "agent-core-external-memory-store/v1",
            "backend_kind": self._spec.backend_kind,
            "store": self._spec.manifest(),
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
        return _rank_memory_records(query.query, records, limit=query.limit)

    async def write(self, item: MemoryWrite) -> None:
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
                capabilities=("keyword", "semantic"),
            ),
            "path": str(self.path),
            "record_count": len(self._records()),
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
        return _rank_memory_records(query.query, records, limit=query.limit)

    async def write(self, item: MemoryWrite) -> None:
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
                capabilities=("keyword", "semantic"),
            ),
            "root": str(self.root),
            "path": str(self.path),
            "record_count": len(self._records()),
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


def _rank_memory_records(query: str, records: list[MemoryRecord], *, limit: int) -> tuple[MemoryHit, ...]:
    documents = tuple(
        SearchDocument(
            item=record,
            text=" ".join(
                (
                    record.content,
                    record.source,
                    " ".join(str(value) for value in record.metadata.values()),
                )
            ),
            name=record.source,
        )
        for record in records
    )
    ranked = rank_documents(query, documents, limit=limit)
    if query.strip():
        return tuple(record.hit(score=max(0.001, 1.0 / (index + 1))) for index, record in enumerate(ranked))
    return tuple(record.hit(score=0.0) for record in records[:limit])


async def _rank_memory_records_semantic(
    query: str,
    records: list[MemoryRecord],
    *,
    provider: EmbeddingProviderPort,
    model: str = "",
    dimensions: int = 0,
    limit: int,
) -> tuple[MemoryHit, ...]:
    documents = tuple(
        SearchDocument(
            item=record,
            text=" ".join(
                (
                    record.content,
                    record.source,
                    " ".join(str(value) for value in record.metadata.values()),
                )
            ),
            name=record.source,
        )
        for record in records
    )
    ranked = await rank_semantic_documents(
        query,
        documents,
        provider=provider,
        model=model,
        dimensions=dimensions,
        limit=limit,
    )
    if query.strip():
        return tuple(hit.item.hit(score=hit.score) for hit in ranked)
    return tuple(record.hit(score=0.0) for record in records[:limit])


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

