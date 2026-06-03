from __future__ import annotations

import pytest

from agent_core.actions import ActionRegistry
from agent_core.memory import (
    InMemoryMemoryStore,
    MarkdownMemoryStore,
    MemoryCenter,
    MemoryHit,
    MemoryGovernanceDeniedError,
    MemoryQuery,
    MemoryRoute,
    MemoryStoreNotFoundError,
    MemoryWrite,
    RuleBasedMemoryGovernance,
    SQLiteMemoryStore,
)
from agent_core.prompt import PromptIR
from agent_core.react import ReActExecutor
from agent_core.testing import InMemoryHarness, MockLLMProvider, MockToolRuntime


class _RecordingMemoryStore:
    def __init__(self, hits: tuple[MemoryHit, ...] = ()) -> None:
        self.hits = hits
        self.queries: list[MemoryQuery] = []
        self.writes: list[MemoryWrite] = []

    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        self.queries.append(query)
        return self.hits

    async def write(self, item: MemoryWrite) -> None:
        self.writes.append(item)


@pytest.mark.asyncio
async def test_sqlite_memory_store_writes_searches_filters_and_persists(tmp_path) -> None:
    path = tmp_path / "memory.sqlite"
    store = SQLiteMemoryStore(path)
    await store.write(
        MemoryWrite(
            content="HTTP service exposes admin panel",
            source="scan",
            metadata={"target": "web", "kind": "finding"},
        )
    )
    await store.write(
        MemoryWrite(
            content="SSH service uses key authentication",
            source="scan",
            metadata={"target": "ssh", "kind": "finding"},
        )
    )

    restored = SQLiteMemoryStore(path)
    hits = await restored.search(MemoryQuery(query="admin web", filters={"target": "web"}))

    assert len(hits) == 1
    assert hits[0].content == "HTTP service exposes admin panel"
    assert hits[0].source == "scan"
    assert hits[0].metadata["kind"] == "finding"


@pytest.mark.asyncio
async def test_markdown_memory_store_writes_and_indexes_markdown_files(tmp_path) -> None:
    store = MarkdownMemoryStore(tmp_path)
    await store.write(
        MemoryWrite(
            content="Prefer passive reconnaissance before active probing.",
            source="operator-note",
            metadata={"kind": "guidance"},
        )
    )
    (tmp_path / "playbook.md").write_text(
        "# Playbook\n\nUse evidence packages for final reports.",
        encoding="utf-8",
    )

    guidance = await store.search(MemoryQuery(query="passive reconnaissance"))
    playbook = await store.search(MemoryQuery(query="evidence packages"))

    assert guidance[0].source == "operator-note"
    assert guidance[0].metadata["kind"] == "guidance"
    assert "Playbook" in playbook[0].content
    assert playbook[0].source.endswith("playbook.md")


@pytest.mark.asyncio
async def test_react_executor_injects_sqlite_memory_hits(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    await store.write(MemoryWrite(content="Use port 8443 for admin UI", source="memory"))
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime(),
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        memory=store,
    )

    result = await executor.run("where is admin UI", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert provider.requests[0].messages[-1].name == "memory"
    assert "port 8443" in provider.requests[0].messages[-1].content


@pytest.mark.asyncio
async def test_memory_center_routes_writes_and_aggregates_search(tmp_path) -> None:
    sqlite = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    markdown = MarkdownMemoryStore(tmp_path / "notes")
    center = MemoryCenter(default_store="sqlite")
    center.register("sqlite", sqlite, priority=10, tags=("structured",))
    center.register("markdown", markdown, priority=1, tags=("notes",))

    await center.write(
        MemoryWrite(
            content="Admin panel is exposed on 8443",
            source="scan",
            metadata={"target": "web"},
        )
    )
    await center.write(
        MemoryWrite(
            content="Operator prefers passive reconnaissance first",
            source="note",
            metadata={"store": "markdown", "target": "web"},
        )
    )

    all_hits = await center.search(MemoryQuery(query="admin passive", filters={"target": "web"}))
    markdown_hits = await center.search(
        MemoryQuery(query="passive", filters={"store": "markdown", "target": "web"})
    )

    assert {hit.metadata["store"] for hit in all_hits} == {"sqlite", "markdown"}
    assert markdown_hits[0].source == "note"
    assert markdown_hits[0].metadata["store"] == "markdown"
    assert center.manifest()["stores"][0]["name"] == "sqlite"
    assert center.manifest()["stores"][0]["backend_kind"] == "sqlite"
    assert center.manifest()["stores"][1]["backend_kind"] == "markdown"
    assert center.search_stores(tag="notes")[0].name == "markdown"


@pytest.mark.asyncio
async def test_memory_center_rejects_missing_or_unwritable_store() -> None:
    center = MemoryCenter()
    center.register("readonly", InMemoryMemoryStore(), writable=False)

    with pytest.raises(MemoryStoreNotFoundError):
        await center.write(MemoryWrite(content="x", metadata={"store": "missing"}))
    with pytest.raises(MemoryStoreNotFoundError):
        await center.write(MemoryWrite(content="x", metadata={"store": "readonly"}))
    with pytest.raises(MemoryStoreNotFoundError):
        await center.search(MemoryQuery(query="x", filters={"store": "missing"}))


@pytest.mark.asyncio
async def test_react_executor_injects_memory_center_hits(tmp_path) -> None:
    center = MemoryCenter(default_store="sqlite")
    center.register("sqlite", SQLiteMemoryStore(tmp_path / "memory.sqlite"), priority=10)
    center.register("markdown", MarkdownMemoryStore(tmp_path / "notes"), priority=1)
    await center.write(MemoryWrite(content="Use /admin for web control plane", source="scan"))
    await center.write(
        MemoryWrite(
            content="Final reports should include evidence packages",
            source="note",
            metadata={"store": "markdown"},
        )
    )
    provider = MockLLMProvider([{"action": "finish", "arguments": {"output": "done"}}])
    executor = ReActExecutor(
        provider=provider,
        tool_runtime=MockToolRuntime(),
        action_registry=ActionRegistry(),
        harness=InMemoryHarness(),
        memory=center,
    )

    result = await executor.run("where is web control plane", PromptIR.from_parts(dynamic="task"))

    assert result.status == "completed"
    assert "[memory]" in provider.requests[0].messages[-1].content
    assert "/admin" in provider.requests[0].messages[-1].content


@pytest.mark.asyncio
async def test_memory_center_governance_adds_scope_and_fingerprint() -> None:
    store = InMemoryMemoryStore()
    center = MemoryCenter(
        default_store="local",
        governance=RuleBasedMemoryGovernance(default_scope="project"),
    )
    center.register("local", store)

    await center.write(
        MemoryWrite(
            content="Evidence package must include replay steps",
            source="operator",
            metadata={"kind": "guidance"},
        )
    )

    record = store.records[0]
    assert record.metadata["scope"] == "project"
    assert record.metadata["memory_fingerprint"]
    assert record.metadata["memory_profile"]["entity_type"] == "guidance"


@pytest.mark.asyncio
async def test_memory_governance_denies_user_or_global_target_leaks() -> None:
    center = MemoryCenter(
        default_store="local",
        governance=RuleBasedMemoryGovernance(),
    )
    center.register("local", InMemoryMemoryStore())

    with pytest.raises(MemoryGovernanceDeniedError) as exc:
        await center.write(
            MemoryWrite(
                content="Target is https://prod.example.com and password=secret123",
                source="scan",
                metadata={"scope": "user"},
            )
        )

    assert exc.value.decision.reason == "memory_target_or_secret_leak"
    assert "https://prod.example.com" in exc.value.decision.leaks


@pytest.mark.asyncio
async def test_memory_governance_requires_allowed_global_bucket() -> None:
    store = InMemoryMemoryStore()
    center = MemoryCenter(
        default_store="local",
        governance=RuleBasedMemoryGovernance(allowed_global_buckets=("global_skill_recon",)),
    )
    center.register("local", store)

    with pytest.raises(MemoryGovernanceDeniedError) as exc:
        await center.write(MemoryWrite(content="Reusable note", metadata={"scope": "global"}))
    assert exc.value.decision.reason == "invalid_global_memory_bucket"

    await center.write(
        MemoryWrite(
            content="Prefer passive recon",
            metadata={"scope": "global:skill:recon"},
        )
    )

    assert store.records[0].metadata["bucket"] == "global_skill_recon"
    assert store.records[0].metadata["scope"] == "global:skill:recon"


@pytest.mark.asyncio
async def test_memory_governance_truncates_large_writes() -> None:
    store = InMemoryMemoryStore()
    center = MemoryCenter(
        default_store="local",
        governance=RuleBasedMemoryGovernance(max_content_chars=12),
    )
    center.register("local", store)

    await center.write(MemoryWrite(content="x" * 20, source="summary"))

    assert store.records[0].content == "x" * 12
    assert store.records[0].metadata["memory_truncated"] is True


@pytest.mark.asyncio
async def test_memory_center_governance_manifest_records_prompt_safe_decisions() -> None:
    store = InMemoryMemoryStore()
    center = MemoryCenter(
        default_store="local",
        governance=RuleBasedMemoryGovernance(max_content_chars=12),
    )
    center.register("local", store)

    await center.write(MemoryWrite(content="x" * 20, source="summary"))
    with pytest.raises(MemoryGovernanceDeniedError):
        await center.write(
            MemoryWrite(
                content="Target is https://prod.example.com and password=secret123",
                source="scan",
                metadata={"scope": "user"},
            )
        )
    manifest = center.manifest()["governance"]

    assert manifest["enabled"] is True
    assert manifest["decision_count"] == 2
    assert manifest["decisions"][0]["decision"] == "rewrite"
    assert manifest["decisions"][0]["item"]["content_sha256"]
    assert manifest["decisions"][1]["decision"] == "deny"
    assert manifest["decisions"][1]["leak_count"] >= 1
    assert manifest["decisions"][1]["leak_sha256s"]
    assert "secret123" not in str(manifest)
    assert "prod.example.com" not in str(manifest)


def test_memory_query_route_and_plan_manifest_for_external_backends() -> None:
    center = MemoryCenter(default_store="pg")
    center.register(
        "pg",
        _RecordingMemoryStore(),
        priority=10,
        backend_kind="postgres",
        namespaces=("project-a",),
        tags=("durable", "tenant"),
        metadata={"dsn_ref": "env:MEMORY_DSN"},
    )
    center.register(
        "vector",
        _RecordingMemoryStore(),
        priority=20,
        backend_kind="vector",
        supports_vector=True,
        namespaces=("project-a",),
        tags=("semantic",),
    )

    query = MemoryQuery(
        query="admin panel",
        limit=3,
        mode="vector",
        namespace="project-a",
        vector=(0.1, 0.2, 0.3),
        entities=("service:admin",),
        filters={"tags": ("semantic",), "target": "web"},
        min_score=0.4,
    )
    route = MemoryRoute.from_query(query)
    plan = center.plan_search(query)
    manifest = plan.manifest()

    assert route.namespace == "project-a"
    assert route.tags == ("semantic",)
    assert query.manifest()["vector_dimensions"] == 3
    assert [store.name for store in plan.selected_stores] == ["vector"]
    assert plan.backend_filters == {"target": "web"}
    assert manifest["selected_stores"][0]["backend_kind"] == "vector"
    assert manifest["query"]["has_vector"] is True
    assert manifest["route"]["mode"] == "vector"


@pytest.mark.asyncio
async def test_memory_center_routes_by_mode_namespace_and_filters_min_score() -> None:
    vector = _RecordingMemoryStore(
        (
            MemoryHit(content="close hit", score=0.91, source="vector"),
            MemoryHit(content="weak hit", score=0.2, source="vector"),
        )
    )
    graph = _RecordingMemoryStore((MemoryHit(content="service node", score=0.8, source="graph"),))
    center = MemoryCenter()
    center.register(
        "vector",
        vector,
        backend_kind="vector",
        supports_vector=True,
        namespaces=("project-a",),
    )
    center.register(
        "graph",
        graph,
        backend_kind="graph",
        supports_graph=True,
        namespaces=("project-a",),
    )

    vector_hits = await center.search(
        MemoryQuery(
            query="admin",
            mode="vector",
            namespace="project-a",
            vector=(0.5, 0.1),
            filters={"kind": "finding"},
            min_score=0.5,
        )
    )
    graph_hits = await center.search(
        MemoryQuery(
            query="admin",
            mode="graph",
            namespace="project-a",
            entities=("service:admin",),
        )
    )

    assert [hit.content for hit in vector_hits] == ["close hit"]
    assert vector.queries[0].mode == "vector"
    assert vector.queries[0].namespace == "project-a"
    assert vector.queries[0].filters == {"kind": "finding"}
    assert vector.queries[0].vector == (0.5, 0.1)
    assert [hit.content for hit in graph_hits] == ["service node"]
    assert graph.queries[0].entities == ("service:admin",)


def test_builtin_memory_store_manifests_describe_backend_shape(tmp_path) -> None:
    in_memory = InMemoryMemoryStore()
    sqlite = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    markdown = MarkdownMemoryStore(tmp_path / "notes")

    assert in_memory.manifest()["backend_kind"] == "in_memory"
    assert sqlite.manifest()["schema_version"] == "agent-core-sqlite-memory-store/v1"
    assert sqlite.manifest()["backend_kind"] == "sqlite"
    assert markdown.manifest()["backend_kind"] == "markdown"

