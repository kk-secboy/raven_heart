from __future__ import annotations

import pytest

from agent_core.actions import ActionRegistry
from agent_core.memory import (
    ExternalMemoryStore,
    InMemoryMemoryStore,
    MarkdownMemoryStore,
    MemoryFlushBuffer,
    MemoryFlushSignal,
    MemoryCenter,
    MemoryHit,
    MemoryGovernanceDeniedError,
    MemoryQuery,
    MemoryRoute,
    MemoryStoreNotFoundError,
    MemoryStoreSpec,
    MemoryWrite,
    RuleBasedMemoryGovernance,
    SQLiteMemoryStore,
    build_memory_injection,
    infer_memory_recall_intent,
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


class _FailingMemoryStore:
    async def search(self, query: MemoryQuery) -> tuple[MemoryHit, ...]:
        raise RuntimeError("memory search failed")

    async def write(self, item: MemoryWrite) -> None:
        raise RuntimeError("memory write failed")


def test_memory_injection_routes_and_reranks_hits_by_intent() -> None:
    hits = (
        MemoryHit(
            content="Operator must verify admin panel exposure before reporting.",
            score=0.82,
            source="policy",
            metadata={"p_score": 0.9, "r_score": 0.8, "o_score": 0.95},
        ),
        MemoryHit(
            content="Previous finding about admin panel came from an unverified guess.",
            score=0.9,
            source="note",
            metadata={"o_score": 0.2, "r_score": 0.88},
        ),
        MemoryHit(
            content="Use passive reconnaissance first, then check /admin.",
            score=0.74,
            source="playbook",
            metadata={"a_score": 0.95, "r_score": 0.7},
        ),
    )

    intent = infer_memory_recall_intent("verify admin evidence before report")
    result = build_memory_injection(hits, query="verify admin evidence before report", intent=intent)

    assert intent == "fact_check"
    assert "[ reliability_warning ]" in result.content
    assert "[ must_aware ]" in result.content
    assert "[ action_tips ]" in result.content
    assert result.manifest()["routes"][0] == "reliability_warning"
    assert result.manifest()["selected_count"] == 3


@pytest.mark.asyncio
async def test_memory_flush_buffer_batches_done_signal_and_dedupes() -> None:
    buffer = MemoryFlushBuffer(byte_threshold=10_000)

    assert await buffer.observe(
        MemoryFlushSignal(
            content="tool found admin panel",
            run_id="run",
            turn_id="turn-1",
            iteration=1,
            status="tool_finished",
        )
    ) == ()
    writes = await buffer.observe(
        MemoryFlushSignal(
            content="second fact confirms admin panel exposure",
            run_id="run",
            turn_id="turn-2",
            iteration=2,
            status="finished",
            is_done=True,
        )
    )

    assert len(writes) == 1
    assert "tool found admin panel" in writes[0].content
    assert writes[0].metadata["signal_count"] == 2
    assert buffer.last_manifest["flushed"] is True

    duplicate = await buffer.observe(
        MemoryFlushSignal(
            content=writes[0].content,
            run_id="run",
            turn_id="turn-3",
            iteration=3,
            status="finished",
            is_done=True,
        )
    )
    assert duplicate == ()


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
async def test_builtin_memory_store_enriches_writes_with_core_pact_profile() -> None:
    store = InMemoryMemoryStore()
    await store.write(
        MemoryWrite(
            content="Use passive reconnaissance before active scans.",
            source="operator-note",
            metadata={
                "kind": "guidance",
                "core_pact_scores": {
                    "c": 0.2,
                    "o": 0.8,
                    "r": 0.7,
                    "e": 0.0,
                    "p": 0.9,
                    "a": 0.95,
                    "t": 0.6,
                },
            },
        )
    )
    await store.write(
        MemoryWrite(
            content="Unverified admin panel rumor.",
            source="scratch",
            metadata={
                "kind": "note",
                "core_pact_scores": {
                    "c": 0.1,
                    "o": 0.1,
                    "r": 0.2,
                    "e": 0.0,
                    "p": 0.1,
                    "a": 0.1,
                    "t": 0.2,
                },
            },
        )
    )

    profile = store.records[0].metadata["memory_entity_profile"]
    vector = tuple(profile["core_pact_vector"])
    hits = await store.search(MemoryQuery(query="", mode="vector", vector=vector, limit=2))

    assert profile["schema_version"] == "agent-core-memory-entity-profile/v1"
    assert len(vector) == 7
    assert store.records[0].metadata["potential_questions"]
    assert hits[0].source == "operator-note"
    assert hits[0].metadata["core_pact_vector"] == list(vector)


@pytest.mark.asyncio
async def test_persistent_memory_stores_keep_profile_and_strategy_vector(tmp_path) -> None:
    sqlite = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    markdown = MarkdownMemoryStore(tmp_path / "notes")
    item = MemoryWrite(
        content="Always verify callback binding before reporting CSRF.",
        source="operator-policy",
        metadata={
            "kind": "policy",
            "core_pact_scores": {
                "c": 0.4,
                "o": 0.9,
                "r": 0.85,
                "e": 0.0,
                "p": 0.95,
                "a": 0.8,
                "t": 0.55,
            },
        },
    )
    await sqlite.write(item)
    await markdown.write(item)

    restored_sqlite = SQLiteMemoryStore(tmp_path / "memory.sqlite")
    sqlite_hits = await restored_sqlite.search(MemoryQuery(query="callback csrf", mode="hybrid"))
    vector = tuple(sqlite_hits[0].metadata["core_pact_vector"])
    markdown_hits = await markdown.search(MemoryQuery(query="", mode="vector", vector=vector))

    assert sqlite_hits[0].source == "operator-policy"
    assert sqlite_hits[0].metadata["memory_entity_profile"]["schema_version"] == (
        "agent-core-memory-entity-profile/v1"
    )
    assert markdown_hits[0].source == "operator-policy"
    assert markdown.manifest()["strategy_vector_index"] is True
    assert "vector" in restored_sqlite.manifest()["backend"]["capabilities"]


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
async def test_external_memory_store_wraps_runtime_pg_adapter_with_manifest() -> None:
    adapter = _RecordingMemoryStore(
        (MemoryHit(content="tenant memory hit", score=0.9, source="pg"),)
    )
    external = ExternalMemoryStore(
        adapter,
        name="tenant-pg",
        backend_kind="postgres",
        priority=30,
        namespaces=("tenant-a",),
        supports_vector=True,
        supports_graph=True,
        tags=("durable", "tenant"),
        location="postgres://memory",
        metadata={"dsn_ref": "env:MEMORY_DSN"},
    )
    center = MemoryCenter(default_store="tenant-pg")
    center.register_spec(external.spec, external)

    await center.write(MemoryWrite(content="remember this", metadata={"store": "tenant-pg"}))
    hits = await center.search(
        MemoryQuery(
            query="private lookup",
            mode="vector",
            namespace="tenant-a",
            vector=(0.1, 0.2),
            filters={"store": "tenant-pg", "kind": "note"},
        )
    )
    manifest = center.manifest()["stores"][0]
    external_manifest = external.manifest()

    assert adapter.writes[0].content == "remember this"
    assert adapter.writes[0].metadata == {}
    assert adapter.queries[0].filters == {"kind": "note"}
    assert hits[0].metadata["store"] == "tenant-pg"
    assert manifest["backend_kind"] == "postgres"
    assert manifest["core_builtin"] is False
    assert manifest["location"] == "postgres://memory"
    assert manifest["backend"]["kind"] == "postgres"
    assert manifest["backend"]["core_builtin"] is False
    assert manifest["backend"]["transactional"] is True
    assert manifest["backend"]["location"] == "postgres://memory"
    assert "vector" in manifest["backend"]["capabilities"]
    assert "graph" in manifest["backend"]["capabilities"]
    assert external_manifest["schema_version"] == "agent-core-external-memory-store/v1"
    assert external_manifest["store"]["name"] == "tenant-pg"
    assert external_manifest["call_count"] == 2
    assert [call["operation"] for call in external_manifest["calls"]] == ["write", "search"]
    assert external_manifest["calls"][0]["write"]["content_bytes"] == len("remember this")
    assert external_manifest["calls"][0]["write"]["content_sha256"]
    assert "remember this" not in str(external_manifest)
    assert external_manifest["calls"][1]["query"]["mode"] == "vector"
    assert external_manifest["calls"][1]["query"]["query_bytes"] == len("private lookup")
    assert external_manifest["calls"][1]["query"]["query_sha256"]
    assert external_manifest["calls"][1]["hit_count"] == 1
    assert "private lookup" not in str(external_manifest)
    assert external_manifest["adapter"] == {}


@pytest.mark.asyncio
async def test_external_memory_store_records_failed_runtime_adapter_calls() -> None:
    external = ExternalMemoryStore(
        _FailingMemoryStore(),
        name="tenant-pg",
        backend_kind="postgres",
        location="postgres://memory",
    )

    with pytest.raises(RuntimeError):
        await external.search(MemoryQuery(query="private lookup"))
    with pytest.raises(RuntimeError):
        await external.write(MemoryWrite(content="secret write"))

    manifest = external.manifest()
    assert manifest["call_count"] == 2
    assert [call["status"] for call in manifest["calls"]] == ["failed", "failed"]
    assert manifest["calls"][0]["operation"] == "search"
    assert manifest["calls"][0]["error"] == "memory search failed"
    assert "private lookup" not in str(manifest)
    assert manifest["calls"][1]["operation"] == "write"
    assert manifest["calls"][1]["error"] == "memory write failed"
    assert "secret write" not in str(manifest)


def test_memory_center_register_spec_requires_named_store() -> None:
    center = MemoryCenter()

    with pytest.raises(ValueError):
        center.register_spec(MemoryStoreSpec(name=""), _RecordingMemoryStore())


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

