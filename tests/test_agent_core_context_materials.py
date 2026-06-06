from __future__ import annotations

import pytest

from agent_core.context import (
    ContextMaterial,
    ContextMaterialCenter,
    ContextMaterialQuery,
    ExternalContextMaterialStore,
    InMemoryContextMaterialStore,
    MarkdownContextMaterialStore,
    SQLiteContextMaterialStore,
)
from agent_core.embeddings import EmbeddingRequest, EmbeddingResponse, EmbeddingVector
from agent_core.knowledge import DefaultKnowledgeRecall, KnowledgeRecallRequest


class _FailingContextMaterialStore:
    async def search(self, query: ContextMaterialQuery) -> tuple[ContextMaterial, ...]:
        raise RuntimeError("context search failed")

    async def write(self, material: ContextMaterial) -> None:
        raise RuntimeError("context write failed")


class _FailingEmbeddingProvider:
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise RuntimeError("embedding batch failed")


class _CountingEmbeddingProvider:
    def __init__(self) -> None:
        self.input_counts: list[int] = []

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        self.input_counts.append(len(request.inputs))
        return EmbeddingResponse(
            vectors=tuple(
                EmbeddingVector(values=(1.0, float(index + 1)), index=index, name=item.name)
                for index, item in enumerate(request.inputs)
            )
        )


@pytest.mark.asyncio
async def test_default_knowledge_recall_accumulates_summary_and_expands_semantic_query() -> None:
    store = InMemoryContextMaterialStore(
        (
            ContextMaterial(
                name="csrf-session",
                content="Validate session binding before accepting csrf callback tokens.",
                role="knowledge",
                priority=50,
                metadata={"source": "kb"},
            ),
        )
    )
    recall = DefaultKnowledgeRecall(store=store, max_summary_bytes=1024)

    result = await recall.recall(
        KnowledgeRecallRequest(
            query="callback token validation",
            topics=("csrf session",),
            keywords=("csrf", "session"),
            limit=3,
        )
    )

    prompt = result.render_prompt()
    manifest = result.manifest()
    assert result.materials
    assert "[accumulated_search_summary]" in prompt
    assert "Validate session binding" in prompt
    assert manifest["metadata"]["strategy"] == "default_keyword_bm25_stateful"
    assert manifest["metadata"]["search_count"] == 1
    assert "csrf" in manifest["metadata"]["expanded_queries"]


@pytest.mark.asyncio
async def test_default_knowledge_recall_injects_fallback_hints_for_empty_results() -> None:
    recall = DefaultKnowledgeRecall(store=InMemoryContextMaterialStore(()))

    result = await recall.recall(KnowledgeRecallRequest(query="missing kb topic", limit=3))

    assert result.materials == ()
    assert result.fallback_hints
    assert result.injections
    prompt = result.injections[0].content
    assert "[fallback_hints]" in prompt
    assert "fall back to web/search/tool evidence" in prompt


@pytest.mark.asyncio
async def test_default_knowledge_recall_falls_back_to_keyword_when_embedding_fails() -> None:
    store = InMemoryContextMaterialStore(
        (
            ContextMaterial(
                name="auth-callback-kb",
                content="Auth callback nonce must be persisted and replay checked.",
                role="knowledge",
                priority=70,
                metadata={"tags": ("auth", "callback")},
            ),
        ),
        embedding_provider=_FailingEmbeddingProvider(),
    )
    recall = DefaultKnowledgeRecall(store=store)

    result = await recall.recall(KnowledgeRecallRequest(query="auth callback nonce", limit=3))

    assert [material.name for material in result.materials] == ["auth-callback-kb"]
    assert result.metadata["failure_count"] == 0
    assert "Auth callback nonce" in result.render_prompt()


@pytest.mark.asyncio
async def test_in_memory_context_material_store_prefilters_semantic_candidates() -> None:
    embedding = _CountingEmbeddingProvider()
    store = InMemoryContextMaterialStore(
        tuple(
            ContextMaterial(
                name=f"kb-{index:02d}",
                content=f"auth callback candidate {index}",
                role="knowledge",
                priority=100 - index,
            )
            for index in range(50)
        ),
        embedding_provider=embedding,
    )

    results = await store.search(ContextMaterialQuery(query="auth callback", limit=3, mode="hybrid"))

    assert len(results) == 3
    assert embedding.input_counts == [13]


@pytest.mark.asyncio
async def test_sqlite_context_material_store_persists_and_searches_materials(tmp_path) -> None:
    store = SQLiteContextMaterialStore(tmp_path / "context.sqlite")

    await store.write(
        ContextMaterial(
            name="auth_trace",
            content="payment auth callback failed csrf token validation",
            role="timeline",
            priority=4,
            metadata={"tags": ("auth",), "namespace": "tenant-a"},
        )
    )
    await store.write(
        ContextMaterial(
            name="dns_note",
            content="dns propagation completed",
            role="memory",
            priority=1,
            metadata={"tags": ("dns",), "namespace": "tenant-a"},
        )
    )
    reloaded = SQLiteContextMaterialStore(tmp_path / "context.sqlite")

    results = await reloaded.search(
        ContextMaterialQuery(
            query="auth csrf risk",
            filters={"tags": ("auth",)},
            namespace="tenant-a",
        )
    )
    manifest = reloaded.manifest()

    assert [material.name for material in results] == ["auth_trace"]
    assert results[0].metadata["created_at"]
    assert manifest["backend_kind"] == "sqlite"
    assert manifest["backend"]["role"] == "context_material"
    assert manifest["material_count"] == 2
    assert "payment auth callback" not in str(manifest)


@pytest.mark.asyncio
async def test_markdown_context_material_store_persists_and_indexes_documents(tmp_path) -> None:
    store = MarkdownContextMaterialStore(tmp_path)
    (tmp_path / "schema.md").write_text('{"required":["risk"]}', encoding="utf-8")

    await store.write(
        ContextMaterial(
            name="operator_hint",
            content="auth callback needs exploitability summary",
            role="runtime",
            priority=3,
            metadata={"tags": ("auth",)},
        )
    )
    reloaded = MarkdownContextMaterialStore(tmp_path)

    auth_results = await reloaded.search(
        ContextMaterialQuery(query="auth exploitability", filters={"tags": ("auth",)})
    )
    schema_results = await reloaded.search(ContextMaterialQuery(query="required risk"))
    manifest = reloaded.manifest()

    assert [material.name for material in auth_results] == ["operator_hint"]
    assert schema_results[0].name == "schema"
    assert schema_results[0].metadata["source"] == "markdown_document"
    assert manifest["backend_kind"] == "markdown"
    assert manifest["backend"]["role"] == "context_material"
    assert manifest["material_count"] == 2
    assert "auth callback needs" not in str(manifest)


@pytest.mark.asyncio
async def test_context_material_center_routes_and_manifests_external_backends() -> None:
    workspace = InMemoryContextMaterialStore(
        (
            ContextMaterial(
                name="auth_trace",
                content="auth callback csrf validation failed",
                role="timeline",
                priority=3,
                metadata={"tags": ("auth",), "namespace": "tenant-a"},
            ),
            ContextMaterial(
                name="dns_note",
                content="dns propagation completed",
                role="memory",
                priority=1,
                metadata={"tags": ("dns",), "namespace": "tenant-a"},
            ),
        )
    )
    pg_adapter = InMemoryContextMaterialStore(
        (
            ContextMaterial(
                name="risk_schema",
                content='{"required":["risk"]}',
                role="schema",
                priority=5,
                metadata={"tags": ("auth",), "namespace": "tenant-a"},
            ),
        )
    )
    external = ExternalContextMaterialStore(
        pg_adapter,
        name="tenant-pg",
        backend_kind="postgres",
        priority=20,
        namespace="tenant-a",
        supports_vector=True,
        tags=("auth", "durable"),
        location="postgres://context",
    )
    center = ContextMaterialCenter(default_store="workspace")
    center.register(
        "workspace",
        workspace,
        backend_kind="markdown",
        priority=5,
        namespace="tenant-a",
        tags=("auth", "local"),
    )
    center.register_spec(external.spec, external)

    await center.write(
        ContextMaterial(
            name="operator_hint",
            content="auth callback needs exploitability summary",
            role="runtime",
            priority=4,
            metadata={"store": "workspace", "tags": ("auth",), "namespace": "tenant-a"},
        )
    )
    results = await center.search(
        ContextMaterialQuery(
            query="auth risk callback",
            limit=3,
            namespace="tenant-a",
            filters={"tags": ("auth",)},
        )
    )
    manifest = center.manifest()

    assert [material.name for material in results] == [
        "risk_schema",
        "operator_hint",
        "auth_trace",
    ]
    assert results[0].metadata["store"] == "tenant-pg"
    assert manifest["stores"][0]["backend_kind"] == "postgres"
    assert manifest["stores"][0]["backend"]["kind"] == "postgres"
    assert "vector" in manifest["stores"][0]["backend"]["capabilities"]
    assert manifest["call_count"] == 3
    assert [call["operation"] for call in manifest["calls"]] == ["write", "search", "search"]
    assert manifest["calls"][1]["query"]["query_sha256"]
    assert "auth risk callback" not in str(manifest)
    assert external.manifest()["call_count"] == 1


@pytest.mark.asyncio
async def test_context_material_center_continues_when_one_store_search_fails() -> None:
    center = ContextMaterialCenter(default_store="workspace")
    center.register(
        "workspace",
        InMemoryContextMaterialStore(
            (
                ContextMaterial(
                    name="good-kb",
                    content="auth callback evidence remains searchable",
                    role="knowledge",
                    priority=10,
                ),
            )
        ),
        priority=5,
    )
    center.register_spec(
        ExternalContextMaterialStore(
            _FailingContextMaterialStore(),
            name="broken",
            backend_kind="postgres",
            priority=50,
        ).spec,
        _FailingContextMaterialStore(),
    )

    results = await center.search(ContextMaterialQuery(query="auth callback", limit=3))
    manifest = center.manifest()

    assert [material.name for material in results] == ["good-kb"]
    assert [call["status"] for call in manifest["calls"]] == ["failed", "completed"]
    assert "context search failed" in manifest["calls"][0]["error"]


@pytest.mark.asyncio
async def test_external_context_material_store_records_failed_runtime_calls() -> None:
    external = ExternalContextMaterialStore(
        _FailingContextMaterialStore(),
        name="tenant-pg",
        backend_kind="postgres",
    )

    with pytest.raises(RuntimeError):
        await external.search(ContextMaterialQuery(query="private target"))
    with pytest.raises(RuntimeError):
        await external.write(ContextMaterial(name="secret", content="sensitive context"))

    manifest = external.manifest()
    assert manifest["call_count"] == 2
    assert [call["status"] for call in manifest["calls"]] == ["failed", "failed"]
    assert manifest["calls"][0]["query"]["query_sha256"]
    assert "private target" not in str(manifest)
    assert manifest["calls"][1]["material"]["sha256"]
    assert "sensitive context" not in str(manifest)
