from __future__ import annotations

import json

import pytest

from agent_core.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingInput,
    EmbeddingProviderCenter,
    EmbeddingProviderNotFoundError,
    EmbeddingRequest,
    cosine_similarity,
    deterministic_text_embedding,
    rank_semantic_documents,
)
from agent_core.live_adapters import ZhipuEmbedding3Provider
from agent_core.memory import InMemoryMemoryStore, MemoryQuery, MemoryRecord
from agent_core.search import SearchDocument


@pytest.mark.asyncio
async def test_deterministic_embedding_provider_returns_prompt_safe_manifest() -> None:
    provider = DeterministicEmbeddingProvider(default_dimensions=16)

    response = await provider.embed(
        EmbeddingRequest(
            inputs=(EmbeddingInput(text="reset admin password", name="note"),),
            model="local-embedding",
        )
    )
    manifest = response.manifest()

    assert response.model == "local-embedding"
    assert len(response.vectors) == 1
    assert len(response.vectors[0].values) == 16
    assert manifest["vectors"][0]["dimensions"] == 16
    assert "reset admin password" not in str(manifest)


@pytest.mark.asyncio
async def test_embedding_provider_center_routes_records_and_searches_specs() -> None:
    center = EmbeddingProviderCenter(default_provider="local", default_model="embed-small")
    center.register(
        "local",
        DeterministicEmbeddingProvider(default_dimensions=8),
        models=("embed-small",),
        default_model="embed-small",
        priority=10,
        dimensions=8,
        max_inputs=4,
        tags=("local", "semantic"),
    )

    response = await center.embed(
        EmbeddingRequest.from_texts(("alpha beta", "beta gamma"), dimensions=8)
    )
    manifest = center.manifest()

    assert response.metadata["embedding_provider"] == "local"
    assert response.metadata["embedding_route"]["model"] == "embed-small"
    assert center.search(tag="semantic")[0].name == "local"
    assert manifest["call_count"] == 1
    assert manifest["calls"][0]["provider_name"] == "local"
    assert manifest["calls"][0]["input_count"] == 2
    assert manifest["calls"][0]["dimensions"] == 8


@pytest.mark.asyncio
async def test_embedding_provider_center_rejects_unsupported_route() -> None:
    center = EmbeddingProviderCenter(default_provider="local")
    center.register(
        "local",
        DeterministicEmbeddingProvider(default_dimensions=8),
        models=("embed-small",),
        dimensions=8,
        max_inputs=1,
    )

    with pytest.raises(EmbeddingProviderNotFoundError):
        await center.embed(
            EmbeddingRequest.from_texts(
                ("one", "two"),
                model="embed-small",
                dimensions=8,
            )
        )


@pytest.mark.asyncio
async def test_rank_semantic_documents_uses_embedding_provider_scores() -> None:
    provider = DeterministicEmbeddingProvider(default_dimensions=32)
    documents = (
        SearchDocument(item="password", text="reset admin password credential", name="password"),
        SearchDocument(item="network", text="scan open port service", name="network"),
    )

    ranked = await rank_semantic_documents(
        "password reset",
        documents,
        provider=provider,
        dimensions=32,
    )

    assert ranked[0].item == "password"
    assert ranked[0].score > ranked[-1].score


@pytest.mark.asyncio
async def test_in_memory_store_can_use_embedding_semantic_ranking() -> None:
    store = InMemoryMemoryStore(
        (
            MemoryRecord(content="scan open ports and services", source="network"),
            MemoryRecord(content="reset admin password procedure", source="credential"),
        ),
        embedding_provider=DeterministicEmbeddingProvider(default_dimensions=32),
    )

    hits = await store.search(MemoryQuery(query="password reset", mode="semantic", limit=2))

    assert hits[0].source == "credential"
    assert hits[0].score > hits[1].score
    assert store.manifest()["semantic_ranking"] is True


class _FakeZhipuEmbedding3Provider(ZhipuEmbedding3Provider):
    def __init__(self) -> None:
        super().__init__(api_key="test", max_inputs_per_request=4, default_dimensions=3)
        self.input_counts: list[int] = []

    def _post(self, body: bytes) -> bytes:
        payload = json.loads(body.decode("utf-8"))
        inputs = payload["input"]
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        self.input_counts.append(len(texts))
        return json.dumps(
            {
                "model": payload["model"],
                "data": [
                    {"index": index, "embedding": [float(index + 1), 0.0, 1.0]}
                    for index, _ in enumerate(texts)
                ],
                "usage": {"total_tokens": len(texts)},
            }
        ).encode("utf-8")


@pytest.mark.asyncio
async def test_zhipu_embedding_provider_chunks_large_batches() -> None:
    provider = _FakeZhipuEmbedding3Provider()

    response = await provider.embed(
        EmbeddingRequest(
            inputs=tuple(
                EmbeddingInput(text=f"doc {index}", name=f"doc-{index}") for index in range(9)
            )
        )
    )

    assert provider.input_counts == [4, 4, 1]
    assert [vector.index for vector in response.vectors] == list(range(9))
    assert [vector.name for vector in response.vectors] == [f"doc-{index}" for index in range(9)]
    assert response.metadata["usage"]["total_tokens"] == 9


def test_deterministic_text_embedding_and_cosine_are_stable() -> None:
    left = deterministic_text_embedding("alpha beta", dimensions=12)
    same = deterministic_text_embedding("alpha beta", dimensions=12)
    different = deterministic_text_embedding("gamma delta", dimensions=12)

    assert left == same
    assert cosine_similarity(left, same) > cosine_similarity(left, different)
