"""Provider-neutral embedding contracts and lightweight semantic ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, Generic, Protocol, TypeVar

from agent_core.search import SearchDocument, tokenize


T = TypeVar("T")


@dataclass(frozen=True)
class EmbeddingInput:
    text: str
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "text_bytes": len(self.text.encode("utf-8")),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EmbeddingRequest:
    inputs: tuple[EmbeddingInput, ...]
    model: str = ""
    dimensions: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_texts(
        cls,
        texts: tuple[str, ...],
        *,
        model: str = "",
        dimensions: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> "EmbeddingRequest":
        return cls(
            inputs=tuple(EmbeddingInput(text=text) for text in texts),
            model=model,
            dimensions=dimensions,
            metadata=dict(metadata or {}),
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-embedding-request/v1",
            "model": self.model,
            "dimensions": self.dimensions,
            "input_count": len(self.inputs),
            "inputs": [item.manifest() for item in self.inputs],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EmbeddingVector:
    values: tuple[float, ...]
    index: int = 0
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "dimensions": len(self.values),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EmbeddingResponse:
    vectors: tuple[EmbeddingVector, ...]
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-embedding-response/v1",
            "model": self.model,
            "vector_count": len(self.vectors),
            "vectors": [vector.manifest() for vector in self.vectors],
            "metadata": dict(self.metadata),
        }


class EmbeddingProviderPort(Protocol):
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Return vectors for one embedding request."""


@dataclass(frozen=True)
class EmbeddingProviderSpec:
    name: str
    models: tuple[str, ...] = ()
    default_model: str = ""
    priority: int = 0
    dimensions: int = 0
    max_inputs: int | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports(self, request: EmbeddingRequest) -> bool:
        model = request.model or self.default_model
        if model and self.models and model not in self.models:
            return False
        if self.max_inputs is not None and len(request.inputs) > self.max_inputs:
            return False
        if request.dimensions and self.dimensions and request.dimensions != self.dimensions:
            return False
        return True

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-embedding-provider-spec/v1",
            "name": self.name,
            "models": list(self.models),
            "default_model": self.default_model,
            "priority": self.priority,
            "dimensions": self.dimensions,
            "max_inputs": self.max_inputs,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EmbeddingRoute:
    provider_name: str
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-embedding-route/v1",
            "provider_name": self.provider_name,
            "model": self.model,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EmbeddingCallRecord:
    provider_name: str
    model: str
    input_count: int
    dimensions: int
    status: str = "completed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-embedding-call/v1",
            "provider_name": self.provider_name,
            "model": self.model,
            "input_count": self.input_count,
            "dimensions": self.dimensions,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


class EmbeddingProviderNotFoundError(KeyError):
    pass


@dataclass(frozen=True)
class _EmbeddingProviderEntry:
    spec: EmbeddingProviderSpec
    provider: EmbeddingProviderPort


class EmbeddingProviderCenter(EmbeddingProviderPort):
    """Registry and router for provider-neutral embedding calls."""

    def __init__(self, *, default_provider: str = "", default_model: str = "") -> None:
        self.default_provider = default_provider
        self.default_model = default_model
        self._providers: dict[str, _EmbeddingProviderEntry] = {}
        self._calls: list[EmbeddingCallRecord] = []

    def register(
        self,
        name: str,
        provider: EmbeddingProviderPort,
        *,
        models: tuple[str, ...] = (),
        default_model: str = "",
        priority: int = 0,
        dimensions: int = 0,
        max_inputs: int | None = None,
        tags: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not name:
            raise ValueError("embedding provider name is required")
        self._providers[name] = _EmbeddingProviderEntry(
            spec=EmbeddingProviderSpec(
                name=name,
                models=tuple(models),
                default_model=default_model,
                priority=priority,
                dimensions=dimensions,
                max_inputs=max_inputs,
                tags=tuple(tags),
                metadata=dict(metadata or {}),
            ),
            provider=provider,
        )

    def get(self, name: str) -> EmbeddingProviderPort:
        entry = self._providers.get(name)
        if entry is None:
            raise EmbeddingProviderNotFoundError(name)
        return entry.provider

    def specs(self) -> tuple[EmbeddingProviderSpec, ...]:
        return tuple(entry.spec for entry in self._ordered_entries())

    def search(self, query: str = "", *, tag: str = "", model: str = "") -> tuple[EmbeddingProviderSpec, ...]:
        terms = tuple(part.casefold() for part in query.split() if part)
        matches: list[EmbeddingProviderSpec] = []
        for entry in self._ordered_entries():
            spec = entry.spec
            if tag and tag not in spec.tags:
                continue
            if model and spec.models and model not in spec.models:
                continue
            haystack = " ".join((spec.name, *spec.models, *spec.tags)).casefold()
            if terms and not all(term in haystack for term in terms):
                continue
            matches.append(spec)
        return tuple(matches)

    def select(self, request: EmbeddingRequest) -> EmbeddingRoute:
        entry = self._select_entry(request)
        model = request.model or entry.spec.default_model or self.default_model
        return EmbeddingRoute(
            provider_name=entry.spec.name,
            model=model,
            metadata={"provider_spec": entry.spec.manifest()},
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        entry = self._select_entry(request)
        routed = self._route_request(request, entry)
        try:
            response = await entry.provider.embed(routed)
        except Exception as exc:
            self._record_call(
                entry.spec.name,
                routed.model,
                len(routed.inputs),
                routed.dimensions or entry.spec.dimensions,
                status="failed",
                metadata={"error": str(exc)},
            )
            raise
        dimensions = _response_dimensions(response)
        self._record_call(
            entry.spec.name,
            routed.model,
            len(routed.inputs),
            dimensions,
            metadata={"request": routed.manifest(), "response": response.manifest()},
        )
        return replace(
            response,
            model=response.model or routed.model,
            metadata={
                **response.metadata,
                "embedding_provider": entry.spec.name,
                "embedding_route": self.select(routed).manifest(),
            },
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-embedding-provider-center/v1",
            "default_provider": self.default_provider,
            "default_model": self.default_model,
            "providers": [spec.manifest() for spec in self.specs()],
            "call_count": len(self._calls),
            "calls": [call.manifest() for call in self._calls],
        }

    def _ordered_entries(self) -> tuple[_EmbeddingProviderEntry, ...]:
        return tuple(
            sorted(
                self._providers.values(),
                key=lambda entry: (-entry.spec.priority, entry.spec.name),
            )
        )

    def _select_entry(self, request: EmbeddingRequest) -> _EmbeddingProviderEntry:
        requested = str(request.metadata.get("provider") or self.default_provider or "")
        if requested:
            entry = self._providers.get(requested)
            if entry is None or not entry.spec.supports(request):
                raise EmbeddingProviderNotFoundError(requested)
            return entry
        for entry in self._ordered_entries():
            if entry.spec.supports(request):
                return entry
        raise EmbeddingProviderNotFoundError(request.model or "<default>")

    def _route_request(
        self,
        request: EmbeddingRequest,
        entry: _EmbeddingProviderEntry,
    ) -> EmbeddingRequest:
        model = request.model or entry.spec.default_model or self.default_model
        dimensions = request.dimensions or entry.spec.dimensions
        return replace(
            request,
            model=model,
            dimensions=dimensions,
            metadata={
                **request.metadata,
                "embedding_provider": entry.spec.name,
                "embedding_provider_spec": entry.spec.manifest(),
            },
        )

    def _record_call(
        self,
        provider_name: str,
        model: str,
        input_count: int,
        dimensions: int,
        *,
        status: str = "completed",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._calls.append(
            EmbeddingCallRecord(
                provider_name=provider_name,
                model=model,
                input_count=input_count,
                dimensions=dimensions,
                status=status,
                metadata=dict(metadata or {}),
            )
        )


class DeterministicEmbeddingProvider(EmbeddingProviderPort):
    """Dependency-free embedding provider for tests and lightweight ranking."""

    def __init__(self, *, default_dimensions: int = 64, name: str = "deterministic") -> None:
        self.default_dimensions = max(1, int(default_dimensions))
        self.name = name

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        dimensions = max(1, request.dimensions or self.default_dimensions)
        vectors = tuple(
            EmbeddingVector(
                values=deterministic_text_embedding(item.text, dimensions=dimensions),
                index=index,
                name=item.name,
                metadata={"text_bytes": len(item.text.encode("utf-8"))},
            )
            for index, item in enumerate(request.inputs)
        )
        return EmbeddingResponse(
            vectors=vectors,
            model=request.model or self.name,
            metadata={"provider": self.name, "dimensions": dimensions},
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-deterministic-embedding-provider/v1",
            "default_dimensions": self.default_dimensions,
            "name": self.name,
        }


@dataclass(frozen=True)
class SemanticSearchHit(Generic[T]):
    item: T
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-semantic-search-hit/v1",
            "score": self.score,
            "metadata": dict(self.metadata),
        }


async def rank_semantic_documents(
    query: str,
    documents: tuple[SearchDocument[T], ...],
    *,
    provider: EmbeddingProviderPort,
    model: str = "",
    dimensions: int = 0,
    limit: int = 8,
) -> tuple[SemanticSearchHit[T], ...]:
    if not query or not documents:
        return ()
    inputs = (EmbeddingInput(text=query, name="query"),) + tuple(
        EmbeddingInput(text=document.text, name=document.name)
        for document in documents
    )
    response = await provider.embed(
        EmbeddingRequest(inputs=inputs, model=model, dimensions=dimensions)
    )
    if len(response.vectors) < 2:
        return ()
    query_vector = response.vectors[0].values
    scored: list[tuple[float, int, str, T]] = []
    for document, vector in zip(documents, response.vectors[1:], strict=False):
        score = cosine_similarity(query_vector, vector.values)
        if score <= 0:
            continue
        scored.append((score, document.priority, document.name, document.item))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    if limit <= 0:
        limit = len(scored)
    return tuple(
        SemanticSearchHit(item=item, score=score, metadata={"rank": index + 1})
        for index, (score, _, _, item) in enumerate(scored[:limit])
    )


def deterministic_text_embedding(text: str, *, dimensions: int = 64) -> tuple[float, ...]:
    dimensions = max(1, int(dimensions))
    values = [0.0] * dimensions
    for token in tokenize(text):
        digest = sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        values[index] += sign * (1.0 + min(len(token), 24) / 24.0)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return tuple(values)
    return tuple(value / norm for value in values)


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[index] * right[index] for index in range(size))
    left_norm = math.sqrt(sum(value * value for value in left[:size]))
    right_norm = math.sqrt(sum(value * value for value in right[:size]))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _response_dimensions(response: EmbeddingResponse) -> int:
    if not response.vectors:
        return 0
    return len(response.vectors[0].values)
