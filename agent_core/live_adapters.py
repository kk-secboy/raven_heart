"""Optional live provider adapters used by SDK smoke and e2e runs.

These adapters intentionally stay small and dependency-free. They are not a
new core runtime; they bridge the existing provider-neutral ports to
OpenAI-compatible chat completions and Zhipu Embedding-3 HTTP endpoints.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from agent_core.embeddings import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
)
from agent_core.providers import (
    ChatCompletionsLLMProviderCodec,
    LLMProviderError,
    LLMProviderPort,
    LLMRequest,
    LLMResponse,
    RetryHint,
    TransportLLMProvider,
)


@dataclass(frozen=True)
class LiveHTTPCallRecord:
    endpoint: str
    status: str
    request_bytes: int = 0
    response_bytes: int = 0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-live-http-call/v1",
            "endpoint": self.endpoint,
            "status": self.status,
            "request_bytes": self.request_bytes,
            "response_bytes": self.response_bytes,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class HTTPJsonTransport:
    """Small blocking-HTTP transport wrapped through asyncio.to_thread."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        timeout_seconds: float = 60.0,
        headers: dict[str, str] | None = None,
        name: str = "",
    ) -> None:
        if not endpoint:
            raise ValueError("endpoint is required")
        if not api_key:
            raise ValueError("api_key is required")
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.headers = dict(headers or {})
        self.name = name or endpoint
        self.calls: list[LiveHTTPCallRecord] = []

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            response_bytes = await asyncio.to_thread(self._post, body)
        except Exception as exc:
            self.calls.append(
                LiveHTTPCallRecord(
                    endpoint=self.endpoint,
                    status="failed",
                    request_bytes=len(body),
                    error=str(exc),
                )
            )
            raise
        self.calls.append(
            LiveHTTPCallRecord(
                endpoint=self.endpoint,
                status="completed",
                request_bytes=len(body),
                response_bytes=len(response_bytes),
            )
        )
        return json.loads(response_bytes.decode("utf-8"))

    def _post(self, body: bytes) -> bytes:
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self.headers,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="ignore")
            raise LLMProviderError(
                f"HTTP {exc.code} from {self.name}: {_shrink(payload, 1200)}",
                retry_hint=RetryHint(retryable=exc.code in {408, 409, 425, 429, 500, 502, 503, 504}),
            ) from exc

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-http-json-transport/v1",
            "endpoint": self.endpoint,
            "name": self.name,
            "timeout_seconds": self.timeout_seconds,
            "has_api_key": bool(self.api_key),
            "call_count": len(self.calls),
            "calls": [call.manifest() for call in self.calls],
        }


class DeepSeekChatProvider(LLMProviderPort):
    """DeepSeek V4/OpenAI-compatible chat completions provider."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "deepseek-v4-pro",
        endpoint: str = "https://api.deepseek.com/chat/completions",
        timeout_seconds: float = 90.0,
    ) -> None:
        self.model = model
        self.transport = HTTPJsonTransport(
            endpoint=endpoint,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            name="deepseek",
        )
        self.provider = TransportLLMProvider(
            self.transport,
            codec=ChatCompletionsLLMProviderCodec(),
            name="deepseek",
            metadata={"model": model},
        )
        self.requests: list[LLMRequest] = []
        self.responses: list[LLMResponse] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        routed = _request_with_default_model(request, self.model)
        self.requests.append(routed)
        response = await self.provider.complete(routed)
        self.responses.append(response)
        return response

    async def stream(self, request: LLMRequest):
        routed = _request_with_default_model(request, self.model)
        self.requests.append(routed)
        async for event in self.provider.stream(routed):
            yield event

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-deepseek-chat-provider/v1",
            "model": self.model,
            "request_count": len(self.requests),
            "response_count": len(self.responses),
            "transport": self.transport.manifest(),
        }


class ZhipuEmbedding3Provider:
    """Zhipu/BigModel Embedding-3 provider."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "embedding-3",
        endpoint: str = "https://open.bigmodel.cn/api/paas/v4/embeddings",
        default_dimensions: int = 512,
        max_inputs_per_request: int = 4,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.default_dimensions = max(0, int(default_dimensions))
        self.max_inputs_per_request = max(1, int(max_inputs_per_request))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.calls: list[LiveHTTPCallRecord] = []

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model = request.model or self.model
        dimensions = max(0, int(request.dimensions or self.default_dimensions))
        vectors: list[EmbeddingVector] = []
        response_model = model
        usage: dict[str, Any] = {}
        for offset in range(0, len(request.inputs), self.max_inputs_per_request):
            chunk = request.inputs[offset : offset + self.max_inputs_per_request]
            payload = await self._embed_chunk(chunk, model=model, dimensions=dimensions)
            response_model = str(payload.get("model") or response_model)
            usage = _merge_usage(usage, dict(payload.get("usage") or {}))
            for index, item in enumerate(payload.get("data") or ()):
                if not isinstance(item, dict):
                    continue
                local_index = int(item.get("index") or index)
                request_index = offset + local_index
                input_item = request.inputs[request_index] if request_index < len(request.inputs) else None
                vectors.append(
                    EmbeddingVector(
                        values=tuple(float(value) for value in item.get("embedding") or ()),
                        index=request_index,
                        name=input_item.name if input_item is not None else "",
                        metadata={"object": str(item.get("object") or "")},
                    )
                )
        return EmbeddingResponse(
            vectors=tuple(sorted(vectors, key=lambda item: item.index)),
            model=response_model,
            metadata={
                "provider": "zhipu",
                "usage": usage,
                "dimensions": len(vectors[0].values) if vectors else dimensions,
                "max_inputs_per_request": self.max_inputs_per_request,
            },
        )

    async def _embed_chunk(
        self,
        inputs: tuple[Any, ...],
        *,
        model: str,
        dimensions: int,
    ) -> dict[str, Any]:
        texts = [item.text for item in inputs]
        payload: dict[str, Any] = {
            "model": model,
            "input": texts[0] if len(texts) == 1 else texts,
        }
        if dimensions:
            payload["dimensions"] = dimensions
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            response_bytes = await asyncio.to_thread(self._post, body)
        except Exception as exc:
            self.calls.append(
                LiveHTTPCallRecord(
                    endpoint=self.endpoint,
                    status="failed",
                    request_bytes=len(body),
                    error=str(exc),
                    metadata={"model": model, "input_count": len(texts), "dimensions": dimensions},
                )
            )
            raise
        self.calls.append(
            LiveHTTPCallRecord(
                endpoint=self.endpoint,
                status="completed",
                request_bytes=len(body),
                response_bytes=len(response_bytes),
                metadata={"model": model, "input_count": len(texts), "dimensions": dimensions},
            )
        )
        return json.loads(response_bytes.decode("utf-8"))

    def _post(self, body: bytes) -> bytes:
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"HTTP {exc.code} from zhipu embedding: {_shrink(payload, 1200)}") from exc

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "agent-core-zhipu-embedding-3-provider/v1",
            "model": self.model,
            "endpoint": self.endpoint,
            "default_dimensions": self.default_dimensions,
            "max_inputs_per_request": self.max_inputs_per_request,
            "has_api_key": bool(self.api_key),
            "call_count": len(self.calls),
            "calls": [call.manifest() for call in self.calls],
        }


def _request_with_default_model(request: LLMRequest, model: str) -> LLMRequest:
    if request.model:
        return request
    from dataclasses import replace

    return replace(request, model=model)


def _shrink(text: str, limit: int) -> str:
    collapsed = " ".join(str(text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: max(0, limit - 3)].rstrip() + "..."


def _merge_usage(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    for key, value in right.items():
        if isinstance(value, (int, float)) and isinstance(merged.get(key), (int, float)):
            merged[key] = merged[key] + value
        elif key not in merged:
            merged[key] = value
    return merged
